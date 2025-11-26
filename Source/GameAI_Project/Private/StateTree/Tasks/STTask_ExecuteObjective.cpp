// Copyright Epic Games, Inc. All Rights Reserved.

#include "StateTree/Tasks/STTask_ExecuteObjective.h"
#include "StateTree/FollowerStateTreeContext.h"
#include "StateTree/FollowerStateTreeComponent.h"
#include "Team/FollowerAgentComponent.h"
#include "Team/Objective.h"
#include "Combat/WeaponComponent.h"
#include "Combat/HealthComponent.h"
#include "AIController.h"
#include "GameFramework/Pawn.h"
#include "GameFramework/CharacterMovementComponent.h"
#include "RL/RLPolicyNetwork.h"
#include "DrawDebugHelpers.h"

EStateTreeRunStatus FSTTask_ExecuteObjective::EnterState(FStateTreeExecutionContext& Context, const FStateTreeTransitionResult& Transition) const
{
	FInstanceDataType& InstanceData = Context.GetInstanceData(*this);

	if (!InstanceData.StateTreeComp)
	{
		UE_LOG(LogTemp, Error, TEXT("STTask_ExecuteObjective: StateTreeComp is null"));
		return EStateTreeRunStatus::Failed;
	}

	FFollowerStateTreeContext& SharedContext = InstanceData.StateTreeComp->GetSharedContext();

	if (!SharedContext.FollowerComponent || !SharedContext.AIController)
	{
		UE_LOG(LogTemp, Error, TEXT("STTask_ExecuteObjective: Missing component/controller"));
		return EStateTreeRunStatus::Failed;
	}

	APawn* Pawn = SharedContext.AIController->GetPawn();
	FString PawnName = Pawn ? Pawn->GetName() : TEXT("Unknown");
	FString ObjectiveName = SharedContext.CurrentObjective
		? UEnum::GetValueAsString(SharedContext.CurrentObjective->Type)
		: TEXT("None");

	UE_LOG(LogTemp, Warning, TEXT("🎯 [EXEC OBJ] '%s': ENTER - Objective: %s, Health: %.1f%%"),
		*PawnName, *ObjectiveName, SharedContext.CurrentObservation.AgentHealth);

	SharedContext.TimeInTacticalAction = 0.0f;
	SharedContext.ActionProgress = 0.0f;

	return EStateTreeRunStatus::Running;
}

EStateTreeRunStatus FSTTask_ExecuteObjective::Tick(FStateTreeExecutionContext& Context, float DeltaTime) const
{
	FInstanceDataType& InstanceData = Context.GetInstanceData(*this);
	FFollowerStateTreeContext& SharedContext = InstanceData.StateTreeComp->GetSharedContext();

	// Check abort conditions
	if (!SharedContext.bIsAlive || !SharedContext.CurrentObjective)
	{
		return EStateTreeRunStatus::Succeeded;
	}

	// Check if objective completed or failed
	if (CheckObjectiveStatus(Context))
	{
		return EStateTreeRunStatus::Succeeded;
	}

	SharedContext.TimeInTacticalAction += DeltaTime;

	// Execute atomic action from policy
	ExecuteAtomicAction(Context, DeltaTime);

	// Calculate and provide reward
	float Reward = CalculateObjectiveReward(Context, DeltaTime);
	if (Reward != 0.0f && SharedContext.FollowerComponent)
	{
		SharedContext.FollowerComponent->ProvideReward(Reward, false);
	}

	return EStateTreeRunStatus::Running;
}

void FSTTask_ExecuteObjective::ExitState(FStateTreeExecutionContext& Context, const FStateTreeTransitionResult& Transition) const
{
	FInstanceDataType& InstanceData = Context.GetInstanceData(*this);
	FFollowerStateTreeContext& SharedContext = InstanceData.StateTreeComp->GetSharedContext();

	UE_LOG(LogTemp, Log, TEXT("STTask_ExecuteObjective: Exit (time: %.1fs)"),
		SharedContext.TimeInTacticalAction);
}

void FSTTask_ExecuteObjective::ExecuteAtomicAction(FStateTreeExecutionContext& Context, float DeltaTime) const
{
	FInstanceDataType& InstanceData = Context.GetInstanceData(*this);
	FFollowerStateTreeContext& SharedContext = InstanceData.StateTreeComp->GetSharedContext();

	// Query RL policy for atomic action
	FTacticalAction RawAction;
	if (SharedContext.TacticalPolicy && SharedContext.CurrentObjective)
	{
		// Get action with objective context and mask
		RawAction = SharedContext.TacticalPolicy->GetActionWithMask(
			SharedContext.CurrentObservation,
			SharedContext.CurrentObjective,
			SharedContext.ActionMask);
	}
	else
	{
		UE_LOG(LogTemp, Warning, TEXT("STTask_ExecuteObjective: No policy or objective, using defaults"));
		RawAction = FTacticalAction(); // Default action
	}

	// Apply spatial constraints
	FTacticalAction MaskedAction = ApplyMask(RawAction, SharedContext.ActionMask);

	// Store in context for experience collection
	SharedContext.CurrentAtomicAction = MaskedAction;

	// Execute components of the action
	ExecuteMovement(Context, MaskedAction, DeltaTime);
	ExecuteAiming(Context, MaskedAction, DeltaTime);
	ExecuteFire(Context, MaskedAction);
	ExecuteCrouch(Context, MaskedAction);
	ExecuteAbility(Context, MaskedAction);
}

void FSTTask_ExecuteObjective::ExecuteMovement(FStateTreeExecutionContext& Context, const FTacticalAction& Action, float DeltaTime) const
{
	FInstanceDataType& InstanceData = Context.GetInstanceData(*this);
	FFollowerStateTreeContext& SharedContext = InstanceData.StateTreeComp->GetSharedContext();

	APawn* Pawn = SharedContext.AIController ? SharedContext.AIController->GetPawn() : nullptr;
	if (!Pawn)
	{
		return;
	}

	// Apply movement direction and speed
	FVector2D MoveDir = Action.MoveDirection;
	float MoveSpeed = Action.MoveSpeed;

	if (MoveDir.SizeSquared() > 0.01f) // Non-zero movement
	{
		// Convert 2D direction to world direction (relative to current rotation)
		FRotator CurrentRotation = Pawn->GetActorRotation();
		FVector ForwardDir = FRotationMatrix(CurrentRotation).GetUnitAxis(EAxis::X);
		FVector RightDir = FRotationMatrix(CurrentRotation).GetUnitAxis(EAxis::Y);

		FVector WorldMoveDir = (ForwardDir * MoveDir.X + RightDir * MoveDir.Y).GetSafeNormal();

		// Set movement destination
		FVector CurrentLocation = Pawn->GetActorLocation();
		FVector TargetLocation = CurrentLocation + WorldMoveDir * 500.0f; // 5m ahead

		// Apply speed to movement component
		if (UCharacterMovementComponent* MovementComp = Pawn->FindComponentByClass<UCharacterMovementComponent>())
		{
			float BaseSpeed = 600.0f;
			MovementComp->MaxWalkSpeed = BaseSpeed * MoveSpeed * InstanceData.MovementSpeedMultiplier;
		}

		// Move using AI controller
		if (SharedContext.AIController)
		{
			SharedContext.AIController->MoveToLocation(TargetLocation, 50.0f);
			SharedContext.MovementDestination = TargetLocation;
			SharedContext.bIsMoving = true;
		}
	}
	else
	{
		// Stop movement
		if (SharedContext.AIController)
		{
			SharedContext.AIController->StopMovement();
			SharedContext.bIsMoving = false;
		}
	}
}

void FSTTask_ExecuteObjective::ExecuteAiming(FStateTreeExecutionContext& Context, const FTacticalAction& Action, float DeltaTime) const
{
	FInstanceDataType& InstanceData = Context.GetInstanceData(*this);
	FFollowerStateTreeContext& SharedContext = InstanceData.StateTreeComp->GetSharedContext();

	APawn* Pawn = SharedContext.AIController ? SharedContext.AIController->GetPawn() : nullptr;
	if (!Pawn)
	{
		return;
	}

	// Convert 2D look direction to rotation
	FVector2D LookDir = Action.LookDirection;

	if (LookDir.SizeSquared() > 0.01f)
	{
		// LookDir.X = Yaw [-1,1], LookDir.Y = Pitch [-1,1]
		float TargetYaw = LookDir.X * 180.0f; // Convert to degrees
		float TargetPitch = LookDir.Y * 45.0f; // Limit pitch range

		FRotator CurrentRotation = Pawn->GetActorRotation();
		FRotator TargetRotation = FRotator(TargetPitch, TargetYaw, 0.0f);

		// Interpolate rotation
		float RotSpeed = InstanceData.RotationSpeed;
		FRotator NewRotation = FMath::RInterpTo(CurrentRotation, TargetRotation, DeltaTime, RotSpeed / 180.0f);

		Pawn->SetActorRotation(NewRotation);
	}
	else if (SharedContext.PrimaryTarget)
	{
		// Default: look at primary target
		SharedContext.AIController->SetFocus(SharedContext.PrimaryTarget);
	}
}

void FSTTask_ExecuteObjective::ExecuteFire(FStateTreeExecutionContext& Context, const FTacticalAction& Action) const
{
	FInstanceDataType& InstanceData = Context.GetInstanceData(*this);
	FFollowerStateTreeContext& SharedContext = InstanceData.StateTreeComp->GetSharedContext();

	if (!Action.bFire)
	{
		return;
	}

	APawn* Pawn = SharedContext.AIController ? SharedContext.AIController->GetPawn() : nullptr;
	if (!Pawn)
	{
		return;
	}

	UWeaponComponent* WeaponComp = Pawn->FindComponentByClass<UWeaponComponent>();
	if (!WeaponComp || !WeaponComp->CanFire())
	{
		return;
	}

	// Fire at primary target if available and in LOS
	if (SharedContext.PrimaryTarget && SharedContext.bHasLOS)
	{
		WeaponComp->FireAtTarget(SharedContext.PrimaryTarget, true);

		UE_LOG(LogTemp, Log, TEXT("[EXEC OBJ] '%s': Firing at '%s'"),
			*Pawn->GetName(), *SharedContext.PrimaryTarget->GetName());
	}
}

void FSTTask_ExecuteObjective::ExecuteCrouch(FStateTreeExecutionContext& Context, const FTacticalAction& Action) const
{
	FInstanceDataType& InstanceData = Context.GetInstanceData(*this);
	FFollowerStateTreeContext& SharedContext = InstanceData.StateTreeComp->GetSharedContext();

	APawn* Pawn = SharedContext.AIController ? SharedContext.AIController->GetPawn() : nullptr;
	if (!Pawn)
	{
		return;
	}

	// Toggle crouch via movement component
	if (UCharacterMovementComponent* MovementComp = Pawn->FindComponentByClass<UCharacterMovementComponent>())
	{
		if (Action.bCrouch && !MovementComp->IsCrouching())
		{
			MovementComp->bWantsToCrouch = true;
		}
		else if (!Action.bCrouch && MovementComp->IsCrouching())
		{
			MovementComp->bWantsToCrouch = false;
		}
	}
}

void FSTTask_ExecuteObjective::ExecuteAbility(FStateTreeExecutionContext& Context, const FTacticalAction& Action) const
{
	if (!Action.bUseAbility)
	{
		return;
	}

	// Placeholder for ability system integration
	UE_LOG(LogTemp, Log, TEXT("STTask_ExecuteObjective: Ability %d requested (not implemented)"), Action.AbilityID);
}

FTacticalAction FSTTask_ExecuteObjective::ApplyMask(const FTacticalAction& RawAction, const FActionSpaceMask& Mask) const
{
	FTacticalAction MaskedAction = RawAction;

	// Apply movement constraints
	if (Mask.bLockMovementX)
	{
		MaskedAction.MoveDirection.X = 0.0f;
	}
	if (Mask.bLockMovementY)
	{
		MaskedAction.MoveDirection.Y = 0.0f;
	}

	// Clamp speed
	MaskedAction.MoveSpeed = FMath::Clamp(MaskedAction.MoveSpeed, 0.0f, Mask.MaxSpeed);

	// Apply aiming constraints (LookDirection is normalized [-1,1], convert to angles for clamping)
	float Yaw = MaskedAction.LookDirection.X * 180.0f;
	float Pitch = MaskedAction.LookDirection.Y * 90.0f;

	Yaw = FMath::Clamp(Yaw, Mask.MinYaw, Mask.MaxYaw);
	Pitch = FMath::Clamp(Pitch, Mask.MinPitch, Mask.MaxPitch);

	MaskedAction.LookDirection.X = Yaw / 180.0f;
	MaskedAction.LookDirection.Y = Pitch / 90.0f;

	// Force crouch if required
	if (Mask.bForceCrouch)
	{
		MaskedAction.bCrouch = true;
	}

	// Safety lock prevents firing
	if (Mask.bSafetyLock)
	{
		MaskedAction.bFire = false;
	}

	return MaskedAction;
}

float FSTTask_ExecuteObjective::CalculateObjectiveReward(FStateTreeExecutionContext& Context, float DeltaTime) const
{
	FInstanceDataType& InstanceData = Context.GetInstanceData(*this);
	FFollowerStateTreeContext& SharedContext = InstanceData.StateTreeComp->GetSharedContext();

	if (!SharedContext.CurrentObjective)
	{
		return 0.0f;
	}

	float Reward = 0.0f;

	// Small progress reward (encourage forward progress)
	float ObjProgress = SharedContext.CurrentObjective->GetProgress();
	if (ObjProgress > SharedContext.ActionProgress)
	{
		float ProgressDelta = ObjProgress - SharedContext.ActionProgress;
		Reward += ProgressDelta * 10.0f; // +10 reward per 100% progress
		SharedContext.ActionProgress = ObjProgress;
	}

	// Penalty for time inefficiency (encourage fast completion)
	if (SharedContext.CurrentObjective->TimeLimit > 0.0f)
	{
		float TimeRatio = SharedContext.CurrentObjective->TimeRemaining / SharedContext.CurrentObjective->TimeLimit;
		if (TimeRatio < 0.3f) // Less than 30% time remaining
		{
			Reward -= 0.5f * DeltaTime; // Small time penalty
		}
	}

	return Reward;
}

bool FSTTask_ExecuteObjective::CheckObjectiveStatus(FStateTreeExecutionContext& Context) const
{
	FInstanceDataType& InstanceData = Context.GetInstanceData(*this);
	FFollowerStateTreeContext& SharedContext = InstanceData.StateTreeComp->GetSharedContext();

	if (!SharedContext.CurrentObjective)
	{
		return true; // No objective = exit
	}

	UObjective* Obj = SharedContext.CurrentObjective;

	// Check completion
	if (Obj->IsCompleted())
	{
		UE_LOG(LogTemp, Warning, TEXT("[EXEC OBJ] Objective COMPLETED"));

		// Provide completion reward
		if (SharedContext.FollowerComponent)
		{
			SharedContext.FollowerComponent->ProvideReward(50.0f, false); // Major reward
		}

		return true;
	}

	// Check failure
	if (Obj->IsFailed())
	{
		UE_LOG(LogTemp, Warning, TEXT("[EXEC OBJ] Objective FAILED"));

		// Provide failure penalty
		if (SharedContext.FollowerComponent)
		{
			SharedContext.FollowerComponent->ProvideReward(-30.0f, false); // Major penalty
		}

		return true;
	}

	return false; // Still active
}
