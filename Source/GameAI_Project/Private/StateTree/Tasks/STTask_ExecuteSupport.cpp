// Copyright Epic Games, Inc. All Rights Reserved.

#include "StateTree/Tasks/STTask_ExecuteSupport.h"
#include "StateTree/FollowerStateTreeContext.h"
#include "Team/FollowerAgentComponent.h"
#include "Combat/WeaponComponent.h"
#include "Combat/HealthComponent.h"
#include "AIController.h"
#include "GameFramework/Pawn.h"
#include "GameFramework/CharacterMovementComponent.h"
#include "DrawDebugHelpers.h"
#include "Navigation/PathFollowingComponent.h"

namespace
{
	// Helper to check if target actor is valid and alive
	bool IsTargetValid(AActor* Target)
	{
		if (!Target || !Target->IsValidLowLevel() || Target->IsPendingKillPending())
		{
			return false;
		}

		// Check if target has health component and is alive
		if (UHealthComponent* HealthComp = Target->FindComponentByClass<UHealthComponent>())
		{
			return HealthComp->IsAlive();
		}

		return true; // No health component, assume valid
	}

	// Helper to find nearest valid enemy from visible enemies
	AActor* FindNearestValidEnemy(const TArray<AActor*>& VisibleEnemies, APawn* FromPawn)
	{
		if (!FromPawn) return nullptr;

		FVector MyLocation = FromPawn->GetActorLocation();
		AActor* NearestEnemy = nullptr;
		float NearestDistance = FLT_MAX;

		for (AActor* Enemy : VisibleEnemies)
		{
			if (IsTargetValid(Enemy))
			{
				float Distance = FVector::Dist(MyLocation, Enemy->GetActorLocation());
				if (Distance < NearestDistance)
				{
					NearestDistance = Distance;
					NearestEnemy = Enemy;
				}
			}
		}

		return NearestEnemy;
	}
}

EStateTreeRunStatus FSTTask_ExecuteSupport::EnterState(FStateTreeExecutionContext& Context, const FStateTreeTransitionResult& Transition) const
{
	FInstanceDataType& InstanceData = Context.GetInstanceData(*this);

	// Validate inputs
	if (!InstanceData.Context.FollowerComponent || !InstanceData.Context.AIController)
	{
		UE_LOG(LogTemp, Error, TEXT("STTask_ExecuteSupport: Invalid inputs (missing component/controller)"));
		return EStateTreeRunStatus::Failed;
	}

	APawn* Pawn = InstanceData.Context.AIController->GetPawn();
	FString PawnName = Pawn ? Pawn->GetName() : TEXT("Unknown");
	FString TargetName = InstanceData.Context.PrimaryTarget ? InstanceData.Context.PrimaryTarget->GetName() : TEXT("None");
	FString CommandTargetName = InstanceData.Context.CurrentCommand.TargetActor ? InstanceData.Context.CurrentCommand.TargetActor->GetName() : TEXT("None");

	UE_LOG(LogTemp, Warning, TEXT("🎯 [SUPPORT TASK] '%s': ENTERED support state - Target: %s, CommandTarget: %s, VisibleEnemies: %d, Tactic: %s"),
		*PawnName,
		*TargetName,
		*CommandTargetName,
		InstanceData.Context.VisibleEnemies.Num(),
		*UEnum::GetValueAsString(InstanceData.Context.CurrentTacticalAction));

	// Reset timers
	InstanceData.TimeSinceLastRLQuery = 0.0f;
	InstanceData.Context.TimeInTacticalAction = 0.0f;
	InstanceData.Context.ActionProgress = 0.0f;

	return EStateTreeRunStatus::Running;
}

EStateTreeRunStatus FSTTask_ExecuteSupport::Tick(FStateTreeExecutionContext& Context, float DeltaTime) const
{
	FInstanceDataType& InstanceData = Context.GetInstanceData(*this);

	// Check if should abort
	if (!InstanceData.Context.bIsAlive || !InstanceData.Context.bIsCommandValid)
	{
		return EStateTreeRunStatus::Succeeded;
	}

	// Update timers
	InstanceData.TimeSinceLastRLQuery += DeltaTime;
	InstanceData.Context.TimeInTacticalAction += DeltaTime;

	// Re-query RL policy if interval elapsed
	if (InstanceData.RLQueryInterval > 0.0f && InstanceData.TimeSinceLastRLQuery >= InstanceData.RLQueryInterval)
	{
		// This would trigger a transition back to QueryRLPolicy state
		InstanceData.TimeSinceLastRLQuery = 0.0f;
	}

	// Execute current tactical action
	ExecuteTacticalAction(Context, DeltaTime);

	// Calculate and provide reward
	float Reward = 0.0f;

	// Reward for providing support
	if (InstanceData.Context.CurrentTacticalAction == ETacticalAction::ProvideCoveringFire ||
		InstanceData.Context.CurrentTacticalAction == ETacticalAction::SuppressiveFire)
	{
		if (InstanceData.Context.bHasLOS && InstanceData.Context.bWeaponReady)
		{
			Reward += FTacticalRewards::COVERING_FIRE * DeltaTime;
		}
	}

	// Reward for following support command
	if (InstanceData.Context.bIsCommandValid)
	{
		Reward += FTacticalRewards::FOLLOW_COMMAND * DeltaTime;
	}

	if (Reward != 0.0f && InstanceData.Context.FollowerComponent)
	{
		InstanceData.Context.FollowerComponent->ProvideReward(Reward * 0.5f, false);
	}

	return EStateTreeRunStatus::Running;
}

void FSTTask_ExecuteSupport::ExitState(FStateTreeExecutionContext& Context, const FStateTreeTransitionResult& Transition) const
{
	FInstanceDataType& InstanceData = Context.GetInstanceData(*this);

	UE_LOG(LogTemp, Log, TEXT("STTask_ExecuteSupport: Exiting support (time in action: %.1fs)"),
		InstanceData.Context.TimeInTacticalAction);
}

void FSTTask_ExecuteSupport::ExecuteTacticalAction(FStateTreeExecutionContext& Context, float DeltaTime) const
{
	FInstanceDataType& InstanceData = Context.GetInstanceData(*this);
	
	switch (InstanceData.Context.CurrentTacticalAction)
	{
	case ETacticalAction::ProvideCoveringFire:
		ExecuteProvideCoveringFire(Context, DeltaTime);
		break;

	case ETacticalAction::SuppressiveFire:
		ExecuteSuppressiveFire(Context, DeltaTime);
		break;

	case ETacticalAction::Reload:
		ExecuteReload(Context, DeltaTime);
		break;

	case ETacticalAction::UseAbility:
		ExecuteUseAbility(Context, DeltaTime);
		break;

	default:
		// Default to providing covering fire
		ExecuteProvideCoveringFire(Context, DeltaTime);
		break;
	}
}

void FSTTask_ExecuteSupport::ExecuteProvideCoveringFire(FStateTreeExecutionContext& Context, float DeltaTime) const
{
	FInstanceDataType& InstanceData = Context.GetInstanceData(*this);

	APawn* Pawn = InstanceData.Context.AIController ? InstanceData.Context.AIController->GetPawn() : nullptr;
	if (!Pawn) return;


	// Stay in position and provide covering fire
	if (InstanceData.Context.AIController)
	{
		InstanceData.Context.AIController->StopMovement();
	}

	// Determine target - validate primary target first
	AActor* CurrentTarget = nullptr;

	if (IsTargetValid(InstanceData.Context.PrimaryTarget))
	{
		CurrentTarget = InstanceData.Context.PrimaryTarget;
	}
	else if (InstanceData.Context.VisibleEnemies.Num() > 0)
	{
		// Primary target invalid/dead - find nearest valid enemy
		CurrentTarget = FindNearestValidEnemy(InstanceData.Context.VisibleEnemies, Pawn);

		// If still no valid target, rotate through visible enemies (they might not have health components)
		if (!CurrentTarget)
		{
			int32 TargetIndex = static_cast<int32>(InstanceData.Context.TimeInTacticalAction) % InstanceData.Context.VisibleEnemies.Num();
			if (InstanceData.Context.VisibleEnemies.IsValidIndex(TargetIndex))
			{
				AActor* CandidateTarget = InstanceData.Context.VisibleEnemies[TargetIndex];
				if (CandidateTarget && CandidateTarget->IsValidLowLevel() && !CandidateTarget->IsPendingKillPending())
				{
					CurrentTarget = CandidateTarget;
				}
			}
		}

		// Update primary target to the new valid enemy
		if (CurrentTarget)
		{
			InstanceData.Context.PrimaryTarget = CurrentTarget;
		}
	}

	if (CurrentTarget)
	{
		// Focus on target
		if (InstanceData.Context.AIController)
		{
			InstanceData.Context.AIController->SetFocus(CurrentTarget);
		}

		// Perform LOS check
		bool bHasLOS = false;
		UWorld* World = Pawn->GetWorld();
		if (World)
		{
			FHitResult HitResult;
			FCollisionQueryParams QueryParams;
			QueryParams.AddIgnoredActor(Pawn);

			FVector StartLoc = Pawn->GetActorLocation() + FVector(0, 0, 80.0f);
			FVector EndLoc = CurrentTarget->GetActorLocation() + FVector(0, 0, 80.0f);

			bool bHit = World->LineTraceSingleByChannel(HitResult, StartLoc, EndLoc, ECC_Visibility, QueryParams);
			bHasLOS = !bHit || HitResult.GetActor() == CurrentTarget;
		}

		// Fire weapon at target
		UWeaponComponent* WeaponComp = Pawn->FindComponentByClass<UWeaponComponent>();
		if (WeaponComp && WeaponComp->CanFire() && bHasLOS)
		{
			WeaponComp->FireAtTarget(CurrentTarget, true);
			UE_LOG(LogTemp, Log, TEXT("[SUPPORT TASK] '%s': Covering fire at '%s'"),
				*Pawn->GetName(),
				*CurrentTarget->GetName());
		}
	}
	else
	{
		// No valid targets - clear focus and idle (support mission complete for now)
		if (InstanceData.Context.AIController)
		{
			InstanceData.Context.AIController->ClearFocus(EAIFocusPriority::Gameplay);
		}
		InstanceData.Context.PrimaryTarget = nullptr;
		// Don't spam warnings - this is a normal state when all enemies are dead
	}

	InstanceData.Context.bIsMoving = false;
}

void FSTTask_ExecuteSupport::ExecuteSuppressiveFire(FStateTreeExecutionContext& Context, float DeltaTime) const
{
	FInstanceDataType& InstanceData = Context.GetInstanceData(*this);

	APawn* Pawn = InstanceData.Context.AIController ? InstanceData.Context.AIController->GetPawn() : nullptr;
	if (!Pawn) return;

	// Stop movement for suppression
	if (InstanceData.Context.AIController)
	{
		InstanceData.Context.AIController->StopMovement();
	}

	// Determine target - prioritize cycling through multiple valid enemies
	AActor* CurrentTarget = nullptr;

	if (InstanceData.Context.VisibleEnemies.Num() > 0)
	{
		// Build list of valid enemies only
		TArray<AActor*> ValidEnemies;
		for (AActor* Enemy : InstanceData.Context.VisibleEnemies)
		{
			if (IsTargetValid(Enemy))
			{
				ValidEnemies.Add(Enemy);
			}
		}

		if (ValidEnemies.Num() > 0)
		{
			// Rapidly cycle through valid targets for suppression
			int32 TargetIndex = (static_cast<int32>(InstanceData.Context.TimeInTacticalAction * 2.0f)) % ValidEnemies.Num();
			CurrentTarget = ValidEnemies[TargetIndex];
		}
	}

	// Fall back to primary target if no visible enemies
	if (!CurrentTarget && IsTargetValid(InstanceData.Context.PrimaryTarget))
	{
		CurrentTarget = InstanceData.Context.PrimaryTarget;
	}

	if (CurrentTarget)
	{
		// Focus on target
		if (InstanceData.Context.AIController)
		{
			InstanceData.Context.AIController->SetFocus(CurrentTarget);
		}

		// Perform LOS check
		bool bHasLOS = false;
		UWorld* World = Pawn->GetWorld();
		if (World)
		{
			FHitResult HitResult;
			FCollisionQueryParams QueryParams;
			QueryParams.AddIgnoredActor(Pawn);

			FVector StartLoc = Pawn->GetActorLocation() + FVector(0, 0, 80.0f);
			FVector EndLoc = CurrentTarget->GetActorLocation() + FVector(0, 0, 80.0f);

			bool bHit = World->LineTraceSingleByChannel(HitResult, StartLoc, EndLoc, ECC_Visibility, QueryParams);
			bHasLOS = !bHit || HitResult.GetActor() == CurrentTarget;
		}

		// Fire weapon - suppressive fire (no prediction for area spray)
		UWeaponComponent* WeaponComp = Pawn->FindComponentByClass<UWeaponComponent>();
		if (WeaponComp && WeaponComp->CanFire() && bHasLOS)
		{
			WeaponComp->FireAtTarget(CurrentTarget, false); // No prediction for suppressive effect
			UE_LOG(LogTemp, Log, TEXT("[SUPPORT TASK] '%s': Suppressive fire at '%s'"),
				*Pawn->GetName(),
				*CurrentTarget->GetName());
		}
	}
	else
	{
		// No valid targets - clear focus
		if (InstanceData.Context.AIController)
		{
			InstanceData.Context.AIController->ClearFocus(EAIFocusPriority::Gameplay);
		}
		InstanceData.Context.PrimaryTarget = nullptr;
	}

	InstanceData.Context.bIsMoving = false;
}

void FSTTask_ExecuteSupport::ExecuteReload(FStateTreeExecutionContext& Context, float DeltaTime) const
{
	FInstanceDataType& InstanceData = Context.GetInstanceData(*this);

	APawn* Pawn = InstanceData.Context.AIController ? InstanceData.Context.AIController->GetPawn() : nullptr;
	if (!Pawn) return;

	// Stop movement and seek cover if possible
	if (InstanceData.Context.AIController)
	{
		InstanceData.Context.AIController->StopMovement();
		InstanceData.Context.AIController->ClearFocus(EAIFocusPriority::Gameplay);
	}

	// If not in cover and under fire, try to get to cover
	if (!InstanceData.Context.bInCover && InstanceData.Context.bUnderFire)
	{
		if (InstanceData.Context.NearestCoverLocation != FVector::ZeroVector)
		{
			if (InstanceData.Context.AIController)
			{
				EPathFollowingRequestResult::Type MoveResult = InstanceData.Context.AIController->MoveToLocation(
					InstanceData.Context.NearestCoverLocation, 50.0f);

				// Log movement result
				if (MoveResult == EPathFollowingRequestResult::Failed)
				{
					UE_LOG(LogTemp, Error, TEXT("[SUPPORT TASK] '%s': MoveToLocation for cover FAILED"), *Pawn->GetName());
				}

				// Apply speed for cover-seeking
				if (UCharacterMovementComponent* MovementComp = Pawn->FindComponentByClass<UCharacterMovementComponent>())
				{
					float BaseSpeed = 600.0f;
					MovementComp->MaxWalkSpeed = BaseSpeed * 1.3f; // Fast movement to cover
				}
			}
			InstanceData.Context.bIsMoving = true;
			return;
		}
	}

	// Reload action progress (simulated 2-second reload)
	InstanceData.Context.ActionProgress = FMath::Min(1.0f, InstanceData.Context.TimeInTacticalAction / 2.0f);

	// Mark weapon as not ready during reload
	InstanceData.Context.bWeaponReady = (InstanceData.Context.ActionProgress >= 1.0f);
	InstanceData.Context.bIsMoving = false;

	UE_LOG(LogTemp, VeryVerbose, TEXT("STTask_ExecuteSupport: Reloading (%.1f%% complete)"),
		InstanceData.Context.ActionProgress * 100.0f);
}

void FSTTask_ExecuteSupport::ExecuteUseAbility(FStateTreeExecutionContext& Context, float DeltaTime) const
{
	FInstanceDataType& InstanceData = Context.GetInstanceData(*this);

	APawn* Pawn = InstanceData.Context.AIController ? InstanceData.Context.AIController->GetPawn() : nullptr;
	if (!Pawn) return;

	// Placeholder for ability usage
	// Specific ability logic would be implemented based on available abilities
	// For now, maintain position and signal ability usage intent

	if (InstanceData.Context.AIController)
	{
		InstanceData.Context.AIController->StopMovement();
	}

	// Simulate ability cooldown/activation (3-second ability)
	InstanceData.Context.ActionProgress = FMath::Min(1.0f, InstanceData.Context.TimeInTacticalAction / 3.0f);
	InstanceData.Context.bIsMoving = false;

	UE_LOG(LogTemp, VeryVerbose, TEXT("STTask_ExecuteSupport: Using ability (%.1f%% complete)"),
		InstanceData.Context.ActionProgress * 100.0f);
}
