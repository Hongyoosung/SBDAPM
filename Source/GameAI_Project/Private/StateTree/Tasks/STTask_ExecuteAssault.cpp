// Copyright Epic Games, Inc. All Rights Reserved.

#include "StateTree/Tasks/STTask_ExecuteAssault.h"
#include "StateTree/FollowerStateTreeContext.h"
#include "StateTree/FollowerStateTreeComponent.h"
#include "Team/FollowerAgentComponent.h"
#include "Combat/WeaponComponent.h"
#include "Combat/HealthComponent.h"
#include "AIController.h"
#include "NavigationSystem.h"
#include "GameFramework/Pawn.h"
#include "GameFramework/CharacterMovementComponent.h"
#include "DrawDebugHelpers.h"
#include "Kismet/GameplayStatics.h"
#include "RL/RLPolicyNetwork.h"
#include "Navigation/PathFollowingComponent.h"
#include "Utill/GameAIHelper.h"



EStateTreeRunStatus FSTTask_ExecuteAssault::EnterState(FStateTreeExecutionContext& Context, const FStateTreeTransitionResult& Transition) const
{
	FInstanceDataType& InstanceData = Context.GetInstanceData(*this);

	if (!InstanceData.StateTreeComp)
	{
		UE_LOG(LogTemp, Error, TEXT("STTask_ExecuteAssault: StateTreeComp is null!"));
		return EStateTreeRunStatus::Failed;
	}

	FFollowerStateTreeContext& SharedContext = InstanceData.StateTreeComp->GetSharedContext();

	// Validate inputs
	if (!SharedContext.FollowerComponent || !SharedContext.AIController)
	{
		UE_LOG(LogTemp, Error, TEXT("STTask_ExecuteAssault: Invalid inputs (missing component/controller)"));
		return EStateTreeRunStatus::Failed;
	}

	APawn* Pawn = SharedContext.AIController->GetPawn();
	FString PawnName = Pawn ? Pawn->GetName() : TEXT("Unknown");
	FString TargetName = SharedContext.PrimaryTarget ? SharedContext.PrimaryTarget->GetName() : TEXT("None");

	UE_LOG(LogTemp, Warning, TEXT("🎯 [ASSAULT TASK] '%s': ENTERED assault state - Target: %s, Tactic: %s, Health: %.1f%%, VisibleEnemies: %d (SHARED Context addr: %p)"),
		*PawnName,
		*TargetName,
		*UEnum::GetValueAsString(SharedContext.CurrentTacticalAction),
		SharedContext.CurrentObservation.AgentHealth,
		SharedContext.CurrentObservation.VisibleEnemyCount,
		&SharedContext);

	// Reset timers
	SharedContext.TimeInTacticalAction = 0.0f;
	SharedContext.ActionProgress = 0.0f;

	return EStateTreeRunStatus::Running;
}

EStateTreeRunStatus FSTTask_ExecuteAssault::Tick(FStateTreeExecutionContext& Context, float DeltaTime) const
{
	FInstanceDataType& InstanceData = Context.GetInstanceData(*this);
	FFollowerStateTreeContext& SharedContext = InstanceData.StateTreeComp->GetSharedContext();

	// Check if should abort
	if (!SharedContext.bIsAlive || !SharedContext.bIsCommandValid)
	{
		return EStateTreeRunStatus::Succeeded;
	}

	// Update timers
	SharedContext.TimeInTacticalAction += DeltaTime;

	// Execute current tactical action
	ExecuteTacticalAction(Context, DeltaTime);

	// Calculate and provide reward
	float Reward = CalculateAssaultReward(Context, DeltaTime);
	if (Reward != 0.0f && SharedContext.FollowerComponent)
	{
		SharedContext.FollowerComponent->ProvideReward(Reward, false);
	}

	return EStateTreeRunStatus::Running;
}

void FSTTask_ExecuteAssault::ExitState(FStateTreeExecutionContext& Context, const FStateTreeTransitionResult& Transition) const
{
	FInstanceDataType& InstanceData = Context.GetInstanceData(*this);
	FFollowerStateTreeContext& SharedContext = InstanceData.StateTreeComp->GetSharedContext();

	UE_LOG(LogTemp, Log, TEXT("STTask_ExecuteAssault: Exiting assault (time in action: %.1fs)"),
		SharedContext.TimeInTacticalAction);
}

void FSTTask_ExecuteAssault::ExecuteTacticalAction(FStateTreeExecutionContext& Context, float DeltaTime) const
{
	FInstanceDataType& InstanceData = Context.GetInstanceData(*this);
	FFollowerStateTreeContext& SharedContext = InstanceData.StateTreeComp->GetSharedContext();

	switch (SharedContext.CurrentTacticalAction)
	{
	case ETacticalAction::AggressiveAssault:
		ExecuteAggressiveAssault(Context, DeltaTime);
		break;

	case ETacticalAction::CautiousAdvance:
		ExecuteCautiousAdvance(Context, DeltaTime);
		break;

	case ETacticalAction::FlankLeft:
		ExecuteFlankManeuver(Context, DeltaTime, true);
		break;

	case ETacticalAction::FlankRight:
		ExecuteFlankManeuver(Context, DeltaTime, false);
		break;

	case ETacticalAction::MaintainDistance:
		ExecuteMaintainDistance(Context, DeltaTime);
		break;

	case ETacticalAction::DefensiveHold:
		ExecuteDefensiveHold(Context, DeltaTime);
		break;

	case ETacticalAction::SeekCover:
		ExecuteSeekCover(Context, DeltaTime);
		break;

	case ETacticalAction::TacticalRetreat:
		ExecuteTacticalRetreat(Context, DeltaTime);
		break;

	case ETacticalAction::SuppressiveFire:
		ExecuteSuppressiveFire(Context, DeltaTime);
		break;

	default:
		// Default to aggressive assault for unhandled actions
		ExecuteAggressiveAssault(Context, DeltaTime);
		break;
	}
}

void FSTTask_ExecuteAssault::ExecuteAggressiveAssault(FStateTreeExecutionContext& Context, float DeltaTime) const
{
	FInstanceDataType& InstanceData = Context.GetInstanceData(*this);
	FFollowerStateTreeContext& SharedContext = InstanceData.StateTreeComp->GetSharedContext();

	APawn* Pawn = SharedContext.AIController ? SharedContext.AIController->GetPawn() : nullptr;
	if (!Pawn)
	{
		UE_LOG(LogTemp, Warning, TEXT("[ASSAULT TASK] AggressiveAssault: No pawn!"));
		return;
	}

	// Validate and update primary target if needed
	if (!UGameAIHelper::IsTargetValid(SharedContext.PrimaryTarget))
	{
		// Try to find a new target from visible enemies
		AActor* NewTarget = UGameAIHelper::FindNearestValidEnemy(SharedContext.VisibleEnemies, Pawn);
		if (NewTarget)
		{
			SharedContext.PrimaryTarget = NewTarget;
			UE_LOG(LogTemp, Log, TEXT("[ASSAULT TASK] '%s': Target died, switching to '%s'"),
				*Pawn->GetName(), *NewTarget->GetName());
		}
		else
		{
			SharedContext.PrimaryTarget = nullptr;
		}
	}

	// Move toward target aggressively
	if (SharedContext.PrimaryTarget)
	{
		FVector CurrentLocation = Pawn->GetActorLocation();
		FVector TargetLocation = SharedContext.PrimaryTarget->GetActorLocation();
		float Distance = FVector::Dist(CurrentLocation, TargetLocation);

		// Set high speed multiplier and apply to movement component
		SharedContext.MovementSpeedMultiplier = InstanceData.AggressiveSpeedMultiplier;

		// Apply speed multiplier to CharacterMovementComponent
		if (UCharacterMovementComponent* MovementComp = Pawn->FindComponentByClass<UCharacterMovementComponent>())
		{
			float BaseSpeed = 600.0f; // Default base speed
			MovementComp->MaxWalkSpeed = BaseSpeed * SharedContext.MovementSpeedMultiplier;
		}

		// Move directly toward target
		if (SharedContext.AIController)
		{
			EPathFollowingRequestResult::Type MoveResult = SharedContext.AIController->MoveToLocation(TargetLocation, 50.0f);

			// Log movement result for debugging
			if (MoveResult == EPathFollowingRequestResult::Failed)
			{
				UE_LOG(LogTemp, Error, TEXT("[ASSAULT TASK] '%s': MoveToLocation FAILED - NavMesh issue?"), *Pawn->GetName());
			}
			else if (MoveResult == EPathFollowingRequestResult::AlreadyAtGoal)
			{
				UE_LOG(LogTemp, Warning, TEXT("[ASSAULT TASK] '%s': Already at goal (dist=%.1f)"), *Pawn->GetName(), Distance);
			}

			SharedContext.AIController->SetFocus(SharedContext.PrimaryTarget);
			SharedContext.MovementDestination = TargetLocation;
		}
		else
		{
			UE_LOG(LogTemp, Error, TEXT("[ASSAULT TASK] '%s': No AIController!"),
				*Pawn->GetName());
		}

		// Perform LOS check directly (evaluator context is separate)
		bool bHasLOS = false;
		UWorld* World = Pawn->GetWorld();
		if (World)
		{
			FHitResult HitResult;
			FCollisionQueryParams QueryParams;
			QueryParams.AddIgnoredActor(Pawn);

			FVector StartLoc = CurrentLocation + FVector(0, 0, 80.0f); // Eye height
			FVector EndLoc = TargetLocation + FVector(0, 0, 80.0f);

			bool bHit = World->LineTraceSingleByChannel(
				HitResult,
				StartLoc,
				EndLoc,
				ECC_Visibility,
				QueryParams
			);

			bHasLOS = !bHit || HitResult.GetActor() == SharedContext.PrimaryTarget;

			// Debug: draw LOS line
			DrawDebugLine(World, StartLoc, EndLoc, bHasLOS ? FColor::Green : FColor::Red, false, 0.1f, 0, 2.0f);
		}

		// Fire weapon at target if available
		UWeaponComponent* WeaponComp = Pawn->FindComponentByClass<UWeaponComponent>();
		UE_LOG(LogTemp, Warning, TEXT("[ASSAULT TASK] '%s': WeaponComp=%s, bHasLOS=%d, Distance=%.1f"),
			*Pawn->GetName(),
			WeaponComp ? TEXT("Found") : TEXT("NULL"),
			bHasLOS ? 1 : 0,
			Distance);

		if (WeaponComp)
		{
			if (WeaponComp->CanFire())
			{
				if (bHasLOS)
				{
					WeaponComp->FireAtTarget(SharedContext.PrimaryTarget, true);
					UE_LOG(LogTemp, Warning, TEXT("[ASSAULT TASK] '%s': FIRING at target '%s'"),
						*Pawn->GetName(),
						*SharedContext.PrimaryTarget->GetName());
				}
				else
				{
					UE_LOG(LogTemp, Warning, TEXT("[ASSAULT TASK] '%s': No LOS to target, moving only"),
						*Pawn->GetName());
				}
			}
			else
			{
				UE_LOG(LogTemp, Warning, TEXT("[ASSAULT TASK] '%s': Weapon on cooldown"),
					*Pawn->GetName());
			}
		}
		else
		{
			UE_LOG(LogTemp, Error, TEXT("[ASSAULT TASK] '%s': No WeaponComponent found!"),
				*Pawn->GetName());
		}

		SharedContext.bIsMoving = true;
		SharedContext.MovementDestination = TargetLocation;
	}
	else if (SharedContext.CurrentCommand.TargetLocation != FVector::ZeroVector)
	{
		// No target, check if we should move to command location
		FVector CurrentLocation = Pawn->GetActorLocation();
		float DistanceToCommandLocation = FVector::Dist(CurrentLocation, SharedContext.CurrentCommand.TargetLocation);

		UE_LOG(LogTemp, Warning, TEXT("[ASSAULT TASK] '%s': No PrimaryTarget - DistanceToCommandLoc=%.1f"),
			*Pawn->GetName(), DistanceToCommandLocation);

		// Only move if we're not already near the command location
		if (DistanceToCommandLocation > 150.0f)
		{
			// Set movement speed
			SharedContext.MovementSpeedMultiplier = InstanceData.AggressiveSpeedMultiplier;
			if (UCharacterMovementComponent* MovementComp = Pawn->FindComponentByClass<UCharacterMovementComponent>())
			{
				float BaseSpeed = 600.0f;
				MovementComp->MaxWalkSpeed = BaseSpeed * SharedContext.MovementSpeedMultiplier;
			}

			if (SharedContext.AIController)
			{
				// Check if NavMesh exists at target location
				UNavigationSystemV1* NavSys = UNavigationSystemV1::GetCurrent(Pawn->GetWorld());
				FNavLocation NavLoc;
				bool bOnNavMesh = NavSys && NavSys->ProjectPointToNavigation(
					SharedContext.CurrentCommand.TargetLocation, NavLoc, FVector(500.0f, 500.0f, 500.0f));

				if (!bOnNavMesh)
				{
					UE_LOG(LogTemp, Error, TEXT("[ASSAULT TASK] '%s': Target location NOT on NavMesh! CommandLoc=%s"),
						*Pawn->GetName(), *SharedContext.CurrentCommand.TargetLocation.ToString());
				}

				EPathFollowingRequestResult::Type MoveResult = SharedContext.AIController->MoveToLocation(
					SharedContext.CurrentCommand.TargetLocation, 50.0f);  // Reduced acceptance radius

				UE_LOG(LogTemp, Warning, TEXT("[ASSAULT TASK] '%s': Moving to command location (result: %d, OnNavMesh: %d)"),
					*Pawn->GetName(), (int32)MoveResult, bOnNavMesh ? 1 : 0);
			}
			SharedContext.bIsMoving = true;
		}
		else
		{
			// Already at command location, stop and wait for new target
			if (SharedContext.AIController)
			{
				SharedContext.AIController->StopMovement();
			}
			SharedContext.bIsMoving = false;

			UE_LOG(LogTemp, Warning, TEXT("[ASSAULT TASK] '%s': Already near command location (%.1f units), holding position"),
				*Pawn->GetName(), DistanceToCommandLocation);
		}
	}
	else
	{
		UE_LOG(LogTemp, Error, TEXT("[ASSAULT TASK] '%s': No target and no command location!"),
			*Pawn->GetName());
	}
}

void FSTTask_ExecuteAssault::ExecuteCautiousAdvance(FStateTreeExecutionContext& Context, float DeltaTime) const
{
	FInstanceDataType& InstanceData = Context.GetInstanceData(*this);
	FFollowerStateTreeContext& SharedContext = InstanceData.StateTreeComp->GetSharedContext();

	APawn* Pawn = SharedContext.AIController ? SharedContext.AIController->GetPawn() : nullptr;
	if (!Pawn) return;

	// Validate and update primary target if needed
	if (!UGameAIHelper::IsTargetValid(SharedContext.PrimaryTarget))
	{
		AActor* NewTarget = UGameAIHelper::FindNearestValidEnemy(SharedContext.VisibleEnemies, Pawn);
		SharedContext.PrimaryTarget = NewTarget;
	}

	// Advance more slowly, prefer cover
	SharedContext.MovementSpeedMultiplier = 1.0f;

	// Apply speed to CharacterMovementComponent
	if (UCharacterMovementComponent* MovementComp = Pawn->FindComponentByClass<UCharacterMovementComponent>())
	{
		float BaseSpeed = 600.0f;
		MovementComp->MaxWalkSpeed = BaseSpeed * SharedContext.MovementSpeedMultiplier;
	}

	if (SharedContext.PrimaryTarget)
	{
		FVector CurrentLocation = Pawn->GetActorLocation();
		FVector TargetLocation = SharedContext.PrimaryTarget->GetActorLocation();
		FVector DirectionToTarget = (TargetLocation - CurrentLocation).GetSafeNormal();

		// If too close and not in cover, seek cover first
		if (SharedContext.DistanceToPrimaryTarget < InstanceData.OptimalEngagementRange &&
			!SharedContext.bInCover)
		{
			// Find cover between current position and target
			if (SharedContext.NearestCoverLocation != FVector::ZeroVector)
			{
				if (SharedContext.AIController)
				{
					SharedContext.AIController->MoveToLocation(
						SharedContext.NearestCoverLocation, 50.0f);
				}
				return;
			}
		}

		// Advance toward target, stopping periodically
		FVector AdvanceDestination = CurrentLocation + DirectionToTarget * 300.0f; // 3m advance

		if (SharedContext.AIController)
		{
			SharedContext.AIController->MoveToLocation(AdvanceDestination, 100.0f);
			SharedContext.AIController->SetFocus(SharedContext.PrimaryTarget);
		}

		// Fire weapon at target if in cover or has LOS
		if (UWeaponComponent* WeaponComp = Pawn->FindComponentByClass<UWeaponComponent>())
		{
			if (WeaponComp->CanFire() && SharedContext.bHasLOS &&
				(SharedContext.bInCover || !SharedContext.bIsMoving))
			{
				WeaponComp->FireAtTarget(SharedContext.PrimaryTarget, true);
			}
		}

		SharedContext.bIsMoving = true;
	}
}

void FSTTask_ExecuteAssault::ExecuteFlankManeuver(FStateTreeExecutionContext& Context, float DeltaTime, bool bFlankLeft) const
{
	FInstanceDataType& InstanceData = Context.GetInstanceData(*this);
	FFollowerStateTreeContext& SharedContext = InstanceData.StateTreeComp->GetSharedContext();

	APawn* Pawn = SharedContext.AIController ? SharedContext.AIController->GetPawn() : nullptr;
	if (!Pawn) return;

	// Validate and update primary target if needed
	if (!UGameAIHelper::IsTargetValid(SharedContext.PrimaryTarget))
	{
		AActor* NewTarget = UGameAIHelper::FindNearestValidEnemy(SharedContext.VisibleEnemies, Pawn);
		SharedContext.PrimaryTarget = NewTarget;
	}

	if (!SharedContext.PrimaryTarget) return;

	FVector CurrentLocation = Pawn->GetActorLocation();
	FVector TargetLocation = SharedContext.PrimaryTarget->GetActorLocation();
	FVector DirectionToTarget = (TargetLocation - CurrentLocation).GetSafeNormal();

	// Calculate perpendicular flank direction
	FVector FlankDirection = FVector::CrossProduct(DirectionToTarget, FVector::UpVector);
	if (!bFlankLeft)
	{
		FlankDirection *= -1.0f; // Flip for right flank
	}

	// Combine forward and flank movement (45-degree angle)
	FVector FlankDestination = CurrentLocation + (DirectionToTarget * 500.0f) + (FlankDirection * 500.0f);

	if (SharedContext.AIController)
	{
		SharedContext.AIController->MoveToLocation(FlankDestination, 100.0f);
		SharedContext.AIController->SetFocus(SharedContext.PrimaryTarget);
	}

	// Fire while flanking if has LOS
	if (UWeaponComponent* WeaponComp = Pawn->FindComponentByClass<UWeaponComponent>())
	{
		if (WeaponComp->CanFire() && SharedContext.bHasLOS)
		{
			WeaponComp->FireAtTarget(SharedContext.PrimaryTarget, true);
		}
	}

	SharedContext.bIsMoving = true;
	SharedContext.MovementSpeedMultiplier = 1.2f; // Slightly faster for flanking

	// Apply speed to CharacterMovementComponent
	if (UCharacterMovementComponent* MovementComp = Pawn->FindComponentByClass<UCharacterMovementComponent>())
	{
		float BaseSpeed = 600.0f;
		MovementComp->MaxWalkSpeed = BaseSpeed * SharedContext.MovementSpeedMultiplier;
	}
}

void FSTTask_ExecuteAssault::ExecuteMaintainDistance(FStateTreeExecutionContext& Context, float DeltaTime) const
{
	FInstanceDataType& InstanceData = Context.GetInstanceData(*this);
	FFollowerStateTreeContext& SharedContext = InstanceData.StateTreeComp->GetSharedContext();

	APawn* Pawn = SharedContext.AIController ? SharedContext.AIController->GetPawn() : nullptr;
	if (!Pawn) return;

	// Validate and update primary target if needed
	if (!UGameAIHelper::IsTargetValid(SharedContext.PrimaryTarget))
	{
		AActor* NewTarget = UGameAIHelper::FindNearestValidEnemy(SharedContext.VisibleEnemies, Pawn);
		SharedContext.PrimaryTarget = NewTarget;
	}

	if (!SharedContext.PrimaryTarget) return;

	float DistanceToTarget = SharedContext.DistanceToPrimaryTarget;
	float OptimalRange = InstanceData.OptimalEngagementRange;

	// Kite: maintain optimal distance while engaging
	if (DistanceToTarget < OptimalRange * 0.8f)
	{
		// Too close, back away
		FVector CurrentLocation = Pawn->GetActorLocation();
		FVector TargetLocation = SharedContext.PrimaryTarget->GetActorLocation();
		FVector AwayDirection = (CurrentLocation - TargetLocation).GetSafeNormal();
		FVector RetreatDestination = CurrentLocation + AwayDirection * 300.0f; // 3m retreat

		if (SharedContext.AIController)
		{
			SharedContext.AIController->MoveToLocation(RetreatDestination, 50.0f);
			SharedContext.AIController->SetFocus(SharedContext.PrimaryTarget);
		}

		SharedContext.bIsMoving = true;
	}
	else if (DistanceToTarget > OptimalRange * 1.2f)
	{
		// Too far, advance
		FVector TargetLocation = SharedContext.PrimaryTarget->GetActorLocation();

		if (SharedContext.AIController)
		{
			SharedContext.AIController->MoveToLocation(TargetLocation, OptimalRange);
			SharedContext.AIController->SetFocus(SharedContext.PrimaryTarget);
		}

		SharedContext.bIsMoving = true;
	}
	else
	{
		// At optimal range, stop and engage
		if (SharedContext.AIController)
		{
			SharedContext.AIController->StopMovement();
			SharedContext.AIController->SetFocus(SharedContext.PrimaryTarget);
		}

		SharedContext.bIsMoving = false;
	}

	// Fire weapon at target (prioritize firing at optimal range)
	if (UWeaponComponent* WeaponComp = Pawn->FindComponentByClass<UWeaponComponent>())
	{
		if (WeaponComp->CanFire() && SharedContext.bHasLOS)
		{
			WeaponComp->FireAtTarget(SharedContext.PrimaryTarget, true);
		}
	}
}

float FSTTask_ExecuteAssault::CalculateAssaultReward(FStateTreeExecutionContext& Context, float DeltaTime) const
{
	FInstanceDataType& InstanceData = Context.GetInstanceData(*this);
	FFollowerStateTreeContext& SharedContext = InstanceData.StateTreeComp->GetSharedContext();

	float Reward = 0.0f;

	// Reward for closing distance to target
	if (SharedContext.PrimaryTarget && SharedContext.bIsMoving)
	{
		// Check if moving toward target
		APawn* Pawn = SharedContext.AIController ? SharedContext.AIController->GetPawn() : nullptr;
		if (Pawn)
		{
			FVector ToTarget = SharedContext.PrimaryTarget->GetActorLocation() - Pawn->GetActorLocation();
			FVector Velocity = Pawn->GetVelocity();

			if (FVector::DotProduct(ToTarget.GetSafeNormal(), Velocity.GetSafeNormal()) > 0.0f)
			{
				Reward += 0.2f * DeltaTime; // +0.2 per second advancing toward target
			}
		}
	}

	// Reward for maintaining line of sight
	if (SharedContext.bHasLOS && SharedContext.PrimaryTarget)
	{
		Reward += 1.5f * DeltaTime; // +1.5 per second with LOS
	}

	// Reward for following assault command
	if (SharedContext.bIsCommandValid)
	{
		Reward += FTacticalRewards::FOLLOW_COMMAND * DeltaTime;
	}

	// Penalty for being under fire without cover during cautious advance
	if (SharedContext.CurrentTacticalAction == ETacticalAction::CautiousAdvance &&
		SharedContext.bUnderFire && !SharedContext.bInCover)
	{
		Reward -= 0.2f * DeltaTime; // -0.2 per second exposed during cautious advance
	}

	return Reward;
}

void FSTTask_ExecuteAssault::ExecuteDefensiveHold(FStateTreeExecutionContext& Context, float DeltaTime) const
{
	FInstanceDataType& InstanceData = Context.GetInstanceData(*this);
	FFollowerStateTreeContext& SharedContext = InstanceData.StateTreeComp->GetSharedContext();

	APawn* Pawn = SharedContext.AIController ? SharedContext.AIController->GetPawn() : nullptr;
	if (!Pawn) return;

	// Stop movement and hold position
	if (SharedContext.AIController)
	{
		SharedContext.AIController->StopMovement();
	}
	SharedContext.bIsMoving = false;
	SharedContext.MovementSpeedMultiplier = 0.0f;

	// Validate and update primary target if needed
	if (!UGameAIHelper::IsTargetValid(SharedContext.PrimaryTarget))
	{
		AActor* NewTarget = UGameAIHelper::FindNearestValidEnemy(SharedContext.VisibleEnemies, Pawn);
		SharedContext.PrimaryTarget = NewTarget;
	}

	// Focus on target and fire
	if (SharedContext.PrimaryTarget)
	{
		SharedContext.AIController->SetFocus(SharedContext.PrimaryTarget);

		// Perform LOS check
		bool bHasLOS = false;
		UWorld* World = Pawn->GetWorld();
		if (World)
		{
			FHitResult HitResult;
			FCollisionQueryParams QueryParams;
			QueryParams.AddIgnoredActor(Pawn);

			FVector StartLoc = Pawn->GetActorLocation() + FVector(0, 0, 80.0f);
			FVector EndLoc = SharedContext.PrimaryTarget->GetActorLocation() + FVector(0, 0, 80.0f);

			bool bHit = World->LineTraceSingleByChannel(HitResult, StartLoc, EndLoc, ECC_Visibility, QueryParams);
			bHasLOS = !bHit || HitResult.GetActor() == SharedContext.PrimaryTarget;
		}

		// Fire weapon at target
		if (UWeaponComponent* WeaponComp = Pawn->FindComponentByClass<UWeaponComponent>())
		{
			if (WeaponComp->CanFire() && bHasLOS)
			{
				WeaponComp->FireAtTarget(SharedContext.PrimaryTarget, true);
			}
		}
	}
}

void FSTTask_ExecuteAssault::ExecuteSeekCover(FStateTreeExecutionContext& Context, float DeltaTime) const
{
	FInstanceDataType& InstanceData = Context.GetInstanceData(*this);
	FFollowerStateTreeContext& SharedContext = InstanceData.StateTreeComp->GetSharedContext();

	APawn* Pawn = SharedContext.AIController ? SharedContext.AIController->GetPawn() : nullptr;
	if (!Pawn) return;

	// If already in cover, hold and fire
	if (SharedContext.bInCover)
	{
		ExecuteDefensiveHold(Context, DeltaTime);
		return;
	}

	// Move to nearest cover
	if (SharedContext.NearestCoverLocation != FVector::ZeroVector)
	{
		if (SharedContext.AIController)
		{
			float DestinationDelta = FVector::Dist(SharedContext.MovementDestination, SharedContext.NearestCoverLocation);
			if (DestinationDelta > 100.0f || SharedContext.MovementDestination == FVector::ZeroVector)
			{
				SharedContext.AIController->MoveToLocation(SharedContext.NearestCoverLocation, 50.0f);
				SharedContext.MovementDestination = SharedContext.NearestCoverLocation;
			}

			// Keep focus on target while moving to cover
			if (SharedContext.PrimaryTarget)
			{
				SharedContext.AIController->SetFocus(SharedContext.PrimaryTarget);
			}
		}
		SharedContext.bIsMoving = true;
		SharedContext.MovementSpeedMultiplier = 1.3f; // Move quickly to cover

		// Apply speed to CharacterMovementComponent
		if (UCharacterMovementComponent* MovementComp = Pawn->FindComponentByClass<UCharacterMovementComponent>())
		{
			float BaseSpeed = 600.0f;
			MovementComp->MaxWalkSpeed = BaseSpeed * SharedContext.MovementSpeedMultiplier;
		}
	}
	else
	{
		// No cover found, fall back to cautious advance
		ExecuteCautiousAdvance(Context, DeltaTime);
	}
}

void FSTTask_ExecuteAssault::ExecuteTacticalRetreat(FStateTreeExecutionContext& Context, float DeltaTime) const
{
	FInstanceDataType& InstanceData = Context.GetInstanceData(*this);
	FFollowerStateTreeContext& SharedContext = InstanceData.StateTreeComp->GetSharedContext();

	APawn* Pawn = SharedContext.AIController ? SharedContext.AIController->GetPawn() : nullptr;
	if (!Pawn) return;

	// Validate and update primary target if needed
	if (!UGameAIHelper::IsTargetValid(SharedContext.PrimaryTarget))
	{
		AActor* NewTarget = UGameAIHelper::FindNearestValidEnemy(SharedContext.VisibleEnemies, Pawn);
		SharedContext.PrimaryTarget = NewTarget;
	}

	FVector CurrentLocation = Pawn->GetActorLocation();
	FVector RetreatDirection;

	// Calculate retreat direction (away from target or toward spawn)
	if (SharedContext.PrimaryTarget)
	{
		FVector TargetLocation = SharedContext.PrimaryTarget->GetActorLocation();
		RetreatDirection = (CurrentLocation - TargetLocation).GetSafeNormal();

		// Keep focus on target while retreating
		if (SharedContext.AIController)
		{
			SharedContext.AIController->SetFocus(SharedContext.PrimaryTarget);
		}
	}
	else
	{
		// Retreat toward spawn/command location
		FVector SpawnLocation = SharedContext.CurrentCommand.TargetLocation;
		if (SpawnLocation != FVector::ZeroVector)
		{
			RetreatDirection = (SpawnLocation - CurrentLocation).GetSafeNormal();
		}
		else
		{
			RetreatDirection = -Pawn->GetActorForwardVector(); // Just go backward
		}
	}

	// Move in retreat direction
	FVector RetreatDestination = CurrentLocation + RetreatDirection * 500.0f;

	if (SharedContext.AIController)
	{
		SharedContext.AIController->MoveToLocation(RetreatDestination, 100.0f);
	}

	SharedContext.bIsMoving = true;
	SharedContext.MovementSpeedMultiplier = 1.2f;

	// Apply speed to CharacterMovementComponent
	if (UCharacterMovementComponent* MovementComp = Pawn->FindComponentByClass<UCharacterMovementComponent>())
	{
		float BaseSpeed = 600.0f;
		MovementComp->MaxWalkSpeed = BaseSpeed * SharedContext.MovementSpeedMultiplier;
	}

	// Fire while retreating if has LOS
	if (SharedContext.PrimaryTarget)
	{
		if (UWeaponComponent* WeaponComp = Pawn->FindComponentByClass<UWeaponComponent>())
		{
			if (WeaponComp->CanFire() && SharedContext.bHasLOS)
			{
				WeaponComp->FireAtTarget(SharedContext.PrimaryTarget, true);
			}
		}
	}
}

void FSTTask_ExecuteAssault::ExecuteSuppressiveFire(FStateTreeExecutionContext& Context, float DeltaTime) const
{
	FInstanceDataType& InstanceData = Context.GetInstanceData(*this);
	FFollowerStateTreeContext& SharedContext = InstanceData.StateTreeComp->GetSharedContext();

	APawn* Pawn = SharedContext.AIController ? SharedContext.AIController->GetPawn() : nullptr;
	if (!Pawn) return;

	// Validate and update primary target if needed
	if (!UGameAIHelper::IsTargetValid(SharedContext.PrimaryTarget))
	{
		AActor* NewTarget = UGameAIHelper::FindNearestValidEnemy(SharedContext.VisibleEnemies, Pawn);
		SharedContext.PrimaryTarget = NewTarget;
	}

	// Minimal movement - stay in position or move slowly
	SharedContext.MovementSpeedMultiplier = 0.3f;

	if (SharedContext.PrimaryTarget)
	{
		// Focus on target
		if (SharedContext.AIController)
		{
			SharedContext.AIController->SetFocus(SharedContext.PrimaryTarget);
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
			FVector EndLoc = SharedContext.PrimaryTarget->GetActorLocation() + FVector(0, 0, 80.0f);

			bool bHit = World->LineTraceSingleByChannel(HitResult, StartLoc, EndLoc, ECC_Visibility, QueryParams);
			bHasLOS = !bHit || HitResult.GetActor() == SharedContext.PrimaryTarget;
		}

		// Fire rapidly at target area (suppressive fire - less accurate but more volume)
		if (UWeaponComponent* WeaponComp = Pawn->FindComponentByClass<UWeaponComponent>())
		{
			if (WeaponComp->CanFire() && bHasLOS)
			{
				// Fire with prediction disabled for suppressive effect (spray area)
				WeaponComp->FireAtTarget(SharedContext.PrimaryTarget, false);
			}
		}

		// Slight movement to find better angle if no LOS
		if (!bHasLOS && SharedContext.AIController)
		{
			FVector CurrentLocation = Pawn->GetActorLocation();
			FVector TargetLocation = SharedContext.PrimaryTarget->GetActorLocation();
			FVector ToTarget = (TargetLocation - CurrentLocation).GetSafeNormal();
			FVector SideStep = FVector::CrossProduct(ToTarget, FVector::UpVector) * 200.0f;

			// Alternate side to find LOS
			if (FMath::Fmod(SharedContext.TimeInTacticalAction, 2.0f) < 1.0f)
			{
				SideStep *= -1.0f;
			}

			SharedContext.AIController->MoveToLocation(CurrentLocation + SideStep, 50.0f);
			SharedContext.bIsMoving = true;
		}
		else
		{
			SharedContext.bIsMoving = false;
		}
	}
}
