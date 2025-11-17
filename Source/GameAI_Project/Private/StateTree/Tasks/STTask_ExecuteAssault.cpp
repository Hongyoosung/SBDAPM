// Copyright Epic Games, Inc. All Rights Reserved.

#include "StateTree/Tasks/STTask_ExecuteAssault.h"
#include "StateTree/FollowerStateTreeContext.h"
#include "Team/FollowerAgentComponent.h"
#include "Combat/WeaponComponent.h"
#include "AIController.h"
#include "NavigationSystem.h"
#include "GameFramework/Pawn.h"
#include "DrawDebugHelpers.h"
#include "Kismet/GameplayStatics.h"

EStateTreeRunStatus FSTTask_ExecuteAssault::EnterState(FStateTreeExecutionContext& Context, const FStateTreeTransitionResult& Transition) const
{
	FInstanceDataType& InstanceData = Context.GetInstanceData(*this);

	// Validate inputs
	if (!InstanceData.Context.FollowerComponent || !InstanceData.Context.AIController)
	{
		UE_LOG(LogTemp, Error, TEXT("STTask_ExecuteAssault: Invalid inputs (missing component/controller)"));
		return EStateTreeRunStatus::Failed;
	}

	APawn* Pawn = InstanceData.Context.AIController->GetPawn();
	FString PawnName = Pawn ? Pawn->GetName() : TEXT("Unknown");
	FString TargetName = InstanceData.Context.PrimaryTarget ? InstanceData.Context.PrimaryTarget->GetName() : TEXT("None");

	UE_LOG(LogTemp, Warning, TEXT("🎯 [ASSAULT TASK] '%s': ENTERED assault state - Target: %s, Tactic: %s"),
		*PawnName,
		*TargetName,
		*UEnum::GetValueAsString(InstanceData.Context.CurrentTacticalAction));

	// Reset timers
	InstanceData.TimeSinceLastRLQuery = 0.0f;
	InstanceData.Context.TimeInTacticalAction = 0.0f;
	InstanceData.Context.ActionProgress = 0.0f;

	return EStateTreeRunStatus::Running;
}

EStateTreeRunStatus FSTTask_ExecuteAssault::Tick(FStateTreeExecutionContext& Context, float DeltaTime) const
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
	float Reward = CalculateAssaultReward(Context, DeltaTime);
	if (Reward != 0.0f && InstanceData.Context.FollowerComponent)
	{
		InstanceData.Context.FollowerComponent->ProvideReward(Reward, false);
	}

	return EStateTreeRunStatus::Running;
}

void FSTTask_ExecuteAssault::ExitState(FStateTreeExecutionContext& Context, const FStateTreeTransitionResult& Transition) const
{
	FInstanceDataType& InstanceData = Context.GetInstanceData(*this);

	UE_LOG(LogTemp, Log, TEXT("STTask_ExecuteAssault: Exiting assault (time in action: %.1fs)"),
		InstanceData.Context.TimeInTacticalAction);
}

void FSTTask_ExecuteAssault::ExecuteTacticalAction(FStateTreeExecutionContext& Context, float DeltaTime) const
{
	FInstanceDataType& InstanceData = Context.GetInstanceData(*this);

	switch (InstanceData.Context.CurrentTacticalAction)
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

	default:
		// Default to aggressive assault
		ExecuteAggressiveAssault(Context, DeltaTime);
		break;
	}
}

void FSTTask_ExecuteAssault::ExecuteAggressiveAssault(FStateTreeExecutionContext& Context, float DeltaTime) const
{
	FInstanceDataType& InstanceData = Context.GetInstanceData(*this);

	APawn* Pawn = InstanceData.Context.AIController ? InstanceData.Context.AIController->GetPawn() : nullptr;
	if (!Pawn)
	{
		UE_LOG(LogTemp, Warning, TEXT("[ASSAULT TASK] AggressiveAssault: No pawn!"));
		return;
	}

	// Move toward target aggressively
	if (InstanceData.Context.PrimaryTarget)
	{
		FVector CurrentLocation = Pawn->GetActorLocation();
		FVector TargetLocation = InstanceData.Context.PrimaryTarget->GetActorLocation();
		float Distance = FVector::Dist(CurrentLocation, TargetLocation);

		UE_LOG(LogTemp, Display, TEXT("[ASSAULT TASK] '%s': Moving to target '%s' (Distance: %.1f cm)"),
			*Pawn->GetName(),
			*InstanceData.Context.PrimaryTarget->GetName(),
			Distance);

		// Set high speed multiplier
		InstanceData.Context.MovementSpeedMultiplier = InstanceData.AggressiveSpeedMultiplier;

		// Move directly toward target
		if (InstanceData.Context.AIController)
		{
			InstanceData.Context.AIController->MoveToLocation(TargetLocation, 100.0f); // 1m acceptance radius
			InstanceData.Context.AIController->SetFocus(InstanceData.Context.PrimaryTarget);
		}

		// Fire weapon at target if available
		if (UWeaponComponent* WeaponComp = Pawn->FindComponentByClass<UWeaponComponent>())
		{
			if (WeaponComp->CanFire())
			{
				if (InstanceData.Context.bHasLOS)
				{
					WeaponComp->FireAtTarget(InstanceData.Context.PrimaryTarget, true);
					UE_LOG(LogTemp, Display, TEXT("[ASSAULT TASK] '%s': Firing at target '%s'"),
						*Pawn->GetName(),
						*InstanceData.Context.PrimaryTarget->GetName());
				}
				else
				{
					UE_LOG(LogTemp, Verbose, TEXT("[ASSAULT TASK] '%s': No LOS to target, moving only"),
						*Pawn->GetName());
				}
			}
			else
			{
				UE_LOG(LogTemp, Verbose, TEXT("[ASSAULT TASK] '%s': Weapon on cooldown"),
					*Pawn->GetName());
			}
		}
		else
		{
			UE_LOG(LogTemp, Warning, TEXT("[ASSAULT TASK] '%s': No WeaponComponent found!"),
				*Pawn->GetName());
		}

		InstanceData.Context.bIsMoving = true;
		InstanceData.Context.MovementDestination = TargetLocation;
	}
	else if (InstanceData.Context.CurrentCommand.TargetLocation != FVector::ZeroVector)
	{
		// No target, move to command location
		UE_LOG(LogTemp, Warning, TEXT("[ASSAULT TASK] '%s': No PrimaryTarget, moving to command location"),
			*Pawn->GetName());

		if (InstanceData.Context.AIController)
		{
			InstanceData.Context.AIController->MoveToLocation(
				InstanceData.Context.CurrentCommand.TargetLocation, 100.0f);
		}
		InstanceData.Context.bIsMoving = true;
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

	APawn* Pawn = InstanceData.Context.AIController ? InstanceData.Context.AIController->GetPawn() : nullptr;
	if (!Pawn) return;

	// Advance more slowly, prefer cover
	InstanceData.Context.MovementSpeedMultiplier = 1.0f;

	if (InstanceData.Context.PrimaryTarget)
	{
		FVector CurrentLocation = Pawn->GetActorLocation();
		FVector TargetLocation = InstanceData.Context.PrimaryTarget->GetActorLocation();
		FVector DirectionToTarget = (TargetLocation - CurrentLocation).GetSafeNormal();

		// If too close and not in cover, seek cover first
		if (InstanceData.Context.DistanceToPrimaryTarget < InstanceData.OptimalEngagementRange &&
			!InstanceData.Context.bInCover)
		{
			// Find cover between current position and target
			if (InstanceData.Context.NearestCoverLocation != FVector::ZeroVector)
			{
				if (InstanceData.Context.AIController)
				{
					InstanceData.Context.AIController->MoveToLocation(
						InstanceData.Context.NearestCoverLocation, 50.0f);
				}
				return;
			}
		}

		// Advance toward target, stopping periodically
		FVector AdvanceDestination = CurrentLocation + DirectionToTarget * 300.0f; // 3m advance

		if (InstanceData.Context.AIController)
		{
			InstanceData.Context.AIController->MoveToLocation(AdvanceDestination, 100.0f);
			InstanceData.Context.AIController->SetFocus(InstanceData.Context.PrimaryTarget);
		}

		// Fire weapon at target if in cover or has LOS
		if (UWeaponComponent* WeaponComp = Pawn->FindComponentByClass<UWeaponComponent>())
		{
			if (WeaponComp->CanFire() && InstanceData.Context.bHasLOS &&
				(InstanceData.Context.bInCover || !InstanceData.Context.bIsMoving))
			{
				WeaponComp->FireAtTarget(InstanceData.Context.PrimaryTarget, true);
			}
		}

		InstanceData.Context.bIsMoving = true;
	}
}

void FSTTask_ExecuteAssault::ExecuteFlankManeuver(FStateTreeExecutionContext& Context, float DeltaTime, bool bFlankLeft) const
{
	FInstanceDataType& InstanceData = Context.GetInstanceData(*this);

	APawn* Pawn = InstanceData.Context.AIController ? InstanceData.Context.AIController->GetPawn() : nullptr;
	if (!Pawn || !InstanceData.Context.PrimaryTarget) return;

	FVector CurrentLocation = Pawn->GetActorLocation();
	FVector TargetLocation = InstanceData.Context.PrimaryTarget->GetActorLocation();
	FVector DirectionToTarget = (TargetLocation - CurrentLocation).GetSafeNormal();

	// Calculate perpendicular flank direction
	FVector FlankDirection = FVector::CrossProduct(DirectionToTarget, FVector::UpVector);
	if (!bFlankLeft)
	{
		FlankDirection *= -1.0f; // Flip for right flank
	}

	// Combine forward and flank movement (45-degree angle)
	FVector FlankDestination = CurrentLocation + (DirectionToTarget * 500.0f) + (FlankDirection * 500.0f);

	if (InstanceData.Context.AIController)
	{
		InstanceData.Context.AIController->MoveToLocation(FlankDestination, 100.0f);
		InstanceData.Context.AIController->SetFocus(InstanceData.Context.PrimaryTarget);
	}

	// Fire while flanking if has LOS
	if (UWeaponComponent* WeaponComp = Pawn->FindComponentByClass<UWeaponComponent>())
	{
		if (WeaponComp->CanFire() && InstanceData.Context.bHasLOS)
		{
			WeaponComp->FireAtTarget(InstanceData.Context.PrimaryTarget, true);
		}
	}

	InstanceData.Context.bIsMoving = true;
	InstanceData.Context.MovementSpeedMultiplier = 1.2f; // Slightly faster for flanking
}

void FSTTask_ExecuteAssault::ExecuteMaintainDistance(FStateTreeExecutionContext& Context, float DeltaTime) const
{
	FInstanceDataType& InstanceData = Context.GetInstanceData(*this);

	APawn* Pawn = InstanceData.Context.AIController ? InstanceData.Context.AIController->GetPawn() : nullptr;
	if (!Pawn || !InstanceData.Context.PrimaryTarget) return;

	float DistanceToTarget = InstanceData.Context.DistanceToPrimaryTarget;
	float OptimalRange = InstanceData.OptimalEngagementRange;

	// Kite: maintain optimal distance while engaging
	if (DistanceToTarget < OptimalRange * 0.8f)
	{
		// Too close, back away
		FVector CurrentLocation = Pawn->GetActorLocation();
		FVector TargetLocation = InstanceData.Context.PrimaryTarget->GetActorLocation();
		FVector AwayDirection = (CurrentLocation - TargetLocation).GetSafeNormal();
		FVector RetreatDestination = CurrentLocation + AwayDirection * 300.0f; // 3m retreat

		if (InstanceData.Context.AIController)
		{
			InstanceData.Context.AIController->MoveToLocation(RetreatDestination, 50.0f);
			InstanceData.Context.AIController->SetFocus(InstanceData.Context.PrimaryTarget);
		}

		InstanceData.Context.bIsMoving = true;
	}
	else if (DistanceToTarget > OptimalRange * 1.2f)
	{
		// Too far, advance
		FVector TargetLocation = InstanceData.Context.PrimaryTarget->GetActorLocation();

		if (InstanceData.Context.AIController)
		{
			InstanceData.Context.AIController->MoveToLocation(TargetLocation, OptimalRange);
			InstanceData.Context.AIController->SetFocus(InstanceData.Context.PrimaryTarget);
		}

		InstanceData.Context.bIsMoving = true;
	}
	else
	{
		// At optimal range, stop and engage
		if (InstanceData.Context.AIController)
		{
			InstanceData.Context.AIController->StopMovement();
			InstanceData.Context.AIController->SetFocus(InstanceData.Context.PrimaryTarget);
		}

		InstanceData.Context.bIsMoving = false;
	}

	// Fire weapon at target (prioritize firing at optimal range)
	if (UWeaponComponent* WeaponComp = Pawn->FindComponentByClass<UWeaponComponent>())
	{
		if (WeaponComp->CanFire() && InstanceData.Context.bHasLOS)
		{
			WeaponComp->FireAtTarget(InstanceData.Context.PrimaryTarget, true);
		}
	}
}

float FSTTask_ExecuteAssault::CalculateAssaultReward(FStateTreeExecutionContext& Context, float DeltaTime) const
{
	FInstanceDataType& InstanceData = Context.GetInstanceData(*this);

	float Reward = 0.0f;

	// Reward for closing distance to target
	if (InstanceData.Context.PrimaryTarget && InstanceData.Context.bIsMoving)
	{
		// Check if moving toward target
		APawn* Pawn = InstanceData.Context.AIController ? InstanceData.Context.AIController->GetPawn() : nullptr;
		if (Pawn)
		{
			FVector ToTarget = InstanceData.Context.PrimaryTarget->GetActorLocation() - Pawn->GetActorLocation();
			FVector Velocity = Pawn->GetVelocity();

			if (FVector::DotProduct(ToTarget.GetSafeNormal(), Velocity.GetSafeNormal()) > 0.0f)
			{
				Reward += 2.0f * DeltaTime; // +2.0 per second advancing toward target
			}
		}
	}

	// Reward for maintaining line of sight
	if (InstanceData.Context.bHasLOS && InstanceData.Context.PrimaryTarget)
	{
		Reward += 1.5f * DeltaTime; // +1.5 per second with LOS
	}

	// Reward for following assault command
	if (InstanceData.Context.bIsCommandValid)
	{
		Reward += FTacticalRewards::FOLLOW_COMMAND * DeltaTime;
	}

	// Penalty for being under fire without cover during cautious advance
	if (InstanceData.Context.CurrentTacticalAction == ETacticalAction::CautiousAdvance &&
		InstanceData.Context.bUnderFire && !InstanceData.Context.bInCover)
	{
		Reward -= 2.0f * DeltaTime; // -2.0 per second exposed during cautious advance
	}

	return Reward;
}
