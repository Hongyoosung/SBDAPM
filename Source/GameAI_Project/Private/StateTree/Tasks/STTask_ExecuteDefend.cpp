// Copyright Epic Games, Inc. All Rights Reserved.

#include "StateTree/Tasks/STTask_ExecuteDefend.h"
#include "StateTree/FollowerStateTreeContext.h"
#include "Team/FollowerAgentComponent.h"
#include "AIController.h"
#include "NavigationSystem.h"
#include "GameFramework/Pawn.h"
#include "DrawDebugHelpers.h"
#include "Kismet/GameplayStatics.h"

EStateTreeRunStatus FSTTask_ExecuteDefend::EnterState(FStateTreeExecutionContext& Context, const FStateTreeTransitionResult& Transition) const
{
	FInstanceDataType& InstanceData = Context.GetInstanceData(*this);

	// Validate inputs
	if (!InstanceData.FollowerComponent || !InstanceData.AIController)
	{
		UE_LOG(LogTemp, Error, TEXT("STTask_ExecuteDefend: Invalid inputs (missing component/controller)"));
		return EStateTreeRunStatus::Failed;
	}

	// Initialize defend position
	if (InstanceData.CurrentCommand.TargetLocation != FVector::ZeroVector)
	{
		InstanceData.DefendPosition = InstanceData.CurrentCommand.TargetLocation;
	}
	else if (APawn* Pawn = InstanceData.AIController->GetPawn())
	{
		InstanceData.DefendPosition = Pawn->GetActorLocation();
	}

	// Reset timers
	InstanceData.TimeSinceLastRLQuery = 0.0f;
	InstanceData.TimeInDefensivePosition = 0.0f;

	UE_LOG(LogTemp, Log, TEXT("STTask_ExecuteDefend: Starting defense at %s"), *InstanceData.DefendPosition.ToString());

	return EStateTreeRunStatus::Running;
}

EStateTreeRunStatus FSTTask_ExecuteDefend::Tick(FStateTreeExecutionContext& Context, float DeltaTime) const
{
	FInstanceDataType& InstanceData = Context.GetInstanceData(*this);

	// Check if should abort
	if (ShouldCompleteDefense(Context))
	{
		return EStateTreeRunStatus::Succeeded;
	}

	// Update timers
	InstanceData.TimeSinceLastRLQuery += DeltaTime;
	InstanceData.TimeInDefensivePosition += DeltaTime;

	// Re-query RL policy if interval elapsed
	if (InstanceData.RLQueryInterval > 0.0f && InstanceData.TimeSinceLastRLQuery >= InstanceData.RLQueryInterval)
	{
		// This would trigger a transition back to QueryRLPolicy state
		// For now, we execute with current tactical action
		InstanceData.TimeSinceLastRLQuery = 0.0f;
	}

	// Execute current tactical action
	ExecuteTacticalAction(Context, DeltaTime);

	// Calculate and provide reward
	float Reward = CalculateDefensiveReward(Context, DeltaTime);
	if (Reward != 0.0f && InstanceData.FollowerComponent)
	{
		InstanceData.FollowerComponent->ProvideReward(Reward, false);
	}

	return EStateTreeRunStatus::Running;
}

void FSTTask_ExecuteDefend::ExitState(FStateTreeExecutionContext& Context, const FStateTreeTransitionResult& Transition) const
{
	FInstanceDataType& InstanceData = Context.GetInstanceData(*this);

	UE_LOG(LogTemp, Log, TEXT("STTask_ExecuteDefend: Exiting defense (time in position: %.1fs)"),
		InstanceData.TimeInDefensivePosition);
}

void FSTTask_ExecuteDefend::ExecuteTacticalAction(FStateTreeExecutionContext& Context, float DeltaTime) const
{
	FInstanceDataType& InstanceData = Context.GetInstanceData(*this);

	switch (InstanceData.CurrentTacticalAction)
	{
	case ETacticalAction::DefensiveHold:
		ExecuteDefensiveHold(Context, DeltaTime);
		break;

	case ETacticalAction::SeekCover:
		ExecuteSeekCover(Context, DeltaTime);
		break;

	case ETacticalAction::SuppressiveFire:
		ExecuteSuppressiveFire(Context, DeltaTime);
		break;

	case ETacticalAction::TacticalRetreat:
		ExecuteTacticalRetreat(Context, DeltaTime);
		break;

	default:
		// Default to defensive hold
		ExecuteDefensiveHold(Context, DeltaTime);
		break;
	}
}

void FSTTask_ExecuteDefend::ExecuteDefensiveHold(FStateTreeExecutionContext& Context, float DeltaTime) const
{
	FInstanceDataType& InstanceData = Context.GetInstanceData(*this);

	APawn* Pawn = InstanceData.AIController ? InstanceData.AIController->GetPawn() : nullptr;
	if (!Pawn) return;

	// Stay in position and engage threats
	float DistanceToDefendPos = FVector::Dist(Pawn->GetActorLocation(), InstanceData.DefendPosition);

	if (DistanceToDefendPos > InstanceData.MaxDefendRadius)
	{
		// Move back to defensive position
		MoveToDefensivePosition(Context, InstanceData.DefendPosition, DeltaTime);
	}
	else
	{
		// Hold position and engage
		InstanceData.AIController->StopMovement();
		EngageThreats(Context, InstanceData.bInCover ? InstanceData.CoverAccuracyBonus : 1.0f);
	}
}

void FSTTask_ExecuteDefend::ExecuteSeekCover(FStateTreeExecutionContext& Context, float DeltaTime) const
{
	FInstanceDataType& InstanceData = Context.GetInstanceData(*this);

	APawn* Pawn = InstanceData.AIController ? InstanceData.AIController->GetPawn() : nullptr;
	if (!Pawn) return;

	// Find nearest cover if not already in cover
	if (!InstanceData.bInCover || !InstanceData.CurrentCover)
	{
		AActor* NearestCover = FindNearestCover(Context, Pawn->GetActorLocation());
		if (NearestCover)
		{
			InstanceData.NearestCoverLocation = NearestCover->GetActorLocation();
			MoveToDefensivePosition(Context, InstanceData.NearestCoverLocation, DeltaTime);
		}
		else
		{
			// No cover found, fallback to defensive hold
			ExecuteDefensiveHold(Context, DeltaTime);
		}
	}
	else
	{
		// Already in cover, maintain position
		ExecuteDefensiveHold(Context, DeltaTime);
	}
}

void FSTTask_ExecuteDefend::ExecuteSuppressiveFire(FStateTreeExecutionContext& Context, float DeltaTime) const
{
	// Suppressive fire: higher fire rate, lower accuracy
	EngageThreats(Context, 0.7f);
}

void FSTTask_ExecuteDefend::ExecuteTacticalRetreat(FStateTreeExecutionContext& Context, float DeltaTime) const
{
	FInstanceDataType& InstanceData = Context.GetInstanceData(*this);

	APawn* Pawn = InstanceData.AIController ? InstanceData.AIController->GetPawn() : nullptr;
	if (!Pawn) return;

	// Retreat away from threats toward defend position
	FVector RetreatDirection = (InstanceData.DefendPosition - Pawn->GetActorLocation()).GetSafeNormal();
	FVector RetreatDestination = Pawn->GetActorLocation() + RetreatDirection * 500.0f; // 5m retreat

	MoveToDefensivePosition(Context, RetreatDestination, DeltaTime);
}

AActor* FSTTask_ExecuteDefend::FindNearestCover(FStateTreeExecutionContext& Context, const FVector& FromLocation) const
{
	FInstanceDataType& InstanceData = Context.GetInstanceData(*this);

	if (!InstanceData.AIController) return nullptr;

	UWorld* World = InstanceData.AIController->GetWorld();
	if (!World) return nullptr;

	// Find all cover actors within search radius
	TArray<AActor*> CoverActors;
	UGameplayStatics::GetAllActorsWithTag(World, FName("Cover"), CoverActors);

	AActor* NearestCover = nullptr;
	float MinDistance = InstanceData.CoverSearchRadius;

	for (AActor* CoverActor : CoverActors)
	{
		float Distance = FVector::Dist(FromLocation, CoverActor->GetActorLocation());
		if (Distance < MinDistance)
		{
			MinDistance = Distance;
			NearestCover = CoverActor;
		}
	}

	if (NearestCover)
	{
		InstanceData.DistanceToNearestCover = MinDistance;
	}

	return NearestCover;
}

float FSTTask_ExecuteDefend::CalculateDefensiveReward(FStateTreeExecutionContext& Context, float DeltaTime) const
{
	FInstanceDataType& InstanceData = Context.GetInstanceData(*this);

	float Reward = 0.0f;

	// Reward for holding position
	if (InstanceData.TimeInDefensivePosition > 1.0f)
	{
		Reward += 3.0f * DeltaTime; // +3.0 per second
	}

	// Reward for using cover
	if (InstanceData.bInCover)
	{
		Reward += 5.0f * DeltaTime; // +5.0 per second in cover
	}

	// Reward for survival under fire
	if (InstanceData.bUnderFire && InstanceData.bIsAlive)
	{
		Reward += 4.0f * DeltaTime; // +4.0 per second surviving under fire
	}

	return Reward;
}

bool FSTTask_ExecuteDefend::ShouldCompleteDefense(FStateTreeExecutionContext& Context) const
{
	FInstanceDataType& InstanceData = Context.GetInstanceData(*this);

	// Abort if dead
	if (!InstanceData.bIsAlive)
	{
		return true;
	}

	// Abort if command changed/invalid
	if (!InstanceData.bIsCommandValid)
	{
		return true;
	}

	// Continue defending
	return false;
}

void FSTTask_ExecuteDefend::MoveToDefensivePosition(FStateTreeExecutionContext& Context, const FVector& Destination, float DeltaTime) const
{
	FInstanceDataType& InstanceData = Context.GetInstanceData(*this);

	if (!InstanceData.AIController) return;

	InstanceData.AIController->MoveToLocation(Destination, 50.0f); // 50cm acceptance radius
}

void FSTTask_ExecuteDefend::EngageThreats(FStateTreeExecutionContext& Context, float AccuracyModifier) const
{
	FInstanceDataType& InstanceData = Context.GetInstanceData(*this);

	if (!InstanceData.PrimaryTarget || !InstanceData.bWeaponReady)
	{
		return;
	}

	// Engage primary target
	// (Weapon firing would be handled by separate component/task)
	// For now, just track the target
	if (InstanceData.AIController)
	{
		InstanceData.AIController->SetFocus(InstanceData.PrimaryTarget);
	}
}
