// Copyright Epic Games, Inc. All Rights Reserved.

#include "StateTree/Tasks/STTask_ExecuteRetreat.h"
#include "StateTree/FollowerStateTreeContext.h"
#include "Team/FollowerAgentComponent.h"
#include "AIController.h"
#include "NavigationSystem.h"
#include "GameFramework/Pawn.h"
#include "GameFramework/CharacterMovementComponent.h"
#include "DrawDebugHelpers.h"
#include "Kismet/GameplayStatics.h"

EStateTreeRunStatus FSTTask_ExecuteRetreat::EnterState(FStateTreeExecutionContext& Context, const FStateTreeTransitionResult& Transition) const
{
	FInstanceDataType& InstanceData = Context.GetInstanceData(*this);

	// Validate inputs
	if (!InstanceData.Context.FollowerComponent || !InstanceData.Context.AIController)
	{
		UE_LOG(LogTemp, Error, TEXT("STTask_ExecuteRetreat: Invalid inputs (missing component/controller)"));
		return EStateTreeRunStatus::Failed;
	}

	APawn* Pawn = InstanceData.Context.AIController->GetPawn();
	if (!Pawn)
	{
		UE_LOG(LogTemp, Error, TEXT("STTask_ExecuteRetreat: No pawn controlled"));
		return EStateTreeRunStatus::Failed;
	}

	// Initialize retreat state
	InstanceData.InitialRetreatPosition = Pawn->GetActorLocation();
	InstanceData.RetreatDestination = CalculateRetreatDestination(Context);
	InstanceData.TotalDistanceRetreated = 0.0f;
	InstanceData.TimeInRetreat = 0.0f;
	InstanceData.TimeSinceLastRLQuery = 0.0f;
	InstanceData.bHasReachedSafeDistance = false;

	UE_LOG(LogTemp, Log, TEXT("STTask_ExecuteRetreat: Starting retreat to %s"), *InstanceData.RetreatDestination.ToString());

	return EStateTreeRunStatus::Running;
}

EStateTreeRunStatus FSTTask_ExecuteRetreat::Tick(FStateTreeExecutionContext& Context, float DeltaTime) const
{
	FInstanceDataType& InstanceData = Context.GetInstanceData(*this);

	// Check if should complete
	if (ShouldCompleteRetreat(Context))
	{
		return EStateTreeRunStatus::Succeeded;
	}

	// Update timers
	InstanceData.TimeSinceLastRLQuery += DeltaTime;
	InstanceData.TimeInRetreat += DeltaTime;

	// Re-query RL policy if interval elapsed
	if (InstanceData.RLQueryInterval > 0.0f && InstanceData.TimeSinceLastRLQuery >= InstanceData.RLQueryInterval)
	{
		InstanceData.TimeSinceLastRLQuery = 0.0f;
		// Re-querying is handled by evaluator or separate task
	}

	// Update distances
	APawn* Pawn = InstanceData.Context.AIController->GetPawn();
	if (Pawn)
	{
		InstanceData.TotalDistanceRetreated = FVector::Dist(Pawn->GetActorLocation(), InstanceData.InitialRetreatPosition);
		InstanceData.DistanceToRetreatDestination = FVector::Dist(Pawn->GetActorLocation(), InstanceData.RetreatDestination);

		if (InstanceData.Context.PrimaryTarget)
		{
			InstanceData.DistanceFromThreat = FVector::Dist(Pawn->GetActorLocation(), InstanceData.Context.PrimaryTarget->GetActorLocation());
		}
	}

	// Execute current tactical action
	ExecuteTacticalAction(Context, DeltaTime);

	// Calculate and provide reward
	float Reward = CalculateRetreatReward(Context, DeltaTime);
	if (Reward != 0.0f && InstanceData.Context.FollowerComponent)
	{
		InstanceData.Context.FollowerComponent->ProvideReward(Reward, false);
	}

	return EStateTreeRunStatus::Running;
}

void FSTTask_ExecuteRetreat::ExitState(FStateTreeExecutionContext& Context, const FStateTreeTransitionResult& Transition) const
{
	FInstanceDataType& InstanceData = Context.GetInstanceData(*this);

	if (InstanceData.Context.AIController)
	{
		InstanceData.Context.AIController->StopMovement();
		InstanceData.Context.AIController->ClearFocus(EAIFocusPriority::Gameplay);
	}

	UE_LOG(LogTemp, Log, TEXT("STTask_ExecuteRetreat: Exiting retreat (distance: %.1fcm, time: %.1fs, safe: %s)"),
		InstanceData.TotalDistanceRetreated,
		InstanceData.TimeInRetreat,
		InstanceData.bHasReachedSafeDistance ? TEXT("Yes") : TEXT("No"));
}

void FSTTask_ExecuteRetreat::ExecuteTacticalAction(FStateTreeExecutionContext& Context, float DeltaTime) const
{
	FInstanceDataType& InstanceData = Context.GetInstanceData(*this);

	switch (InstanceData.Context.CurrentTacticalAction)
	{
	case ETacticalAction::TacticalRetreat:
		ExecuteTacticalRetreat(Context, DeltaTime);
		break;

	case ETacticalAction::Sprint:
		ExecuteSprintRetreat(Context, DeltaTime);
		break;

	case ETacticalAction::SeekCover:
		ExecuteCoverRetreat(Context, DeltaTime);
		break;

	case ETacticalAction::SuppressiveFire:
		ExecuteSuppressiveRetreat(Context, DeltaTime);
		break;

	default:
		// Default to tactical retreat
		ExecuteTacticalRetreat(Context, DeltaTime);
		break;
	}
}

void FSTTask_ExecuteRetreat::ExecuteTacticalRetreat(FStateTreeExecutionContext& Context, float DeltaTime) const
{
	FInstanceDataType& InstanceData = Context.GetInstanceData(*this);

	// Controlled fallback: Move away from threats at normal speed
	MoveToRetreatDestination(Context, InstanceData.RetreatDestination, 1.0f);

	// Maintain awareness of threats
	if (InstanceData.Context.PrimaryTarget && InstanceData.Context.AIController)
	{
		InstanceData.Context.AIController->SetFocus(InstanceData.Context.PrimaryTarget);
	}
}

void FSTTask_ExecuteRetreat::ExecuteSprintRetreat(FStateTreeExecutionContext& Context, float DeltaTime) const
{
	FInstanceDataType& InstanceData = Context.GetInstanceData(*this);

	// Fast escape: Sprint away at maximum speed
	MoveToRetreatDestination(Context, InstanceData.RetreatDestination, InstanceData.RetreatSprintMultiplier);

	// Clear focus for faster movement
	if (InstanceData.Context.AIController)
	{
		InstanceData.Context.AIController->ClearFocus(EAIFocusPriority::Gameplay);
	}
}

void FSTTask_ExecuteRetreat::ExecuteCoverRetreat(FStateTreeExecutionContext& Context, float DeltaTime) const
{
	FInstanceDataType& InstanceData = Context.GetInstanceData(*this);

	APawn* Pawn = InstanceData.Context.AIController->GetPawn();
	if (!Pawn) return;

	// Find and move to nearest cover
	AActor* NearestCover = FindNearestCover(Context, Pawn->GetActorLocation());
	if (NearestCover)
	{
		InstanceData.NearestCoverLocation = NearestCover->GetActorLocation();
		MoveToRetreatDestination(Context, InstanceData.NearestCoverLocation, 1.2f);
	}
	else
	{
		// No cover found, fallback to tactical retreat
		ExecuteTacticalRetreat(Context, DeltaTime);
	}
}

void FSTTask_ExecuteRetreat::ExecuteSuppressiveRetreat(FStateTreeExecutionContext& Context, float DeltaTime) const
{
	FInstanceDataType& InstanceData = Context.GetInstanceData(*this);

	// Retreat while providing suppressive fire
	MoveToRetreatDestination(Context, InstanceData.RetreatDestination, 0.7f); // Slower due to firing

	ProvideSuppressiveFire(Context);
}

FVector FSTTask_ExecuteRetreat::CalculateRetreatDestination(FStateTreeExecutionContext& Context) const
{
	FInstanceDataType& InstanceData = Context.GetInstanceData(*this);

	APawn* Pawn = InstanceData.Context.AIController->GetPawn();
	if (!Pawn) return FVector::ZeroVector;

	FVector CurrentPosition = Pawn->GetActorLocation();

	// Priority 1: Command's target location (rally point)
	if (InstanceData.Context.CurrentCommand.TargetLocation != FVector::ZeroVector)
	{
		return InstanceData.Context.CurrentCommand.TargetLocation;
	}

	// Priority 2: Away from primary threat
	if (InstanceData.Context.PrimaryTarget)
	{
		FVector AwayFromThreat = (CurrentPosition - InstanceData.Context.PrimaryTarget->GetActorLocation()).GetSafeNormal();
		return CurrentPosition + AwayFromThreat * InstanceData.RetreatStepDistance;
	}

	// Fallback: Retreat backwards
	FVector BackwardDirection = -Pawn->GetActorForwardVector();
	return CurrentPosition + BackwardDirection * InstanceData.RetreatStepDistance;
}

AActor* FSTTask_ExecuteRetreat::FindNearestCover(FStateTreeExecutionContext& Context, const FVector& FromLocation) const
{
	FInstanceDataType& InstanceData = Context.GetInstanceData(*this);

	if (!InstanceData.Context.AIController) return nullptr;

	UWorld* World = InstanceData.Context.AIController->GetWorld();
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
			// Prefer cover that's away from threats
			if (InstanceData.Context.PrimaryTarget)
			{
				float CoverDistFromThreat = FVector::Dist(CoverActor->GetActorLocation(), InstanceData.Context.PrimaryTarget->GetActorLocation());
				float CurrentDistFromThreat = FVector::Dist(FromLocation, InstanceData.Context.PrimaryTarget->GetActorLocation());

				// Only use cover if it's farther from threat
				if (CoverDistFromThreat > CurrentDistFromThreat)
				{
					MinDistance = Distance;
					NearestCover = CoverActor;
				}
			}
			else
			{
				MinDistance = Distance;
				NearestCover = CoverActor;
			}
		}
	}

	return NearestCover;
}

float FSTTask_ExecuteRetreat::CalculateRetreatReward(FStateTreeExecutionContext& Context, float DeltaTime) const
{
	FInstanceDataType& InstanceData = Context.GetInstanceData(*this);

	float Reward = 0.0f;

	// Reward for increasing distance from threats
	if (InstanceData.DistanceFromThreat > InstanceData.MinSafeDistance)
	{
		Reward += 4.0f * DeltaTime; // +4.0 per second at safe distance
		InstanceData.bHasReachedSafeDistance = true;
	}

	// Reward for reaching cover
	if (InstanceData.Context.bInCover)
	{
		Reward += 8.0f; // One-time bonus for cover
	}

	// Reward for survival during retreat
	if (InstanceData.Context.bIsAlive && InstanceData.TimeInRetreat > 2.0f)
	{
		Reward += 5.0f * DeltaTime; // +5.0 per second for surviving retreat
	}

	// Bonus for reaching safe zone
	if (InstanceData.bHasReachedSafeDistance && !InstanceData.Context.bUnderFire)
	{
		Reward += 15.0f; // Large bonus for successful retreat
	}

	// Penalty for getting hit during retreat
	if (InstanceData.Context.bUnderFire)
	{
		Reward -= 2.0f * DeltaTime; // Small penalty for being under fire
	}

	return Reward;
}

bool FSTTask_ExecuteRetreat::ShouldCompleteRetreat(FStateTreeExecutionContext& Context) const
{
	FInstanceDataType& InstanceData = Context.GetInstanceData(*this);

	// Abort if dead
	if (!InstanceData.Context.bIsAlive)
	{
		return true;
	}

	// Abort if command changed/invalid
	if (!InstanceData.Context.bIsCommandValid)
	{
		return true;
	}

	// Complete if reached safe distance and destination
	if (InstanceData.bHasReachedSafeDistance && InstanceData.DistanceToRetreatDestination < 150.0f)
	{
		return true;
	}

	// Continue retreating
	return false;
}

void FSTTask_ExecuteRetreat::MoveToRetreatDestination(FStateTreeExecutionContext& Context, const FVector& Destination, float SpeedMultiplier) const
{
	FInstanceDataType& InstanceData = Context.GetInstanceData(*this);

	if (!InstanceData.Context.AIController) return;

	InstanceData.Context.AIController->MoveToLocation(Destination, 100.0f); // 1m acceptance radius

	// Adjust movement speed
	APawn* Pawn = InstanceData.Context.AIController->GetPawn();
	if (Pawn)
	{
		if (UCharacterMovementComponent* MovementComp = Pawn->FindComponentByClass<UCharacterMovementComponent>())
		{
			float BaseSpeed = 600.0f;
			MovementComp->MaxWalkSpeed = BaseSpeed * SpeedMultiplier;
		}
	}
}

void FSTTask_ExecuteRetreat::ProvideSuppressiveFire(FStateTreeExecutionContext& Context) const
{
	FInstanceDataType& InstanceData = Context.GetInstanceData(*this);

	if (!InstanceData.Context.PrimaryTarget || !InstanceData.Context.bWeaponReady)
	{
		return;
	}

	// Aim at threat while retreating
	if (InstanceData.Context.AIController)
	{
		InstanceData.Context.AIController->SetFocus(InstanceData.Context.PrimaryTarget);
	}

	// Fire weapon (lower accuracy due to moving)
	// TODO: Trigger weapon fire at reduced rate
}
