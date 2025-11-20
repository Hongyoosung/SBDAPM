// Copyright Epic Games, Inc. All Rights Reserved.

#include "StateTree/Tasks/STTask_ExecuteDefend.h"
#include "StateTree/FollowerStateTreeContext.h"
#include "Team/FollowerAgentComponent.h"
#include "Combat/WeaponComponent.h"
#include "Combat/HealthComponent.h"
#include "AIController.h"
#include "NavigationSystem.h"
#include "GameFramework/Pawn.h"
#include "DrawDebugHelpers.h"
#include "Kismet/GameplayStatics.h"

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

EStateTreeRunStatus FSTTask_ExecuteDefend::EnterState(FStateTreeExecutionContext& Context, const FStateTreeTransitionResult& Transition) const
{
	FInstanceDataType& InstanceData = Context.GetInstanceData(*this);

	// Validate inputs
	if (!InstanceData.Context.FollowerComponent || !InstanceData.Context.AIController)
	{
		UE_LOG(LogTemp, Error, TEXT("STTask_ExecuteDefend: Invalid inputs (missing component/controller)"));
		return EStateTreeRunStatus::Failed;
	}

	// Initialize defend position
	if (InstanceData.Context.CurrentCommand.TargetLocation != FVector::ZeroVector)
	{
		InstanceData.DefendPosition = InstanceData.Context.CurrentCommand.TargetLocation;
	}
	else if (APawn* Pawn = InstanceData.Context.AIController->GetPawn())
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
	if (Reward != 0.0f && InstanceData.Context.FollowerComponent)
	{
		InstanceData.Context.FollowerComponent->ProvideReward(Reward, false);
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

	switch (InstanceData.Context.CurrentTacticalAction)
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

	APawn* Pawn = InstanceData.Context.AIController ? InstanceData.Context.AIController->GetPawn() : nullptr;
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
		InstanceData.Context.AIController->StopMovement();
		EngageThreats(Context, InstanceData.Context.bInCover ? InstanceData.CoverAccuracyBonus : 1.0f);
	}
}

void FSTTask_ExecuteDefend::ExecuteSeekCover(FStateTreeExecutionContext& Context, float DeltaTime) const
{
	FInstanceDataType& InstanceData = Context.GetInstanceData(*this);

	APawn* Pawn = InstanceData.Context.AIController ? InstanceData.Context.AIController->GetPawn() : nullptr;
	if (!Pawn) return;

	// Find nearest cover if not already in cover
	if (!InstanceData.Context.bInCover || !InstanceData.Context.CurrentCover)
	{
		AActor* NearestCover = FindNearestCover(Context, Pawn->GetActorLocation());
		if (NearestCover)
		{
			InstanceData.Context.NearestCoverLocation = NearestCover->GetActorLocation();
			MoveToDefensivePosition(Context, InstanceData.Context.NearestCoverLocation, DeltaTime);
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

	APawn* Pawn = InstanceData.Context.AIController ? InstanceData.Context.AIController->GetPawn() : nullptr;
	if (!Pawn) return;

	// Retreat away from threats toward defend position
	FVector RetreatDirection = (InstanceData.DefendPosition - Pawn->GetActorLocation()).GetSafeNormal();
	FVector RetreatDestination = Pawn->GetActorLocation() + RetreatDirection * 500.0f; // 5m retreat

	MoveToDefensivePosition(Context, RetreatDestination, DeltaTime);
}

AActor* FSTTask_ExecuteDefend::FindNearestCover(FStateTreeExecutionContext& Context, const FVector& FromLocation) const
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
			MinDistance = Distance;
			NearestCover = CoverActor;
		}
	}

	if (NearestCover)
	{
		InstanceData.Context.DistanceToNearestCover = MinDistance;
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
		Reward += 0.2f * DeltaTime; // +0.3 per second
	}

	// Reward for using cover
	if (InstanceData.Context.bInCover)
	{
		Reward += 0.3f * DeltaTime; // +0.3 per second in cover
	}

	// Reward for survival under fire
	if (InstanceData.Context.bUnderFire && InstanceData.Context.bIsAlive)
	{
		Reward += 0.4f * DeltaTime; // +0.4 per second surviving under fire
	}

	return Reward;
}

bool FSTTask_ExecuteDefend::ShouldCompleteDefense(FStateTreeExecutionContext& Context) const
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

	// Continue defending
	return false;
}

void FSTTask_ExecuteDefend::MoveToDefensivePosition(FStateTreeExecutionContext& Context, const FVector& Destination, float DeltaTime) const
{
	FInstanceDataType& InstanceData = Context.GetInstanceData(*this);

	if (!InstanceData.Context.AIController) return;

	InstanceData.Context.AIController->MoveToLocation(Destination, 50.0f); // 50cm acceptance radius
}

void FSTTask_ExecuteDefend::EngageThreats(FStateTreeExecutionContext& Context, float AccuracyModifier) const
{
	FInstanceDataType& InstanceData = Context.GetInstanceData(*this);

	APawn* Pawn = InstanceData.Context.AIController ? InstanceData.Context.AIController->GetPawn() : nullptr;
	if (!Pawn)
	{
		return;
	}

	// Validate and update primary target if needed
	if (!IsTargetValid(InstanceData.Context.PrimaryTarget))
	{
		AActor* NewTarget = FindNearestValidEnemy(InstanceData.Context.VisibleEnemies, Pawn);
		InstanceData.Context.PrimaryTarget = NewTarget;

		if (NewTarget)
		{
			UE_LOG(LogTemp, Log, TEXT("[DEFEND TASK] '%s': Target died, switching to '%s'"),
				*Pawn->GetName(), *NewTarget->GetName());
		}
	}

	if (!InstanceData.Context.PrimaryTarget)
	{
		// No valid targets - clear focus
		if (InstanceData.Context.AIController)
		{
			InstanceData.Context.AIController->ClearFocus(EAIFocusPriority::Gameplay);
		}
		return;
	}

	// Focus on primary target
	if (InstanceData.Context.AIController)
	{
		InstanceData.Context.AIController->SetFocus(InstanceData.Context.PrimaryTarget);
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
		FVector EndLoc = InstanceData.Context.PrimaryTarget->GetActorLocation() + FVector(0, 0, 80.0f);

		bool bHit = World->LineTraceSingleByChannel(HitResult, StartLoc, EndLoc, ECC_Visibility, QueryParams);
		bHasLOS = !bHit || HitResult.GetActor() == InstanceData.Context.PrimaryTarget;

		// Debug visualization
		DrawDebugLine(World, StartLoc, EndLoc, bHasLOS ? FColor::Green : FColor::Red, false, 0.1f, 0, 2.0f);
	}

	// Fire weapon at target
	UWeaponComponent* WeaponComp = Pawn->FindComponentByClass<UWeaponComponent>();
	if (WeaponComp && WeaponComp->CanFire() && bHasLOS)
	{
		WeaponComp->FireAtTarget(InstanceData.Context.PrimaryTarget, true);
		UE_LOG(LogTemp, Log, TEXT("[DEFEND TASK] '%s': FIRING at target '%s' (Accuracy: %.1f)"),
			*Pawn->GetName(),
			*InstanceData.Context.PrimaryTarget->GetName(),
			AccuracyModifier);
	}
}
