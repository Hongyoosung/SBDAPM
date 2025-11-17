// Copyright Epic Games, Inc. All Rights Reserved.

#include "StateTree/Evaluators/STEvaluator_UpdateObservation.h"
#include "StateTree/FollowerStateTreeContext.h"
#include "Team/FollowerAgentComponent.h"
#include "Perception/AgentPerceptionComponent.h"
#include "Combat/WeaponComponent.h"
#include "Combat/HealthComponent.h"
#include "AIController.h"
#include "GameFramework/Pawn.h"
#include "Kismet/GameplayStatics.h"
#include "DrawDebugHelpers.h"
#include "Engine\OverlapResult.h"


void FSTEvaluator_UpdateObservation::TreeStart(FStateTreeExecutionContext& Context) const
{
	FInstanceDataType& InstanceData = Context.GetInstanceData(*this);

	// Initialize time accumulator
	InstanceData.TimeAccumulator = 0.0f;
}

void FSTEvaluator_UpdateObservation::Tick(FStateTreeExecutionContext& Context, float DeltaTime) const
{
	FInstanceDataType& InstanceData = Context.GetInstanceData(*this);

	// Check interval
	InstanceData.TimeAccumulator += DeltaTime;
	if (InstanceData.TimeAccumulator < InstanceData.UpdateInterval)
	{
		return;
	}

	InstanceData.TimeAccumulator = 0.0f;

	// Validate context components (auto-bound from schema)
	if (!InstanceData.Context.FollowerComponent || !InstanceData.Context.AIController)
	{
		return;
	}

	APawn* ControlledPawn = InstanceData.Context.AIController->GetPawn();
	if (!ControlledPawn)
	{
		return;
	}

	// Get observation from follower component
	InstanceData.Context.PreviousObservation = InstanceData.Context.CurrentObservation;
	InstanceData.Context.CurrentObservation = InstanceData.Context.FollowerComponent->GetLocalObservation();

	// Update target tracking from perception system
	ScanForEnemies(InstanceData, ControlledPawn, ControlledPawn->GetWorld());

	// Update combat state (LOS, under fire, etc.)
	UpdateCombatState(InstanceData, ControlledPawn);

	// Update cover state
	DetectCover(InstanceData, ControlledPawn, ControlledPawn->GetWorld());

	// Update distance to primary target
	if (InstanceData.Context.PrimaryTarget)
	{
		InstanceData.Context.DistanceToPrimaryTarget = FVector::Dist(
			ControlledPawn->GetActorLocation(),
			InstanceData.Context.PrimaryTarget->GetActorLocation()
		);
	}
	else
	{
		InstanceData.Context.DistanceToPrimaryTarget = 99999.0f;
	}

	// NOTE: No need to "write back" to external context - InstanceData.Context IS the shared context
	// All tasks/conditions/evaluators access the same FFollowerStateTreeContext instance
}

void FSTEvaluator_UpdateObservation::TreeStop(FStateTreeExecutionContext& Context) const
{
	// No cleanup needed
}

FObservationElement FSTEvaluator_UpdateObservation::GatherObservationData(FStateTreeExecutionContext& Context) const
{
	// Placeholder - full implementation would gather all 71 features
	// For now, return empty observation
	return FObservationElement();
}

void FSTEvaluator_UpdateObservation::UpdateAgentState(FObservationElement& Observation, APawn* ControlledPawn) const
{
	// Placeholder
}

void FSTEvaluator_UpdateObservation::PerformRaycastPerception(FObservationElement& Observation, APawn* ControlledPawn, UWorld* World) const
{
	// Placeholder
}

void FSTEvaluator_UpdateObservation::ScanForEnemies(FSTEvaluator_UpdateObservationInstanceData& InstanceData, APawn* ControlledPawn, UWorld* World) const
{
	if (!ControlledPawn || !World)
	{
		return;
	}

	// Get perception component
	UAgentPerceptionComponent* PerceptionComp = ControlledPawn->FindComponentByClass<UAgentPerceptionComponent>();
	if (!PerceptionComp)
	{
		// No perception component, clear enemies
		InstanceData.Context.VisibleEnemies.Empty();
		InstanceData.Context.PrimaryTarget = nullptr;
		return;
	}

	// Get detected enemies from perception system
	TArray<AActor*> DetectedEnemies = PerceptionComp->GetDetectedEnemies();

	// Update visible enemies list in context
	InstanceData.Context.VisibleEnemies.Empty();
	for (AActor* Enemy : DetectedEnemies)
	{
		if (Enemy)
		{
			InstanceData.Context.VisibleEnemies.Add(Enemy);
		}
	}

	// Set primary target - PRIORITIZE command target over perception target
	AActor* CommandTarget = InstanceData.Context.CurrentCommand.TargetActor;

	if (CommandTarget && CommandTarget->IsValidLowLevel() && !CommandTarget->IsPendingKillPending())
	{
		// Use command-specified target if valid
		InstanceData.Context.PrimaryTarget = CommandTarget;

		if (InstanceData.bDrawDebugInfo)
		{
			FVector TargetLocation = CommandTarget->GetActorLocation();
			DrawDebugLine(World, ControlledPawn->GetActorLocation(), TargetLocation,
				FColor::Orange, false, 0.2f, 0, 3.0f); // Orange for command target
			DrawDebugSphere(World, TargetLocation, 50.0f, 12, FColor::Orange, false, 0.2f);
		}
	}
	else if (DetectedEnemies.Num() > 0)
	{
		// Fall back to nearest detected enemy if no command target
		InstanceData.Context.PrimaryTarget = DetectedEnemies[0]; // Already sorted by distance

		if (InstanceData.bDrawDebugInfo)
		{
			FVector TargetLocation = InstanceData.Context.PrimaryTarget->GetActorLocation();
			DrawDebugLine(World, ControlledPawn->GetActorLocation(), TargetLocation,
				FColor::Red, false, 0.2f, 0, 2.0f);
			DrawDebugSphere(World, TargetLocation, 50.0f, 12, FColor::Red, false, 0.2f);
		}
	}
	else
	{
		InstanceData.Context.PrimaryTarget = nullptr;
	}
}

void FSTEvaluator_UpdateObservation::DetectCover(FSTEvaluator_UpdateObservationInstanceData& InstanceData, APawn* ControlledPawn, UWorld* World) const
{
	if (!ControlledPawn || !World)
	{
		return;
	}

	// Simple cover detection: check for nearby actors tagged with "Cover"
	FVector AgentLocation = ControlledPawn->GetActorLocation();
	float CoverSearchRadius = 500.0f; // 5 meters

	TArray<FOverlapResult> OverlapResults;
	FCollisionQueryParams QueryParams;
	QueryParams.AddIgnoredActor(ControlledPawn);

	bool bFoundCover = World->OverlapMultiByChannel(
		OverlapResults,
		AgentLocation,
		FQuat::Identity,
		ECC_WorldStatic,
		FCollisionShape::MakeSphere(CoverSearchRadius),
		QueryParams
	);

	InstanceData.Context.bInCover = false;

	if (bFoundCover)
	{
		// Check if any overlapped actor is tagged as cover
		for (const FOverlapResult& Result : OverlapResults)
		{
			AActor* OverlappedActor = Result.GetActor();
			if (OverlappedActor && OverlappedActor->ActorHasTag(FName("Cover")))
			{
				// Check if agent is within close proximity to cover
				float DistanceToCover = FVector::Dist(AgentLocation, OverlappedActor->GetActorLocation());
				if (DistanceToCover < 200.0f) // 2 meters
				{
					InstanceData.Context.bInCover = true;
					InstanceData.Context.CurrentCover = OverlappedActor;

					if (InstanceData.bDrawDebugInfo)
					{
						DrawDebugSphere(World, OverlappedActor->GetActorLocation(), 100.0f,
							12, FColor::Blue, false, 0.2f);
					}
					break;
				}
			}
		}
	}
}

void FSTEvaluator_UpdateObservation::UpdateCombatState(FSTEvaluator_UpdateObservationInstanceData& InstanceData, APawn* ControlledPawn) const
{
	if (!ControlledPawn)
	{
		return;
	}

	UWorld* World = ControlledPawn->GetWorld();
	if (!World)
	{
		return;
	}

	// Check LOS and distance to primary target
	if (InstanceData.Context.PrimaryTarget)
	{
		FVector StartLocation = ControlledPawn->GetActorLocation();
		FVector TargetLocation = InstanceData.Context.PrimaryTarget->GetActorLocation();

		// Calculate distance
		float Distance = FVector::Dist(StartLocation, TargetLocation);
		InstanceData.Context.CurrentObservation.DistanceToNearestEnemy = Distance;

		// Check line of sight
		FHitResult HitResult;
		FCollisionQueryParams QueryParams;
		QueryParams.AddIgnoredActor(ControlledPawn);

		bool bHit = World->LineTraceSingleByChannel(
			HitResult,
			StartLocation,
			TargetLocation,
			ECC_Visibility,
			QueryParams
		);

		// Has LOS if hit the target or no blocking hit
		InstanceData.Context.bHasLOS = !bHit || HitResult.GetActor() == InstanceData.Context.PrimaryTarget;

		if (InstanceData.bDrawDebugInfo)
		{
			FColor LOSColor = InstanceData.Context.bHasLOS ? FColor::Green : FColor::Yellow;
			DrawDebugLine(World, StartLocation, TargetLocation, LOSColor, false, 0.2f, 0, 1.0f);
		}
	}
	else
	{
		InstanceData.Context.bHasLOS = false;
		InstanceData.Context.CurrentObservation.DistanceToNearestEnemy = 99999.0f;
	}

	// Check if under fire (simplified - check if health component recently took damage)
	if (UHealthComponent* HealthComp = ControlledPawn->FindComponentByClass<UHealthComponent>())
	{
		// Consider "under fire" if took damage in last 2 seconds
		float TimeSinceLastDamage = HealthComp->GetTimeSinceLastDamage();
		InstanceData.Context.bUnderFire = (TimeSinceLastDamage >= 0.0f && TimeSinceLastDamage < 2.0f);
	}
	else
	{
		InstanceData.Context.bUnderFire = false;
	}
}

ERaycastHitType FSTEvaluator_UpdateObservation::ClassifyHitType(const FHitResult& HitResult) const
{
	return ERaycastHitType::None;
}

ETerrainType FSTEvaluator_UpdateObservation::DetectTerrainType(APawn* ControlledPawn) const
{
	return ETerrainType::Flat;
}
