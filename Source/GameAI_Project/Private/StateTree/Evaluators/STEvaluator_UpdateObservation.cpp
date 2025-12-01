// Copyright Epic Games, Inc. All Rights Reserved.

#include "StateTree/Evaluators/STEvaluator_UpdateObservation.h"
#include "StateTree/FollowerStateTreeContext.h"
#include "StateTree/FollowerStateTreeComponent.h"
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

	UE_LOG(LogTemp, Warning, TEXT("[UPDATE OBS EVALUATOR] TreeStart called - UpdateInterval=%.3f"),
		InstanceData.UpdateInterval);
}

void FSTEvaluator_UpdateObservation::Tick(FStateTreeExecutionContext& Context, float DeltaTime) const
{
	FInstanceDataType& InstanceData = Context.GetInstanceData(*this);

	// Validate context components (FollowerComponent is REQUIRED, AIController is OPTIONAL for Schola)
	if (!InstanceData.Context.FollowerComponent)
	{
		static bool bLoggedOnce = false;
		if (!bLoggedOnce)
		{
			UE_LOG(LogTemp, Error, TEXT("[UPDATE OBS] Missing FollowerComponent!"));
			bLoggedOnce = true;
		}
		return;
	}

	// Get Pawn from either AIController (normal AI) or directly from owner (Schola)
	APawn* ControlledPawn = nullptr;
	if (InstanceData.Context.AIController)
	{
		// Normal AI mode: Get pawn from AIController
		ControlledPawn = InstanceData.Context.AIController->GetPawn();
	}
	else
	{
		// Schola mode: Get pawn directly from component owner
		UFollowerStateTreeComponent* StateTreeComp = Cast<UFollowerStateTreeComponent>(
			InstanceData.Context.FollowerComponent->GetOwner()->FindComponentByClass<UFollowerStateTreeComponent>()
		);
		if (StateTreeComp)
		{
			ControlledPawn = Cast<APawn>(StateTreeComp->GetOwner());
		}

		static bool bLoggedScholaOnce = false;
		if (!bLoggedScholaOnce)
		{
			UE_LOG(LogTemp, Warning, TEXT("[UPDATE OBS] Running in SCHOLA MODE (no AIController) for '%s'"),
				*GetNameSafe(ControlledPawn));
			bLoggedScholaOnce = true;
		}
	}

	if (!ControlledPawn)
	{
		static bool bLoggedPawnOnce = false;
		if (!bLoggedPawnOnce)
		{
			UE_LOG(LogTemp, Error, TEXT("[UPDATE OBS] Cannot get Pawn (AIController=%s, Owner=%s)!"),
				InstanceData.Context.AIController ? TEXT("Valid") : TEXT("NULL"),
				*GetNameSafe(InstanceData.Context.FollowerComponent->GetOwner()));
			bLoggedPawnOnce = true;
		}
		return;
	}

	// Check interval - but still update critical combat data every tick
	InstanceData.TimeAccumulator += DeltaTime;
	bool bFullUpdate = (InstanceData.TimeAccumulator >= InstanceData.UpdateInterval);

	// DEBUG: Log update frequency
	static int32 TickCount = 0;
	if (++TickCount % 60 == 0)
	{
		UE_LOG(LogTemp, Display, TEXT("[UPDATE OBS] '%s': TimeAccum=%.3f, UpdateInterval=%.3f, bFullUpdate=%d"),
			*ControlledPawn->GetName(), InstanceData.TimeAccumulator, InstanceData.UpdateInterval, bFullUpdate ? 1 : 0);
	}

	if (bFullUpdate)
	{
		InstanceData.TimeAccumulator = 0.0f;
	}

	// Full update (observations, perception, cover) - run at intervals
	if (bFullUpdate)
	{
		UE_LOG(LogTemp, Warning, TEXT("[UPDATE OBS] '%s': ⏰ FULL UPDATE triggered (every %.2fs)"), 
			*ControlledPawn->GetName(), InstanceData.UpdateInterval);

		// Get observation from follower component
		InstanceData.Context.PreviousObservation = InstanceData.Context.CurrentObservation;
		InstanceData.Context.CurrentObservation = InstanceData.Context.FollowerComponent->GetLocalObservation();

		// Update target tracking from perception system
		ScanForEnemies(InstanceData, ControlledPawn, ControlledPawn->GetWorld());

		// Update cover state
		DetectCover(InstanceData, ControlledPawn, ControlledPawn->GetWorld());
	}
	else
	{
		// Log when NOT doing full update to see timing
		static int32 SkipCount = 0;
		if (++SkipCount % 300 == 0) // Log every 5 seconds at 60fps
		{
			UE_LOG(LogTemp, Display, TEXT("[UPDATE OBS] '%s': Skipping full update (TimeAccum=%.3fs < %.3fs), PrimaryTarget='%s'"),
				*ControlledPawn->GetName(), 
				InstanceData.TimeAccumulator, 
				InstanceData.UpdateInterval,
				*GetNameSafe(InstanceData.Context.PrimaryTarget));
		}
	}

	// CRITICAL: Update combat state EVERY tick (LOS, distance) - needed for firing
	UpdateCombatState(InstanceData, ControlledPawn);

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
		UE_LOG(LogTemp, Error, TEXT("[SCAN ENEMIES] '%s': ControlledPawn or World is NULL"),
			*GetNameSafe(ControlledPawn));
		return;
	}

	// Get perception component
	UAgentPerceptionComponent* PerceptionComp = ControlledPawn->FindComponentByClass<UAgentPerceptionComponent>();
	if (!PerceptionComp)
	{
		UE_LOG(LogTemp, Warning, TEXT("[SCAN ENEMIES] '%s': No AgentPerceptionComponent found"),
			*ControlledPawn->GetName());
		// No perception component, clear enemies
		InstanceData.Context.VisibleEnemies.Empty();
		InstanceData.Context.PrimaryTarget = nullptr;
		return;
	}

	// Get detected enemies from perception system (team-based filtering)
	TArray<AActor*> DetectedEnemies = PerceptionComp->GetDetectedEnemies();

	UE_LOG(LogTemp, Display, TEXT("[SCAN ENEMIES] '%s': Perception detected %d enemies (team-based)"),
		*ControlledPawn->GetName(), DetectedEnemies.Num());

	// FALLBACK: If team system returns no enemies, use raw perception data
	// This handles Schola training mode where SimulationManager team registration may not be set up
	if (DetectedEnemies.Num() == 0)
	{
		TArray<AActor*> RawPerceivedActors;
		PerceptionComp->GetCurrentlyPerceivedActors(nullptr, RawPerceivedActors);

		UE_LOG(LogTemp, Warning, TEXT("[SCAN ENEMIES] '%s': Team system returned 0 enemies, using RAW perception (%d actors)"),
			*ControlledPawn->GetName(), RawPerceivedActors.Num());

		// Filter out self and sort by distance
		FVector OwnerLocation = ControlledPawn->GetActorLocation();
		for (AActor* Actor : RawPerceivedActors)
		{
			if (Actor && Actor != ControlledPawn)
			{
				// Skip dead actors
				UHealthComponent* HealthComp = Actor->FindComponentByClass<UHealthComponent>();
				if (HealthComp && HealthComp->IsDead())
				{
					continue;
				}

				DetectedEnemies.Add(Actor);
			}
		}

		// Sort by distance (nearest first)
		DetectedEnemies.Sort([OwnerLocation](const AActor& A, const AActor& B)
		{
			float DistA = FVector::DistSquared(OwnerLocation, A.GetActorLocation());
			float DistB = FVector::DistSquared(OwnerLocation, B.GetActorLocation());
			return DistA < DistB;
		});

		UE_LOG(LogTemp, Warning, TEXT("[SCAN ENEMIES] '%s': FALLBACK mode found %d valid targets"),
			*ControlledPawn->GetName(), DetectedEnemies.Num());
	}

	// Update visible enemies list in context
	InstanceData.Context.VisibleEnemies.Empty();
	for (AActor* Enemy : DetectedEnemies)
	{
		if (Enemy)
		{
			InstanceData.Context.VisibleEnemies.Add(Enemy);
			float Distance = FVector::Dist(ControlledPawn->GetActorLocation(), Enemy->GetActorLocation());
			UE_LOG(LogTemp, Display, TEXT("  → Enemy '%s' at distance %.1f cm"),
				*Enemy->GetName(), Distance);
		}
	}


	// Set primary target - PRIORITIZE objective target over perception target
	AActor* ObjectiveTarget = InstanceData.Context.CurrentObjective ? InstanceData.Context.CurrentObjective->TargetActor : nullptr;

	if (ObjectiveTarget && ObjectiveTarget->IsValidLowLevel() && !ObjectiveTarget->IsPendingKillPending())
	{
		// Use objective-specified target if valid
		InstanceData.Context.PrimaryTarget = ObjectiveTarget;
		float Distance = FVector::Dist(ControlledPawn->GetActorLocation(), ObjectiveTarget->GetActorLocation());

		UE_LOG(LogTemp, Warning, TEXT("[SCAN ENEMIES] '%s': PRIMARY TARGET set to OBJECTIVE target '%s' at %.1f cm"),
			*ControlledPawn->GetName(), *ObjectiveTarget->GetName(), Distance);

		if (InstanceData.bDrawDebugInfo)
		{
			FVector TargetLocation = ObjectiveTarget->GetActorLocation();
			DrawDebugLine(World, ControlledPawn->GetActorLocation(), TargetLocation,
				FColor::Orange, false, 0.2f, 0, 3.0f); // Orange for objective target
			DrawDebugSphere(World, TargetLocation, 50.0f, 12, FColor::Orange, false, 0.2f);
		}
	}
	else if (DetectedEnemies.Num() > 0)
	{
		// Fall back to nearest detected enemy if no command target
		InstanceData.Context.PrimaryTarget = DetectedEnemies[0]; // Already sorted by distance
		float Distance = FVector::Dist(ControlledPawn->GetActorLocation(), DetectedEnemies[0]->GetActorLocation());

		UE_LOG(LogTemp, Warning, TEXT("[SCAN ENEMIES] '%s': PRIMARY TARGET set to NEAREST enemy '%s' at %.1f cm"),
			*ControlledPawn->GetName(), *DetectedEnemies[0]->GetName(), Distance);

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
		UE_LOG(LogTemp, Display, TEXT("[SCAN ENEMIES] '%s': No enemies detected, PRIMARY TARGET cleared"),
			*ControlledPawn->GetName());
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
		UE_LOG(LogTemp, Error, TEXT("[UPDATE COMBAT] ControlledPawn is NULL"));
		return;
	}

	UWorld* World = ControlledPawn->GetWorld();
	if (!World)
	{
		UE_LOG(LogTemp, Error, TEXT("[UPDATE COMBAT] '%s': World is NULL"), *ControlledPawn->GetName());
		return;
	}

	// Check LOS and distance to primary target
	if (InstanceData.Context.PrimaryTarget)
	{
		// Use eye height for LOS trace (avoid hitting ground)
		FVector StartLocation = ControlledPawn->GetActorLocation() + FVector(0, 0, 80.0f); // Eye height offset
		FVector TargetLocation = InstanceData.Context.PrimaryTarget->GetActorLocation() + FVector(0, 0, 80.0f);

		// Calculate distance
		float Distance = FVector::Dist(StartLocation, TargetLocation);
		InstanceData.Context.CurrentObservation.DistanceToNearestEnemy = Distance;

		UE_LOG(LogTemp, Display, TEXT("[UPDATE COMBAT] '%s': Checking LOS to target '%s' at distance %.1f cm"),
			*ControlledPawn->GetName(), *InstanceData.Context.PrimaryTarget->GetName(), Distance);
		UE_LOG(LogTemp, Display, TEXT("  → Start: (%.1f, %.1f, %.1f), Target: (%.1f, %.1f, %.1f)"),
			StartLocation.X, StartLocation.Y, StartLocation.Z,
			TargetLocation.X, TargetLocation.Y, TargetLocation.Z);

		// Check line of sight
		FHitResult HitResult;
		FCollisionQueryParams QueryParams;
		QueryParams.AddIgnoredActor(ControlledPawn);
		QueryParams.bTraceComplex = false;
		QueryParams.bReturnPhysicalMaterial = false;

		bool bHit = World->LineTraceSingleByChannel(
			HitResult,
			StartLocation,
			TargetLocation,
			ECC_Visibility,
			QueryParams
		);

		UE_LOG(LogTemp, Display, TEXT("  → LineTrace result: bHit=%d"), bHit ? 1 : 0);

		if (bHit)
		{
			AActor* HitActor = HitResult.GetActor();
			UE_LOG(LogTemp, Display, TEXT("  → Hit Actor: '%s' at distance %.1f cm"),
				*GetNameSafe(HitActor), HitResult.Distance);
			UE_LOG(LogTemp, Display, TEXT("  → Hit Location: (%.1f, %.1f, %.1f)"),
				HitResult.Location.X, HitResult.Location.Y, HitResult.Location.Z);
			UE_LOG(LogTemp, Display, TEXT("  → Hit Component: '%s'"),
				*GetNameSafe(HitResult.GetComponent()));
			UE_LOG(LogTemp, Display, TEXT("  → Is Target? %d"),
				(HitActor == InstanceData.Context.PrimaryTarget) ? 1 : 0);
		}
		else
		{
			UE_LOG(LogTemp, Display, TEXT("  → No blocking hit detected (clear LOS)"));
		}

		// Has LOS if hit the target or no blocking hit
		InstanceData.Context.bHasLOS = !bHit || HitResult.GetActor() == InstanceData.Context.PrimaryTarget;

		UE_LOG(LogTemp, Warning, TEXT("[UPDATE COMBAT] '%s': bHasLOS = %d (bHit=%d, HitActor='%s', Target='%s')"),
			*ControlledPawn->GetName(),
			InstanceData.Context.bHasLOS ? 1 : 0,
			bHit ? 1 : 0,
			*GetNameSafe(HitResult.GetActor()),
			*InstanceData.Context.PrimaryTarget->GetName());

		// Debug: Log what blocked LOS
		if (bHit && HitResult.GetActor() != InstanceData.Context.PrimaryTarget)
		{
			UE_LOG(LogTemp, Error, TEXT("[UPDATE COMBAT] '%s': ❌ LOS BLOCKED by '%s' at distance %.1f cm"),
				*ControlledPawn->GetName(),
				HitResult.GetActor() ? *HitResult.GetActor()->GetName() : TEXT("Unknown"),
				HitResult.Distance);
		}

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
		UE_LOG(LogTemp, Display, TEXT("[UPDATE COMBAT] '%s': No PrimaryTarget, bHasLOS = false"),
			*ControlledPawn->GetName());
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
