// Copyright Epic Games, Inc. All Rights Reserved.

#include "StateTree/Evaluators/STEvaluator_SpatialContext.h"
#include "StateTree/FollowerStateTreeContext.h"
#include "StateTree/FollowerStateTreeComponent.h"
#include "AIController.h"
#include "GameFramework/Pawn.h"
#include "NavigationSystem.h"
#include "DrawDebugHelpers.h"
#include "Kismet/GameplayStatics.h"

void FSTEvaluator_SpatialContext::TreeStart(FStateTreeExecutionContext& Context) const
{
	FInstanceDataType& InstanceData = Context.GetInstanceData(*this);

	UE_LOG(LogTemp, Log, TEXT("STEvaluator_SpatialContext: TreeStart"));

	InstanceData.TimeAccumulator = 0.0f;

	if (!InstanceData.StateTreeComp)
	{
		UE_LOG(LogTemp, Error, TEXT("[SPATIAL CONTEXT] TreeStart: ❌ StateTreeComp is null!"));
		return;
	}

	// Get SHARED context and initialize mask with defaults
	FFollowerStateTreeContext& SharedContext = InstanceData.StateTreeComp->GetSharedContext();
	SharedContext.ActionMask = FActionSpaceMask();
}

void FSTEvaluator_SpatialContext::Tick(FStateTreeExecutionContext& Context, float DeltaTime) const
{
	FInstanceDataType& InstanceData = Context.GetInstanceData(*this);

	if (!InstanceData.StateTreeComp)
	{
		return;
	}

	InstanceData.TimeAccumulator += DeltaTime;

	// Update at specified interval
	if (InstanceData.TimeAccumulator >= InstanceData.UpdateInterval)
	{
		InstanceData.TimeAccumulator = 0.0f;

		// Compute and update action mask in SHARED context
		FActionSpaceMask NewMask = ComputeActionMask(Context);
		FFollowerStateTreeContext& SharedContext = InstanceData.StateTreeComp->GetSharedContext();
		SharedContext.ActionMask = NewMask;
	}
}

void FSTEvaluator_SpatialContext::TreeStop(FStateTreeExecutionContext& Context) const
{
	UE_LOG(LogTemp, Log, TEXT("STEvaluator_SpatialContext: TreeStop"));
}

FActionSpaceMask FSTEvaluator_SpatialContext::ComputeActionMask(FStateTreeExecutionContext& Context) const
{
	FInstanceDataType& InstanceData = Context.GetInstanceData(*this);

	if (!InstanceData.StateTreeComp)
	{
		return FActionSpaceMask(); // Return default mask
	}

	// Get SHARED context (not a local copy!)
	FFollowerStateTreeContext& SharedContext = InstanceData.StateTreeComp->GetSharedContext();

	FActionSpaceMask Mask;

	// Get pawn and world
	APawn* Pawn = SharedContext.AIController ? SharedContext.AIController->GetPawn() : nullptr;
	if (!Pawn)
	{
		return Mask; // Return default mask
	}

	UWorld* World = Pawn->GetWorld();
	if (!World)
	{
		return Mask;
	}

	// 1. Detect indoor/corridor constraints
	bool bIndoor = DetectIndoor(Pawn, World, InstanceData.CorridorWidthThreshold);

	if (bIndoor)
	{
		// Restrict movement in narrow corridors
		float LateralClearance = MeasureLateralClearance(Pawn, World);

		if (LateralClearance < 100.0f) // Very narrow
		{
			Mask.bLockMovementY = true; // Lock lateral movement
			Mask.MaxSpeed = 0.5f; // Reduce speed
		}
		else if (LateralClearance < 200.0f) // Narrow
		{
			Mask.MaxSpeed = 0.7f; // Reduce speed
		}
	}

	// 2. Check NavMesh edge proximity
	float EdgeDistance = MeasureNavMeshEdgeDistance(Pawn, World);

	if (EdgeDistance < InstanceData.NavMeshEdgeDistance)
	{
		// Near edge, restrict movement away from center
		Mask.MaxSpeed = 0.5f; // Reduce speed near edges
		UE_LOG(LogTemp, Warning, TEXT("[SPATIAL] Near NavMesh edge (dist=%.1f), reducing speed"), EdgeDistance);
	}

	// 3. Cover aiming restrictions
	if (SharedContext.bInCover)
	{
		ApplyCoverAimingRestrictions(Mask, SharedContext);
	}

	// 4. Sprint availability
	Mask.bCanSprint = CanSprint(SharedContext, bIndoor);

	// 5. Crouch requirements
	if (bIndoor && SharedContext.bUnderFire)
	{
		Mask.bForceCrouch = true; // Force crouch when under fire indoors
	}

	// 6. Fire safety lock
	if (!IsFiringSafe(Pawn, World, SharedContext))
	{
		Mask.bSafetyLock = true; // Prevent friendly fire
		UE_LOG(LogTemp, Warning, TEXT("[SPATIAL] Firing UNSAFE, safety lock engaged"));
	}

	// Debug visualization
	if (InstanceData.bDrawDebugInfo)
	{
		FVector Location = Pawn->GetActorLocation();
		FColor MaskColor = Mask.bSafetyLock ? FColor::Red : (bIndoor ? FColor::Yellow : FColor::Green);
		DrawDebugSphere(World, Location + FVector(0, 0, 200), 50.0f, 8, MaskColor, false, InstanceData.UpdateInterval);
	}

	return Mask;
}

bool FSTEvaluator_SpatialContext::DetectIndoor(APawn* ControlledPawn, UWorld* World, float CorridorThreshold) const
{
	if (!ControlledPawn || !World)
	{
		return false;
	}

	// Simple indoor detection: trace upward to detect ceiling
	FVector StartLoc = ControlledPawn->GetActorLocation();
	FVector EndLoc = StartLoc + FVector(0, 0, 500.0f); // 5m up

	FHitResult HitResult;
	FCollisionQueryParams QueryParams;
	QueryParams.AddIgnoredActor(ControlledPawn);

	bool bHitCeiling = World->LineTraceSingleByChannel(
		HitResult,
		StartLoc,
		EndLoc,
		ECC_Visibility,
		QueryParams
	);

	// If hit ceiling within 5m, likely indoors
	return bHitCeiling && HitResult.Distance < 500.0f;
}

float FSTEvaluator_SpatialContext::MeasureLateralClearance(APawn* ControlledPawn, UWorld* World) const
{
	if (!ControlledPawn || !World)
	{
		return 1000.0f; // Large value = no constraint
	}

	FVector Location = ControlledPawn->GetActorLocation();
	FRotator Rotation = ControlledPawn->GetActorRotation();
	FVector RightDir = FRotationMatrix(Rotation).GetUnitAxis(EAxis::Y);

	// Trace left and right
	FHitResult LeftHit, RightHit;
	FCollisionQueryParams QueryParams;
	QueryParams.AddIgnoredActor(ControlledPawn);

	FVector LeftEnd = Location - RightDir * 300.0f; // 3m left
	FVector RightEnd = Location + RightDir * 300.0f; // 3m right

	bool bLeftBlocked = World->LineTraceSingleByChannel(LeftHit, Location, LeftEnd, ECC_Visibility, QueryParams);
	bool bRightBlocked = World->LineTraceSingleByChannel(RightHit, Location, RightEnd, ECC_Visibility, QueryParams);

	float LeftDist = bLeftBlocked ? LeftHit.Distance : 300.0f;
	float RightDist = bRightBlocked ? RightHit.Distance : 300.0f;

	float TotalClearance = LeftDist + RightDist;

	return TotalClearance;
}

float FSTEvaluator_SpatialContext::MeasureNavMeshEdgeDistance(APawn* ControlledPawn, UWorld* World) const
{
	if (!ControlledPawn || !World)
	{
		return 1000.0f; // Large value = no constraint
	}

	UNavigationSystemV1* NavSys = FNavigationSystem::GetCurrent<UNavigationSystemV1>(World);
	if (!NavSys)
	{
		return 1000.0f;
	}

	FVector Location = ControlledPawn->GetActorLocation();

	// Sample NavMesh in 8 directions, find minimum distance to invalid NavMesh
	float MinEdgeDistance = 1000.0f;
	const int32 NumDirections = 8;

	for (int32 i = 0; i < NumDirections; i++)
	{
		float Angle = (i * 360.0f / NumDirections) * PI / 180.0f;
		FVector Direction = FVector(FMath::Cos(Angle), FMath::Sin(Angle), 0.0f);
		FVector TestLoc = Location + Direction * 200.0f; // 2m out

		FNavLocation NavLoc;
		bool bValid = NavSys->ProjectPointToNavigation(TestLoc, NavLoc, FVector(100, 100, 100));

		if (!bValid)
		{
			// Found invalid NavMesh, calculate distance
			float Distance = FVector::Dist2D(Location, TestLoc);
			MinEdgeDistance = FMath::Min(MinEdgeDistance, Distance);
		}
	}

	return MinEdgeDistance;
}

void FSTEvaluator_SpatialContext::ApplyCoverAimingRestrictions(FActionSpaceMask& Mask, const FFollowerStateTreeContext& SharedContext) const
{
	// If in cover, restrict aiming angles to avoid exposing self
	if (SharedContext.CurrentCover)
	{
		// Limit pitch to prevent aiming over cover
		Mask.MinPitch = -30.0f; // Prevent looking down too much
		Mask.MaxPitch = 30.0f;  // Prevent exposing head

		// Limit yaw based on cover direction (simplified)
		Mask.MinYaw = -60.0f;
		Mask.MaxYaw = 60.0f;
	}
}

bool FSTEvaluator_SpatialContext::CanSprint(const FFollowerStateTreeContext& SharedContext, bool bIndoor) const
{
	// No sprinting indoors
	if (bIndoor)
	{
		return false;
	}

	// No sprinting when in cover
	if (SharedContext.bInCover)
	{
		return false;
	}

	// No sprinting when under fire
	if (SharedContext.bUnderFire)
	{
		return false;
	}

	return true;
}

bool FSTEvaluator_SpatialContext::IsFiringSafe(APawn* ControlledPawn, UWorld* World, const FFollowerStateTreeContext& SharedContext) const
{
	if (!ControlledPawn || !World || !SharedContext.PrimaryTarget)
	{
		return false; // No target = not safe to fire
	}

	// Check if any friendly is in line of fire
	FVector StartLoc = ControlledPawn->GetActorLocation() + FVector(0, 0, 80.0f);
	FVector TargetLoc = SharedContext.PrimaryTarget->GetActorLocation() + FVector(0, 0, 80.0f);
	FVector Direction = (TargetLoc - StartLoc).GetSafeNormal();
	float Distance = FVector::Dist(StartLoc, TargetLoc);

	// Perform sphere sweep to detect friendlies in firing cone
	FHitResult HitResult;
	FCollisionQueryParams QueryParams;
	QueryParams.AddIgnoredActor(ControlledPawn);
	QueryParams.AddIgnoredActor(SharedContext.PrimaryTarget);

	bool bHit = World->SweepSingleByChannel(
		HitResult,
		StartLoc,
		TargetLoc,
		FQuat::Identity,
		ECC_Pawn,
		FCollisionShape::MakeSphere(50.0f), // 50cm radius
		QueryParams
	);

	if (bHit && HitResult.GetActor())
	{
		// Check if hit actor is a friendly (same team)
		// This requires team checking logic (simplified here)
		UE_LOG(LogTemp, Warning, TEXT("[SPATIAL] Potential friendly in line of fire: %s"), *HitResult.GetActor()->GetName());
		return false; // Friendly in line of fire
	}

	return true; // Safe to fire
}
