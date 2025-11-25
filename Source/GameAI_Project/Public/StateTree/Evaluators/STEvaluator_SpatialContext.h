// Copyright Epic Games, Inc. All Rights Reserved.

#pragma once

#include "CoreMinimal.h"
#include "StateTreeEvaluatorBase.h"
#include "RL/RLTypes.h"
#include "StateTree/FollowerStateTreeSchema.h"
#include "STEvaluator_SpatialContext.generated.h"

/**
 * State Tree Evaluator: Spatial Context
 *
 * Computes the action space mask (FActionSpaceMask) based on environmental constraints.
 * This evaluator analyzes the agent's surroundings to determine valid action ranges:
 *
 * 1. Movement constraints (indoor corridors, NavMesh edges)
 * 2. Aiming constraints (cover restrictions, LOS blockers)
 * 3. Action availability (sprint, crouch, fire safety)
 *
 * The mask is used by the RL policy to constrain atomic actions to valid ranges,
 * preventing illegal actions like sprinting indoors or moving off NavMesh edges.
 *
 * Sprint 3 - Atomic Action Space
 */

USTRUCT()
struct GAMEAI_PROJECT_API FSTEvaluator_SpatialContextInstanceData
{
	GENERATED_BODY()

	/** Shared context struct - auto-binds to FollowerContext from schema */
	UPROPERTY(EditAnywhere, Category = "Context")
	FFollowerStateTreeContext Context;

	/** Update interval (seconds) - 0.2 = 5Hz */
	UPROPERTY(EditAnywhere, Category = "Config", meta = (ClampMin = "0.05", ClampMax = "0.5"))
	float UpdateInterval = 0.2f;

	/** Corridor width threshold for indoor detection (cm) */
	UPROPERTY(EditAnywhere, Category = "Config", meta = (ClampMin = "100.0", ClampMax = "500.0"))
	float CorridorWidthThreshold = 300.0f;

	/** Distance to NavMesh edge to restrict movement (cm) */
	UPROPERTY(EditAnywhere, Category = "Config", meta = (ClampMin = "50.0", ClampMax = "300.0"))
	float NavMeshEdgeDistance = 150.0f;

	/** Draw debug visualizations */
	UPROPERTY(EditAnywhere, Category = "Debug")
	bool bDrawDebugInfo = false;

	/** Time accumulator for update interval */
	UPROPERTY()
	float TimeAccumulator = 0.0f;
};

USTRUCT(meta = (DisplayName = "Spatial Context", BlueprintType))
struct GAMEAI_PROJECT_API FSTEvaluator_SpatialContext : public FStateTreeEvaluatorBase
{
	GENERATED_BODY()

	using FInstanceDataType = FSTEvaluator_SpatialContextInstanceData;

	FSTEvaluator_SpatialContext() = default;

	virtual const UStruct* GetInstanceDataType() const override { return FInstanceDataType::StaticStruct(); }

	virtual void TreeStart(FStateTreeExecutionContext& Context) const override;
	virtual void Tick(FStateTreeExecutionContext& Context, float DeltaTime) const override;
	virtual void TreeStop(FStateTreeExecutionContext& Context) const override;

protected:
	/** Compute action space mask based on environment */
	FActionSpaceMask ComputeActionMask(FStateTreeExecutionContext& Context) const;

	/** Detect if agent is indoors (narrow corridors) */
	bool DetectIndoor(APawn* ControlledPawn, UWorld* World, float CorridorThreshold) const;

	/** Measure lateral clearance (corridor width) */
	float MeasureLateralClearance(APawn* ControlledPawn, UWorld* World) const;

	/** Measure distance to NavMesh edge */
	float MeasureNavMeshEdgeDistance(APawn* ControlledPawn, UWorld* World) const;

	/** Apply cover aiming restrictions (if in cover) */
	void ApplyCoverAimingRestrictions(FActionSpaceMask& Mask, const FFollowerStateTreeContext& SharedContext) const;

	/** Check if sprinting is allowed */
	bool CanSprint(const FFollowerStateTreeContext& SharedContext, bool bIndoor) const;

	/** Check if firing is safe (no friendlies in line of fire) */
	bool IsFiringSafe(APawn* ControlledPawn, UWorld* World, const FFollowerStateTreeContext& SharedContext) const;
};
