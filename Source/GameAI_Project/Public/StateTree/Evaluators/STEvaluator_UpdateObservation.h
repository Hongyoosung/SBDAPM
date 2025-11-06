// Copyright Epic Games, Inc. All Rights Reserved.

#pragma once

#include "CoreMinimal.h"
#include "StateTreeEvaluatorBase.h"
#include "Observation/ObservationElement.h"
#include "StateTree/FollowerStateTreeSchema.h"
#include "STEvaluator_UpdateObservation.generated.h"


/**
 * State Tree Evaluator: Update Observation
 *
 * Continuously gathers environmental data and updates the context observation.
 * Runs every tick while State Tree is active (like a BT service).
 *
 * Responsibilities:
 * 1. Gather agent state (position, velocity, health, etc.)
 * 2. Perform 16-ray raycasts for 360° environment perception
 * 3. Scan for nearby enemies (top 5 closest)
 * 4. Detect cover availability and positions
 * 5. Update weapon/combat state
 * 6. Update context.CurrentObservation
 *
 * This evaluator is CRITICAL - it provides the observation data needed
 * by all tasks and the RL policy.
 */


/**
 * Instance data for UpdateObservation evaluator
 */
USTRUCT()
struct GAMEAI_PROJECT_API FSTEvaluator_UpdateObservationInstanceData
{
	GENERATED_BODY()

	//--------------------------------------------------------------------------
	// INPUTS (bound from context)
	//--------------------------------------------------------------------------

	/** Follower component */
	UPROPERTY(EditAnywhere, Category = "Input")
	TObjectPtr<UFollowerAgentComponent> FollowerComponent = nullptr;

	/** AI Controller */
	UPROPERTY(EditAnywhere, Category = "Input")
	TObjectPtr<AAIController> AIController = nullptr;

	//--------------------------------------------------------------------------
	// OUTPUTS (bound to context)
	//--------------------------------------------------------------------------

	/** Current observation */
	UPROPERTY(EditAnywhere, Category = "Output")
	FObservationElement CurrentObservation;

	/** Previous observation */
	UPROPERTY(EditAnywhere, Category = "Output")
	FObservationElement PreviousObservation;

	/** Visible enemies */
	UPROPERTY(EditAnywhere, Category = "Output")
	TArray<TObjectPtr<AActor>> VisibleEnemies;

	/** Primary target */
	UPROPERTY(EditAnywhere, Category = "Output")
	TObjectPtr<AActor> PrimaryTarget = nullptr;

	/** In cover */
	UPROPERTY(EditAnywhere, Category = "Output")
	bool bInCover = false;

	/** Under fire */
	UPROPERTY(EditAnywhere, Category = "Output")
	bool bUnderFire = false;

	/** Has LOS to target */
	UPROPERTY(EditAnywhere, Category = "Output")
	bool bHasLOS = false;

	//--------------------------------------------------------------------------
	// CONFIGURATION
	//--------------------------------------------------------------------------

	/** Update interval (seconds) - 0.1 = 10Hz */
	UPROPERTY(EditAnywhere, Category = "Config", meta = (ClampMin = "0.01", ClampMax = "1.0"))
	float UpdateInterval = 0.1f;

	/** Draw debug visualizations */
	UPROPERTY(EditAnywhere, Category = "Debug")
	bool bDrawDebugInfo = false;

	//--------------------------------------------------------------------------
	// RUNTIME STATE
	//--------------------------------------------------------------------------

	/** Time accumulator for update interval */
	UPROPERTY()
	float TimeAccumulator = 0.0f;
};

USTRUCT(meta = (DisplayName = "Update Observation", BlueprintType))
struct GAMEAI_PROJECT_API FSTEvaluator_UpdateObservation : public FStateTreeEvaluatorBase
{
	GENERATED_BODY()

	using FInstanceDataType = FSTEvaluator_UpdateObservationInstanceData;

	FSTEvaluator_UpdateObservation() = default;

	virtual const UStruct* GetInstanceDataType() const override { return FInstanceDataType::StaticStruct(); }

	virtual void TreeStart(FStateTreeExecutionContext& Context) const override;
	virtual void Tick(FStateTreeExecutionContext& Context, float DeltaTime) const override;
	virtual void TreeStop(FStateTreeExecutionContext& Context) const override;

protected:
	/** Gather all observation data */
	FObservationElement GatherObservationData(FStateTreeExecutionContext& Context) const;

	/** Update agent state information */
	void UpdateAgentState(FObservationElement& Observation, APawn* ControlledPawn) const;

	/** Perform raycast-based environment perception */
	void PerformRaycastPerception(FObservationElement& Observation, APawn* ControlledPawn, UWorld* World) const;

	/** Scan for nearby enemies */
	void ScanForEnemies(FObservationElement& Observation, APawn* ControlledPawn, UWorld* World) const;

	/** Detect cover availability */
	void DetectCover(FObservationElement& Observation, APawn* ControlledPawn, UWorld* World) const;

	/** Update combat state */
	void UpdateCombatState(FObservationElement& Observation, APawn* ControlledPawn) const;

	/** Classify raycast hit type */
	ERaycastHitType ClassifyHitType(const FHitResult& HitResult) const;

	/** Detect terrain type */
	ETerrainType DetectTerrainType(APawn* ControlledPawn) const;
};