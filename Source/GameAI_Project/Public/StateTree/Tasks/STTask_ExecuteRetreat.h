// Copyright Epic Games, Inc. All Rights Reserved.

#pragma once

#include "CoreMinimal.h"
#include "StateTreeTaskBase.h"
#include "RL/RLTypes.h"
#include "StateTree/FollowerStateTreeSchema.h"
#include "STTask_ExecuteRetreat.generated.h"


/**
 * State Tree Task: Execute Retreat
 *
 * Executes retreat tactics based on RL policy selection.
 * Handles tactical withdrawal including:
 * - Fast retreat (sprint away from danger)
 * - Cover retreat (fall back to nearest cover)
 * - Suppressive retreat (retreat while suppressing enemies)
 * - Tactical withdrawal (orderly fallback to rally point)
 *
 * Tactical Actions:
 * - TacticalRetreat (controlled fallback)
 * - Sprint (fast escape)
 * - SeekCover (retreat to cover)
 * - SuppressiveFire (cover retreat with fire)
 *
 * This task runs continuously (Ticks) until:
 * - Safe distance reached
 * - Rally point reached
 * - Command changes
 * - Follower dies
 *
 * Requirements:
 * - CurrentCommand.CommandType must be Retreat
 * - CurrentTacticalAction set by QueryRLPolicy task
 *
 * Rewards provided:
 * - Distance from threats: +4.0
 * - Cover reached: +8.0
 * - Survival during retreat: +5.0
 * - Safe zone reached: +15.0
 */


/**
 * Instance data for ExecuteRetreat task
 */
USTRUCT()
struct GAMEAI_PROJECT_API FSTTask_ExecuteRetreatInstanceData
{
	GENERATED_BODY()

	UPROPERTY(EditAnywhere, Category = "Context")
	FFollowerStateTreeContext Context;

	//--------------------------------------------------------------------------
	// CONFIGURATION
	//--------------------------------------------------------------------------

	/** Minimum safe distance from threats (cm) */
	UPROPERTY(EditAnywhere, Category = "Config", meta = (ClampMin = "1000.0", ClampMax = "5000.0"))
	float MinSafeDistance = 2500.0f;

	/** Sprint speed multiplier during retreat */
	UPROPERTY(EditAnywhere, Category = "Config", meta = (ClampMin = "1.5", ClampMax = "3.0"))
	float RetreatSprintMultiplier = 2.0f;

	/** Cover search radius (cm) */
	UPROPERTY(EditAnywhere, Category = "Config", meta = (ClampMin = "500.0", ClampMax = "5000.0"))
	float CoverSearchRadius = 2000.0f;

	/** Retreat distance per iteration (cm) */
	UPROPERTY(EditAnywhere, Category = "Config", meta = (ClampMin = "300.0", ClampMax = "1500.0"))
	float RetreatStepDistance = 800.0f;

	/** Suppressive fire rate during retreat */
	UPROPERTY(EditAnywhere, Category = "Config", meta = (ClampMin = "0.3", ClampMax = "1.0"))
	float SuppressiveFireRate = 0.6f;

	//--------------------------------------------------------------------------
	// RUNTIME STATE (managed by task)
	//--------------------------------------------------------------------------

	/** Time in retreat */
	UPROPERTY()
	float TimeInRetreat = 0.0f;

	/** Initial retreat position (when retreat started) */
	UPROPERTY()
	FVector InitialRetreatPosition = FVector::ZeroVector;

	/** Total distance retreated */
	UPROPERTY()
	float TotalDistanceRetreated = 0.0f;

	/** Retreat destination */
	UPROPERTY()
	FVector RetreatDestination = FVector::ZeroVector;

	/** Nearest cover location */
	UPROPERTY()
	FVector NearestCoverLocation = FVector::ZeroVector;

	/** Distance to retreat destination (updated each tick) */
	UPROPERTY()
	float DistanceToRetreatDestination = 0.0f;

	/** Distance from primary threat (updated each tick) */
	UPROPERTY()
	float DistanceFromThreat = 0.0f;

	/** Has reached safe distance */
	UPROPERTY()
	bool bHasReachedSafeDistance = false;
};

USTRUCT(meta = (DisplayName = "Execute Retreat"))
struct GAMEAI_PROJECT_API FSTTask_ExecuteRetreat : public FStateTreeTaskBase
{
	GENERATED_BODY()

	using FInstanceDataType = FSTTask_ExecuteRetreatInstanceData;

	FSTTask_ExecuteRetreat() = default;

	virtual const UStruct* GetInstanceDataType() const override { return FInstanceDataType::StaticStruct(); }

	virtual EStateTreeRunStatus EnterState(FStateTreeExecutionContext& Context, const FStateTreeTransitionResult& Transition) const override;
	virtual EStateTreeRunStatus Tick(FStateTreeExecutionContext& Context, float DeltaTime) const override;
	virtual void ExitState(FStateTreeExecutionContext& Context, const FStateTreeTransitionResult& Transition) const override;

protected:
	/** Execute the current tactical action */
	void ExecuteTacticalAction(FStateTreeExecutionContext& Context, float DeltaTime) const;

	/** Execute tactical retreat (controlled fallback) */
	void ExecuteTacticalRetreat(FStateTreeExecutionContext& Context, float DeltaTime) const;

	/** Execute sprint retreat (fast escape) */
	void ExecuteSprintRetreat(FStateTreeExecutionContext& Context, float DeltaTime) const;

	/** Execute cover retreat (retreat to cover) */
	void ExecuteCoverRetreat(FStateTreeExecutionContext& Context, float DeltaTime) const;

	/** Execute suppressive retreat (retreat while firing) */
	void ExecuteSuppressiveRetreat(FStateTreeExecutionContext& Context, float DeltaTime) const;

	/** Calculate retreat destination */
	FVector CalculateRetreatDestination(FStateTreeExecutionContext& Context) const;

	/** Find nearest cover for retreat */
	AActor* FindNearestCover(FStateTreeExecutionContext& Context, const FVector& FromLocation) const;

	/** Calculate retreat reward */
	float CalculateRetreatReward(FStateTreeExecutionContext& Context, float DeltaTime) const;

	/** Check if should complete retreat */
	bool ShouldCompleteRetreat(FStateTreeExecutionContext& Context) const;

	/** Move to retreat destination */
	void MoveToRetreatDestination(FStateTreeExecutionContext& Context, const FVector& Destination, float SpeedMultiplier) const;

	/** Provide suppressive fire while retreating */
	void ProvideSuppressiveFire(FStateTreeExecutionContext& Context) const;
};
