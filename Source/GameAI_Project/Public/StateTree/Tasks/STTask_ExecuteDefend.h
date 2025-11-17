// Copyright Epic Games, Inc. All Rights Reserved.

#pragma once

#include "CoreMinimal.h"
#include "StateTreeTaskBase.h"
#include "RL/RLTypes.h"
#include "StateTree/FollowerStateTreeSchema.h"
#include "STTask_ExecuteDefend.generated.h"


/**
 * State Tree Task: Execute Defend
 *
 * Executes defensive tactics based on RL policy selection.
 * Handles defensive combat maneuvers including:
 * - Defensive hold (maintain position and engage)
 * - Seek cover (find and use cover)
 * - Suppressive fire (suppress enemies from cover)
 * - Tactical retreat (fall back to better position)
 *
 * This task runs continuously (Ticks) until:
 * - Command changes
 * - Follower dies
 * - Tactical objective complete
 *
 * Requirements:
 * - CurrentCommand.CommandType must be Defend-related
 * - CurrentTacticalAction set by QueryRLPolicy task
 *
 * Rewards provided:
 * - Position held: +3.0
 * - Cover usage: +5.0
 * - Survival under fire: +4.0
 * - Position abandoned: -5.0
 */


/**
 * Instance data for ExecuteDefend task
 */
USTRUCT()
struct GAMEAI_PROJECT_API FSTTask_ExecuteDefendInstanceData
{
	GENERATED_BODY()

	//--------------------------------------------------------------------------
	// CONTEXT BINDING (UE 5.6 - auto-binds to FollowerContext from schema)
	//--------------------------------------------------------------------------

	/**
	 * Shared context struct - automatically bound by StateTree
	 * Contains all agent state, commands, observations, and targets
	 */
	UPROPERTY(EditAnywhere, Category = "Context")
	FFollowerStateTreeContext Context;

	//--------------------------------------------------------------------------
	// CONFIGURATION
	//--------------------------------------------------------------------------

	/** Query RL policy every N seconds (0 = only at state entry) */
	UPROPERTY(EditAnywhere, Category = "Config", meta = (ClampMin = "0.0", ClampMax = "10.0"))
	float RLQueryInterval = 3.0f;

	/** Maximum distance from defend location (cm) */
	UPROPERTY(EditAnywhere, Category = "Config", meta = (ClampMin = "100.0", ClampMax = "3000.0"))
	float MaxDefendRadius = 1000.0f;

	/** Minimum safe distance from threats (cm) */
	UPROPERTY(EditAnywhere, Category = "Config", meta = (ClampMin = "500.0", ClampMax = "5000.0"))
	float MinSafeDistance = 1500.0f;

	/** Cover search radius (cm) */
	UPROPERTY(EditAnywhere, Category = "Config", meta = (ClampMin = "500.0", ClampMax = "5000.0"))
	float CoverSearchRadius = 2000.0f;

	/** Fire rate multiplier while defending */
	UPROPERTY(EditAnywhere, Category = "Config", meta = (ClampMin = "0.3", ClampMax = "2.0"))
	float DefensiveFireRateMultiplier = 0.8f;

	/** Accuracy bonus when in cover */
	UPROPERTY(EditAnywhere, Category = "Config", meta = (ClampMin = "1.0", ClampMax = "2.0"))
	float CoverAccuracyBonus = 1.5f;

	//--------------------------------------------------------------------------
	// RUNTIME STATE (managed by task)
	//--------------------------------------------------------------------------

	/** Time since last RL query */
	UPROPERTY()
	float TimeSinceLastRLQuery = 0.0f;

	/** Time in defensive position */
	UPROPERTY()
	float TimeInDefensivePosition = 0.0f;

	/** Defend position (from command or current location) */
	UPROPERTY()
	FVector DefendPosition = FVector::ZeroVector;

	/** Last known threat location */
	UPROPERTY()
	FVector LastKnownThreatLocation = FVector::ZeroVector;
};

USTRUCT(meta = (DisplayName = "Execute Defend"))
struct GAMEAI_PROJECT_API FSTTask_ExecuteDefend : public FStateTreeTaskBase
{
	GENERATED_BODY()

	using FInstanceDataType = FSTTask_ExecuteDefendInstanceData;

	FSTTask_ExecuteDefend() = default;

	virtual const UStruct* GetInstanceDataType() const override { return FInstanceDataType::StaticStruct(); }

	virtual EStateTreeRunStatus EnterState(FStateTreeExecutionContext& Context, const FStateTreeTransitionResult& Transition) const override;
	virtual EStateTreeRunStatus Tick(FStateTreeExecutionContext& Context, float DeltaTime) const override;
	virtual void ExitState(FStateTreeExecutionContext& Context, const FStateTreeTransitionResult& Transition) const override;

protected:
	/** Execute the current tactical action */
	void ExecuteTacticalAction(FStateTreeExecutionContext& Context, float DeltaTime) const;

	/** Execute defensive hold tactic */
	void ExecuteDefensiveHold(FStateTreeExecutionContext& Context, float DeltaTime) const;

	/** Execute seek cover tactic */
	void ExecuteSeekCover(FStateTreeExecutionContext& Context, float DeltaTime) const;

	/** Execute suppressive fire */
	void ExecuteSuppressiveFire(FStateTreeExecutionContext& Context, float DeltaTime) const;

	/** Execute tactical retreat */
	void ExecuteTacticalRetreat(FStateTreeExecutionContext& Context, float DeltaTime) const;

	/** Find nearest cover point */
	AActor* FindNearestCover(FStateTreeExecutionContext& Context, const FVector& FromLocation) const;

	/** Calculate defensive reward */
	float CalculateDefensiveReward(FStateTreeExecutionContext& Context, float DeltaTime) const;

	/** Check if should complete/abort defense */
	bool ShouldCompleteDefense(FStateTreeExecutionContext& Context) const;

	/** Move to position with defensive posture */
	void MoveToDefensivePosition(FStateTreeExecutionContext& Context, const FVector& Destination, float DeltaTime) const;

	/** Engage threats from defensive position */
	void EngageThreats(FStateTreeExecutionContext& Context, float AccuracyModifier) const;
};