// Copyright Epic Games, Inc. All Rights Reserved.

#pragma once

#include "CoreMinimal.h"
#include "StateTreeTaskBase.h"
#include "RL/RLTypes.h"
#include "StateTree/FollowerStateTreeSchema.h"
#include "STTask_ExecuteMove.generated.h"


/**
 * State Tree Task: Execute Move
 *
 * Executes movement tactics based on RL policy selection.
 * Handles movement commands including:
 * - Patrol (cyclic waypoint movement)
 * - MoveTo (navigate to target location)
 * - Advance (move toward objective/enemy)
 *
 * Tactical Actions:
 * - Sprint (fast movement, exposed)
 * - Crouch (slow movement, stealthy)
 * - Patrol (standard speed patrol)
 * - CautiousAdvance (cover-to-cover movement)
 * - FlankLeft/FlankRight (flanking movement)
 *
 * This task runs continuously (Ticks) until:
 * - Destination reached
 * - Command changes
 * - Follower dies
 *
 * Requirements:
 * - CurrentCommand.CommandType must be Patrol/MoveTo/Advance
 * - CurrentTacticalAction set by QueryRLPolicy task
 *
 * Rewards provided:
 * - Distance traveled: +2.0 per 100cm
 * - Destination reached: +10.0
 * - Patrol waypoint reached: +5.0
 * - Stealth maintained: +3.0
 */


/**
 * Instance data for ExecuteMove task
 */
USTRUCT()
struct GAMEAI_PROJECT_API FSTTask_ExecuteMoveInstanceData
{
	GENERATED_BODY()

	UPROPERTY(EditAnywhere, Category = "Context")
	FFollowerStateTreeContext Context;

	//--------------------------------------------------------------------------
	// CONFIGURATION
	//--------------------------------------------------------------------------

	/** Query RL policy every N seconds (0 = only at state entry) */
	UPROPERTY(EditAnywhere, Category = "Config", meta = (ClampMin = "0.0", ClampMax = "10.0"))
	float RLQueryInterval = 5.0f;

	/** Sprint speed multiplier */
	UPROPERTY(EditAnywhere, Category = "Config", meta = (ClampMin = "1.0", ClampMax = "3.0"))
	float SprintSpeedMultiplier = 2.0f;

	/** Crouch speed multiplier */
	UPROPERTY(EditAnywhere, Category = "Config", meta = (ClampMin = "0.3", ClampMax = "1.0"))
	float CrouchSpeedMultiplier = 0.6f;

	/** Patrol speed multiplier */
	UPROPERTY(EditAnywhere, Category = "Config", meta = (ClampMin = "0.5", ClampMax = "1.5"))
	float PatrolSpeedMultiplier = 1.0f;

	/** Acceptance radius for waypoints (cm) */
	UPROPERTY(EditAnywhere, Category = "Config", meta = (ClampMin = "50.0", ClampMax = "500.0"))
	float WaypointAcceptanceRadius = 150.0f;

	/** Flank offset distance (cm) */
	UPROPERTY(EditAnywhere, Category = "Config", meta = (ClampMin = "300.0", ClampMax = "1500.0"))
	float FlankOffsetDistance = 700.0f;

	//--------------------------------------------------------------------------
	// RUNTIME STATE (managed by task)
	//--------------------------------------------------------------------------

	/** Time since last RL query */
	UPROPERTY()
	float TimeSinceLastRLQuery = 0.0f;

	/** Total distance traveled */
	UPROPERTY()
	float TotalDistanceTraveled = 0.0f;

	/** Last position (for distance tracking) */
	UPROPERTY()
	FVector LastPosition = FVector::ZeroVector;

	/** Current movement destination */
	UPROPERTY()
	FVector CurrentDestination = FVector::ZeroVector;

	/** Current patrol waypoint index */
	UPROPERTY()
	int32 CurrentWaypointIndex = 0;

	/** Has reached destination */
	UPROPERTY()
	bool bHasReachedDestination = false;
};

USTRUCT(meta = (DisplayName = "Execute Move"))
struct GAMEAI_PROJECT_API FSTTask_ExecuteMove : public FStateTreeTaskBase
{
	GENERATED_BODY()

	using FInstanceDataType = FSTTask_ExecuteMoveInstanceData;

	FSTTask_ExecuteMove() = default;

	virtual const UStruct* GetInstanceDataType() const override { return FInstanceDataType::StaticStruct(); }

	virtual EStateTreeRunStatus EnterState(FStateTreeExecutionContext& Context, const FStateTreeTransitionResult& Transition) const override;
	virtual EStateTreeRunStatus Tick(FStateTreeExecutionContext& Context, float DeltaTime) const override;
	virtual void ExitState(FStateTreeExecutionContext& Context, const FStateTreeTransitionResult& Transition) const override;

protected:
	/** Execute the current tactical action */
	void ExecuteTacticalAction(FStateTreeExecutionContext& Context, float DeltaTime) const;

	/** Execute sprint movement */
	void ExecuteSprint(FStateTreeExecutionContext& Context, float DeltaTime) const;

	/** Execute crouch movement */
	void ExecuteCrouch(FStateTreeExecutionContext& Context, float DeltaTime) const;

	/** Execute patrol movement */
	void ExecutePatrol(FStateTreeExecutionContext& Context, float DeltaTime) const;

	/** Execute cautious advance (cover-to-cover) */
	void ExecuteCautiousAdvance(FStateTreeExecutionContext& Context, float DeltaTime) const;

	/** Execute flank maneuver */
	void ExecuteFlankManeuver(FStateTreeExecutionContext& Context, float DeltaTime, bool bFlankLeft) const;

	/** Get destination based on command type */
	FVector GetDestinationForCommand(FStateTreeExecutionContext& Context) const;

	/** Get next patrol waypoint */
	FVector GetNextPatrolWaypoint(FStateTreeExecutionContext& Context) const;

	/** Calculate movement reward */
	float CalculateMovementReward(FStateTreeExecutionContext& Context, float DeltaTime) const;

	/** Check if should complete movement */
	bool ShouldCompleteMovement(FStateTreeExecutionContext& Context) const;

	/** Move to location with speed modifier */
	void MoveToLocation(FStateTreeExecutionContext& Context, const FVector& Destination, float SpeedMultiplier) const;
};
