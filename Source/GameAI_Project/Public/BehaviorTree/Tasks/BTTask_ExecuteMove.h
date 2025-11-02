// Copyright Epic Games, Inc. All Rights Reserved.

#pragma once

#include "CoreMinimal.h"
#include "BehaviorTree/BTTaskNode.h"
#include "RL/RLTypes.h"
#include "Team/TeamTypes.h"
#include "BTTask_ExecuteMove.generated.h"

/**
 * Behavior Tree Task: Execute Move
 *
 * Executes movement tactics based on RL policy selection.
 * This task handles tactical movement including:
 * - Sprint (fast movement, exposed)
 * - Crouch (slow, stealthy movement)
 * - Patrol (methodical area coverage)
 * - Hold (stay at destination)
 *
 * Requirements:
 * - Actor must have UFollowerAgentComponent
 * - Current command must be movement-type (Advance, Retreat, Patrol, MoveTo, Follow)
 * - FollowerAgentComponent must have TacticalPolicy initialized
 *
 * Execution Flow:
 * 1. Query RL policy for tactical action
 * 2. Execute selected movement tactic
 * 3. Monitor progress and obstacles
 * 4. Provide reward feedback based on efficiency and safety
 *
 * Blackboard Keys:
 * - Input: "CurrentCommand" (FStrategicCommand) - Strategic command from leader
 * - Input: "MoveDestination" (Vector) - Destination to move to
 * - Input: "PatrolPoints" (Array) - Patrol points for patrol command
 * - Output: "TacticalAction" (Enum) - Selected tactical action
 * - Output: "ActionProgress" (Float) - Progress of current action (0-1)
 *
 * Usage:
 * 1. Add to Behavior Tree under [CommandType == Move] decorator
 * 2. Configure Blackboard keys
 * 3. Set movement parameters (speed multipliers, acceptance radius, etc.)
 */
UCLASS()
class GAMEAI_PROJECT_API UBTTask_ExecuteMove : public UBTTaskNode
{
	GENERATED_BODY()

public:
	UBTTask_ExecuteMove();

	virtual EBTNodeResult::Type ExecuteTask(UBehaviorTreeComponent& OwnerComp, uint8* NodeMemory) override;
	virtual EBTNodeResult::Type AbortTask(UBehaviorTreeComponent& OwnerComp, uint8* NodeMemory) override;
	virtual void TickTask(UBehaviorTreeComponent& OwnerComp, uint8* NodeMemory, float DeltaSeconds) override;
	virtual FString GetStaticDescription() const override;

	//--------------------------------------------------------------------------
	// CONFIGURATION
	//--------------------------------------------------------------------------

	/** Query RL policy every N seconds (0 = only once at start) */
	UPROPERTY(EditAnywhere, Category = "Move|RL", meta = (ClampMin = "0.0", ClampMax = "10.0"))
	float RLQueryInterval = 3.0f;

	/** Distance tolerance to consider destination reached (cm) */
	UPROPERTY(EditAnywhere, Category = "Move|Tactical", meta = (ClampMin = "50.0", ClampMax = "500.0"))
	float AcceptanceRadius = 100.0f;

	/** Sprint speed multiplier */
	UPROPERTY(EditAnywhere, Category = "Move|Tactical", meta = (ClampMin = "1.0", ClampMax = "3.0"))
	float SprintSpeedMultiplier = 1.8f;

	/** Crouch speed multiplier */
	UPROPERTY(EditAnywhere, Category = "Move|Tactical", meta = (ClampMin = "0.3", ClampMax = "1.0"))
	float CrouchSpeedMultiplier = 0.5f;

	/** Patrol speed multiplier */
	UPROPERTY(EditAnywhere, Category = "Move|Tactical", meta = (ClampMin = "0.5", ClampMax = "1.5"))
	float PatrolSpeedMultiplier = 0.7f;

	/** Pause duration at patrol points (seconds) */
	UPROPERTY(EditAnywhere, Category = "Move|Tactical", meta = (ClampMin = "0.0", ClampMax = "10.0"))
	float PatrolPauseDuration = 2.0f;

	/** Scan for enemies while moving */
	UPROPERTY(EditAnywhere, Category = "Move|Tactical")
	bool bScanForEnemies = true;

	/** Enemy detection range while moving (cm) */
	UPROPERTY(EditAnywhere, Category = "Move|Tactical", meta = (ClampMin = "500.0", ClampMax = "5000.0"))
	float EnemyDetectionRange = 2000.0f;

	/** Reward for efficient movement (reaching destination quickly) */
	UPROPERTY(EditAnywhere, Category = "Move|Reward")
	float EfficientMovementReward = 3.0f;

	/** Reward for safe movement (no damage taken) */
	UPROPERTY(EditAnywhere, Category = "Move|Reward")
	float SafeMovementReward = 5.0f;

	/** Reward for detecting enemy during movement */
	UPROPERTY(EditAnywhere, Category = "Move|Reward")
	float EnemyDetectionReward = 2.0f;

	/** Penalty for getting stuck */
	UPROPERTY(EditAnywhere, Category = "Move|Reward")
	float StuckPenalty = -3.0f;

	//--------------------------------------------------------------------------
	// BLACKBOARD KEYS
	//--------------------------------------------------------------------------

	/** Current strategic command from leader */
	UPROPERTY(EditAnywhere, Category = "Blackboard")
	FBlackboardKeySelector CurrentCommandKey;

	/** Destination to move to */
	UPROPERTY(EditAnywhere, Category = "Blackboard")
	FBlackboardKeySelector MoveDestinationKey;

	/** Patrol points array (optional, for patrol command) */
	UPROPERTY(EditAnywhere, Category = "Blackboard")
	FBlackboardKeySelector PatrolPointsKey;

	/** Selected tactical action (output) */
	UPROPERTY(EditAnywhere, Category = "Blackboard")
	FBlackboardKeySelector TacticalActionKey;

	/** Action progress 0-1 (output) */
	UPROPERTY(EditAnywhere, Category = "Blackboard")
	FBlackboardKeySelector ActionProgressKey;

	//--------------------------------------------------------------------------
	// DEBUG
	//--------------------------------------------------------------------------

	/** Log tactical actions to console */
	UPROPERTY(EditAnywhere, Category = "Debug")
	bool bLogActions = true;

	/** Draw debug visualization */
	UPROPERTY(EditAnywhere, Category = "Debug")
	bool bDrawDebugInfo = false;

protected:
	/** Task memory structure */
	struct FBTExecuteMoveMemory
	{
		ETacticalAction CurrentTactic = ETacticalAction::Sprint;
		float TimeInCurrentTactic = 0.0f;
		float TimeSinceLastRLQuery = 0.0f;
		FVector StartLocation = FVector::ZeroVector;
		FVector Destination = FVector::ZeroVector;
		float InitialDistance = 0.0f;
		float LastDistanceToDestination = 0.0f;
		int32 CurrentPatrolIndex = 0;
		float TimeAtPatrolPoint = 0.0f;
		bool bIsPatrolling = false;
		bool bReachedDestination = false;
		bool bEnemyDetected = false;
		int32 DamageTaken = 0;
		FVector LastPosition = FVector::ZeroVector;
		float TimeStuck = 0.0f;
	};

	virtual uint16 GetInstanceMemorySize() const override
	{
		return sizeof(FBTExecuteMoveMemory);
	}

	/** Get follower component from AI controller */
	class UFollowerAgentComponent* GetFollowerComponent(UBehaviorTreeComponent& OwnerComp) const;

	/** Query RL policy for new tactical action */
	ETacticalAction QueryTacticalAction(UBehaviorTreeComponent& OwnerComp) const;

	/** Execute selected tactical action */
	void ExecuteTacticalAction(UBehaviorTreeComponent& OwnerComp, FBTExecuteMoveMemory* Memory, ETacticalAction Action, float DeltaSeconds);

	/** Execute sprint movement */
	void ExecuteSprint(UBehaviorTreeComponent& OwnerComp, FBTExecuteMoveMemory* Memory, float DeltaSeconds);

	/** Execute crouch movement */
	void ExecuteCrouch(UBehaviorTreeComponent& OwnerComp, FBTExecuteMoveMemory* Memory, float DeltaSeconds);

	/** Execute patrol movement */
	void ExecutePatrol(UBehaviorTreeComponent& OwnerComp, FBTExecuteMoveMemory* Memory, float DeltaSeconds);

	/** Execute hold position */
	void ExecuteHold(UBehaviorTreeComponent& OwnerComp, FBTExecuteMoveMemory* Memory, float DeltaSeconds);

	/** Get move destination from blackboard */
	FVector GetMoveDestination(UBehaviorTreeComponent& OwnerComp) const;

	/** Get patrol points from blackboard */
	TArray<FVector> GetPatrolPoints(UBehaviorTreeComponent& OwnerComp) const;

	/** Check if destination is reached */
	bool HasReachedDestination(UBehaviorTreeComponent& OwnerComp, const FVector& Destination) const;

	/** Scan for nearby enemies */
	AActor* ScanForEnemies(UBehaviorTreeComponent& OwnerComp) const;

	/** Check if stuck (not moving for extended time) */
	bool IsStuck(UBehaviorTreeComponent& OwnerComp, FBTExecuteMoveMemory* Memory) const;

	/** Calculate tactical reward based on movement performance */
	float CalculateTacticalReward(UBehaviorTreeComponent& OwnerComp, const FBTExecuteMoveMemory* Memory) const;

	/** Update action progress in blackboard */
	void UpdateActionProgress(UBehaviorTreeComponent& OwnerComp, float Progress) const;

	/** Draw debug visualization */
	void DrawDebugInfo(UBehaviorTreeComponent& OwnerComp, const FBTExecuteMoveMemory* Memory) const;

	/** Check if movement is complete or should abort */
	bool ShouldCompleteMovement(UBehaviorTreeComponent& OwnerComp, const FBTExecuteMoveMemory* Memory) const;

	/** Move to location with specified speed */
	void MoveToLocation(UBehaviorTreeComponent& OwnerComp, const FVector& Destination, float SpeedMultiplier, bool bCrouched, float DeltaSeconds);

	/** Get next patrol point */
	FVector GetNextPatrolPoint(UBehaviorTreeComponent& OwnerComp, FBTExecuteMoveMemory* Memory) const;
};
