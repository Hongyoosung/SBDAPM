// Copyright Epic Games, Inc. All Rights Reserved.

#pragma once

#include "CoreMinimal.h"
#include "BehaviorTree/BTTaskNode.h"
#include "RL/RLTypes.h"
#include "Team/TeamTypes.h"
#include "BTTask_ExecuteDefend.generated.h"

/**
 * Behavior Tree Task: Execute Defend
 *
 * Executes defensive tactics based on RL policy selection.
 * This task handles defensive combat maneuvers including:
 * - Defensive hold (maintain position and engage)
 * - Seek cover (find and use cover)
 * - Suppressive fire (suppress enemies from cover)
 * - Tactical retreat (fall back to better position)
 *
 * Requirements:
 * - Actor must have UFollowerAgentComponent
 * - Current command must be defend-type (StayAlert, HoldPosition, TakeCover, Fortify)
 * - FollowerAgentComponent must have TacticalPolicy initialized
 *
 * Execution Flow:
 * 1. Query RL policy for tactical action
 * 2. Execute selected defensive tactic
 * 3. Monitor threats and position
 * 4. Provide reward feedback based on survival and effectiveness
 *
 * Blackboard Keys:
 * - Input: "CurrentCommand" (FStrategicCommand) - Strategic command from leader
 * - Input: "DefendLocation" (Vector) - Location to defend
 * - Input: "ThreatActors" (Array) - Known enemy threats
 * - Output: "TacticalAction" (Enum) - Selected tactical action
 * - Output: "CoverActor" (AActor) - Current cover being used
 * - Output: "ActionProgress" (Float) - Progress of current action (0-1)
 *
 * Usage:
 * 1. Add to Behavior Tree under [CommandType == Defend] decorator
 * 2. Configure Blackboard keys
 * 3. Set defensive parameters (cover search radius, engagement rules, etc.)
 */
UCLASS()
class GAMEAI_PROJECT_API UBTTask_ExecuteDefend : public UBTTaskNode
{
	GENERATED_BODY()

public:
	UBTTask_ExecuteDefend();

	virtual EBTNodeResult::Type ExecuteTask(UBehaviorTreeComponent& OwnerComp, uint8* NodeMemory) override;
	virtual EBTNodeResult::Type AbortTask(UBehaviorTreeComponent& OwnerComp, uint8* NodeMemory) override;
	virtual void TickTask(UBehaviorTreeComponent& OwnerComp, uint8* NodeMemory, float DeltaSeconds) override;
	virtual FString GetStaticDescription() const override;

	//--------------------------------------------------------------------------
	// CONFIGURATION
	//--------------------------------------------------------------------------

	/** Query RL policy every N seconds (0 = only once at start) */
	UPROPERTY(EditAnywhere, Category = "Defend|RL", meta = (ClampMin = "0.0", ClampMax = "10.0"))
	float RLQueryInterval = 3.0f;

	/** Maximum distance from defend location (cm) */
	UPROPERTY(EditAnywhere, Category = "Defend|Tactical", meta = (ClampMin = "100.0", ClampMax = "3000.0"))
	float MaxDefendRadius = 1000.0f;

	/** Minimum safe distance from threats (cm) */
	UPROPERTY(EditAnywhere, Category = "Defend|Tactical", meta = (ClampMin = "500.0", ClampMax = "5000.0"))
	float MinSafeDistance = 1500.0f;

	/** Cover search radius (cm) */
	UPROPERTY(EditAnywhere, Category = "Defend|Tactical", meta = (ClampMin = "500.0", ClampMax = "5000.0"))
	float CoverSearchRadius = 2000.0f;

	/** Fire rate multiplier while defending */
	UPROPERTY(EditAnywhere, Category = "Defend|Combat", meta = (ClampMin = "0.3", ClampMax = "2.0"))
	float DefensiveFireRateMultiplier = 0.8f;

	/** Accuracy bonus when in cover */
	UPROPERTY(EditAnywhere, Category = "Defend|Combat", meta = (ClampMin = "1.0", ClampMax = "2.0"))
	float CoverAccuracyBonus = 1.5f;

	/** Time to stay in defensive hold before re-evaluating (seconds) */
	UPROPERTY(EditAnywhere, Category = "Defend|Tactical", meta = (ClampMin = "1.0", ClampMax = "30.0"))
	float DefensiveHoldDuration = 10.0f;

	/** Reward for maintaining defensive position */
	UPROPERTY(EditAnywhere, Category = "Defend|Reward")
	float PositionHeldReward = 3.0f;

	/** Reward for successfully using cover */
	UPROPERTY(EditAnywhere, Category = "Defend|Reward")
	float CoverUsageReward = 5.0f;

	/** Reward for surviving while under fire */
	UPROPERTY(EditAnywhere, Category = "Defend|Reward")
	float SurvivalReward = 4.0f;

	/** Penalty for abandoning defensive position */
	UPROPERTY(EditAnywhere, Category = "Defend|Reward")
	float PositionAbandonedPenalty = -5.0f;

	//--------------------------------------------------------------------------
	// BLACKBOARD KEYS
	//--------------------------------------------------------------------------

	/** Current strategic command from leader */
	UPROPERTY(EditAnywhere, Category = "Blackboard")
	FBlackboardKeySelector CurrentCommandKey;

	/** Location to defend */
	UPROPERTY(EditAnywhere, Category = "Blackboard")
	FBlackboardKeySelector DefendLocationKey;

	/** Array of threat actors */
	UPROPERTY(EditAnywhere, Category = "Blackboard")
	FBlackboardKeySelector ThreatActorsKey;

	/** Selected tactical action (output) */
	UPROPERTY(EditAnywhere, Category = "Blackboard")
	FBlackboardKeySelector TacticalActionKey;

	/** Current cover actor (output) */
	UPROPERTY(EditAnywhere, Category = "Blackboard")
	FBlackboardKeySelector CoverActorKey;

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
	struct FBTExecuteDefendMemory
	{
		ETacticalAction CurrentTactic = ETacticalAction::DefensiveHold;
		float TimeInCurrentTactic = 0.0f;
		float TimeSinceLastRLQuery = 0.0f;
		float TimeInDefensivePosition = 0.0f;
		int32 ShotsBlockedByCover = 0;
		int32 DamageTaken = 0;
		bool bHasCover = false;
		AActor* CurrentCover = nullptr;
		FVector DefendPosition = FVector::ZeroVector;
		FVector LastKnownThreatLocation = FVector::ZeroVector;
		int32 VisibleThreats = 0;
	};

	virtual uint16 GetInstanceMemorySize() const override
	{
		return sizeof(FBTExecuteDefendMemory);
	}

	/** Get follower component from AI controller */
	class UFollowerAgentComponent* GetFollowerComponent(UBehaviorTreeComponent& OwnerComp) const;

	/** Query RL policy for new tactical action */
	ETacticalAction QueryTacticalAction(UBehaviorTreeComponent& OwnerComp) const;

	/** Execute selected tactical action */
	void ExecuteTacticalAction(UBehaviorTreeComponent& OwnerComp, FBTExecuteDefendMemory* Memory, ETacticalAction Action, float DeltaSeconds);

	/** Execute defensive hold tactic */
	void ExecuteDefensiveHold(UBehaviorTreeComponent& OwnerComp, FBTExecuteDefendMemory* Memory, float DeltaSeconds);

	/** Execute seek cover tactic */
	void ExecuteSeekCover(UBehaviorTreeComponent& OwnerComp, FBTExecuteDefendMemory* Memory, float DeltaSeconds);

	/** Execute suppressive fire from defensive position */
	void ExecuteSuppressiveFire(UBehaviorTreeComponent& OwnerComp, FBTExecuteDefendMemory* Memory, float DeltaSeconds);

	/** Execute tactical retreat */
	void ExecuteTacticalRetreat(UBehaviorTreeComponent& OwnerComp, FBTExecuteDefendMemory* Memory, float DeltaSeconds);

	/** Get defend location from blackboard or command */
	FVector GetDefendLocation(UBehaviorTreeComponent& OwnerComp) const;

	/** Find nearest cover point */
	AActor* FindNearestCover(UBehaviorTreeComponent& OwnerComp, const FVector& FromLocation) const;

	/** Check if position is within defensive radius */
	bool IsWithinDefensiveRadius(UBehaviorTreeComponent& OwnerComp, const FVector& Position) const;

	/** Get threat actors from blackboard */
	TArray<AActor*> GetThreatActors(UBehaviorTreeComponent& OwnerComp) const;

	/** Get nearest threat */
	AActor* GetNearestThreat(UBehaviorTreeComponent& OwnerComp) const;

	/** Calculate tactical reward based on defensive performance */
	float CalculateTacticalReward(UBehaviorTreeComponent& OwnerComp, const FBTExecuteDefendMemory* Memory) const;

	/** Update action progress in blackboard */
	void UpdateActionProgress(UBehaviorTreeComponent& OwnerComp, float Progress) const;

	/** Draw debug visualization */
	void DrawDebugInfo(UBehaviorTreeComponent& OwnerComp, const FBTExecuteDefendMemory* Memory) const;

	/** Check if defense is complete or should abort */
	bool ShouldCompleteDefense(UBehaviorTreeComponent& OwnerComp, const FBTExecuteDefendMemory* Memory) const;

	/** Move to position with defensive posture */
	void MoveToDefensivePosition(UBehaviorTreeComponent& OwnerComp, const FVector& Destination, bool bUseCover, float DeltaSeconds);

	/** Engage threats from defensive position */
	void EngageThreats(UBehaviorTreeComponent& OwnerComp, float AccuracyModifier, float FireRateMultiplier);

	/** Update cover status */
	void UpdateCoverStatus(UBehaviorTreeComponent& OwnerComp, FBTExecuteDefendMemory* Memory);
};
