// Copyright Epic Games, Inc. All Rights Reserved.

#pragma once

#include "CoreMinimal.h"
#include "BehaviorTree/BTTaskNode.h"
#include "Team/TeamTypes.h"
#include "BTTask_SignalEventToLeader.generated.h"

/**
 * BTTask_SignalEventToLeader
 *
 * **FOLLOWER-ONLY TASK**
 * Team leaders do not signal events to themselves.
 * This task should only be used in BT_FollowerAgent.
 *
 * Signals a strategic event from the follower to the team leader.
 * Used to trigger event-driven MCTS decision-making.
 *
 * Requirements:
 *   - Actor must have UFollowerAgentComponent
 *   - FollowerAgentComponent must have TeamLeader reference
 *   - Use with BT_FollowerAgent behavior tree
 *
 * Usage:
 *   - Add to BT when significant events occur (enemy spotted, under attack, etc.)
 *   - Configure EventType to signal
 *   - Optionally specify target actor (e.g., spotted enemy)
 *   - Task succeeds if leader receives signal, fails if no leader assigned
 *
 * Examples:
 *   - Signal EnemyEncounter when enemy detected
 *   - Signal UnderAttack when taking damage
 *   - Signal ObjectiveReached when arriving at location
 */
UCLASS()
class GAMEAI_PROJECT_API UBTTask_SignalEventToLeader : public UBTTaskNode
{
	GENERATED_BODY()

public:
	UBTTask_SignalEventToLeader();

	virtual EBTNodeResult::Type ExecuteTask(UBehaviorTreeComponent& OwnerComp, uint8* NodeMemory) override;
	virtual FString GetStaticDescription() const override;

	// ========================================
	// Configuration
	// ========================================

	/**
	 * Type of strategic event to signal to the leader
	 */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Event")
	EStrategicEvent EventType = EStrategicEvent::EnemyEncounter;

	/**
	 * Blackboard key containing the target actor (optional)
	 * For example, the enemy that was spotted
	 */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Event")
	FBlackboardKeySelector TargetActorKey;

	/**
	 * If true, task only signals if leader is not already running MCTS
	 * Prevents spamming events during active planning
	 */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Event")
	bool bOnlySignalIfLeaderIdle = false;

	/**
	 * Minimum time between signals of the same event type (seconds)
	 * Prevents duplicate signals in rapid succession
	 */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Event")
	float MinSignalInterval = 2.0f;

	/**
	 * If true, logs event signal to console for debugging
	 */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Debug")
	bool bLogSignals = true;

private:
	// Last time this event type was signaled (per FollowerAgentComponent)
	// Used to enforce MinSignalInterval
	TMap<class UFollowerAgentComponent*, double> LastSignalTimes;
};
