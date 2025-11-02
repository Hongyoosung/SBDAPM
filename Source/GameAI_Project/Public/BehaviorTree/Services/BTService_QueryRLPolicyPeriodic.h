// Copyright Epic Games, Inc. All Rights Reserved.

#pragma once

#include "CoreMinimal.h"
#include "BehaviorTree/BTService.h"
#include "Observation/ObservationElement.h"
#include "BTService_QueryRLPolicyPeriodic.generated.h"

/**
 * BTService_QueryRLPolicyPeriodic
 *
 * Periodically queries the RL policy for tactical action selection.
 * Updates the blackboard with the selected action, allowing BT tasks
 * and decorators to react to RL policy decisions.
 *
 * Workflow:
 *   1. Get local observation from FollowerAgentComponent
 *   2. Query RLPolicyNetwork for tactical action
 *   3. Update blackboard with selected action
 *   4. Update last action time for temporal tracking
 *
 * This service provides continuous tactical decision-making based on
 * the current observation, strategic command, and combat state.
 *
 * Usage:
 *   - Add to high-level composite node (e.g., root selector)
 *   - Configure update interval (default: 1.0s)
 *   - Set TacticalActionKey blackboard key
 *   - Use BTDecorator_CheckTacticalAction to branch based on selected action
 *
 * Note:
 *   For one-shot queries (e.g., "query once then execute"), use
 *   BTTask_QueryRLPolicy instead. This service is for continuous queries.
 */
UCLASS()
class GAMEAI_PROJECT_API UBTService_QueryRLPolicyPeriodic : public UBTService
{
	GENERATED_BODY()

public:
	UBTService_QueryRLPolicyPeriodic();

	virtual void TickNode(UBehaviorTreeComponent& OwnerComp, uint8* NodeMemory, float DeltaSeconds) override;
	virtual FString GetStaticDescription() const override;

	// ========================================
	// Blackboard Keys
	// ========================================

	/**
	 * Blackboard key to store selected tactical action (enum/byte)
	 * Will store ETacticalAction as uint8
	 */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Blackboard")
	FBlackboardKeySelector TacticalActionKey;

	/**
	 * Blackboard key to store action probability (float, optional)
	 * Stores the confidence/probability of the selected action
	 */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Blackboard")
	FBlackboardKeySelector ActionProbabilityKey;

	/**
	 * Blackboard key to store whether RL policy is ready (bool, optional)
	 * True if policy network is initialized and ready for queries
	 */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Blackboard")
	FBlackboardKeySelector IsPolicyReadyKey;

	// ========================================
	// Configuration
	// ========================================

	/**
	 * If true, only queries policy when observation has changed significantly
	 * Reduces unnecessary queries when situation is stable
	 */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Config")
	bool bQueryOnlyWhenObservationChanged = false;

	/**
	 * Minimum similarity threshold for "observation changed" detection
	 * Only used if bQueryOnlyWhenObservationChanged is true
	 * Range: 0.0 (always different) to 1.0 (must be identical)
	 */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Config")
	float ObservationSimilarityThreshold = 0.95f;

	/**
	 * If true, only queries policy when follower has an active command
	 * Prevents unnecessary queries when idle
	 */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Config")
	bool bRequireActiveCommand = false;

	/**
	 * If true, uses exploration (epsilon-greedy) when querying policy
	 */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Config")
	bool bEnableExploration = true;

	/**
	 * If true, logs policy queries to console for debugging
	 */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Debug")
	bool bLogQueries = false;

private:
	// Last observation used for similarity comparison
	// Used when bQueryOnlyWhenObservationChanged is true
	TMap<class UFollowerAgentComponent*, FObservationElement> LastObservations;
};
