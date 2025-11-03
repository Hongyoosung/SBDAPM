// Copyright Epic Games, Inc. All Rights Reserved.

#pragma once

#include "CoreMinimal.h"
#include "BehaviorTree/BTTaskNode.h"
#include "RL/RLTypes.h"
#include "BTTask_QueryRLPolicy.generated.h"

/**
 * Behavior Tree Task: Query RL Policy
 *
 * **FOLLOWER-ONLY TASK**
 * Team leaders use MCTS for strategic decisions, not RL policies.
 * This task should only be used in BT_FollowerAgent.
 *
 * Queries the follower's RL policy for the next tactical action
 * and updates the Blackboard with the selected action.
 *
 * Requirements:
 * - Actor must have UFollowerAgentComponent
 * - FollowerAgentComponent must have TacticalPolicy initialized
 * - Use with BT_FollowerAgent behavior tree
 *
 * Blackboard Keys:
 * - Output: "TacticalAction" (Enum) - Selected tactical action
 * - Output: "TacticalActionName" (String) - Human-readable action name
 *
 * Usage:
 * 1. Add this task to BT_FollowerAgent
 * 2. Configure Blackboard keys
 * 3. Task will query RL policy and update Blackboard
 * 4. Subsequent BT tasks can use TacticalAction to execute behavior
 */
UCLASS()
class GAMEAI_PROJECT_API UBTTask_QueryRLPolicy : public UBTTaskNode
{
	GENERATED_BODY()

public:
	UBTTask_QueryRLPolicy();

	virtual EBTNodeResult::Type ExecuteTask(UBehaviorTreeComponent& OwnerComp, uint8* NodeMemory) override;
	virtual FString GetStaticDescription() const override;

	//--------------------------------------------------------------------------
	// CONFIGURATION
	//--------------------------------------------------------------------------

	/** Blackboard key to store the selected tactical action (Enum) */
	UPROPERTY(EditAnywhere, Category = "Blackboard")
	FBlackboardKeySelector TacticalActionKey;

	/** Blackboard key to store the action name (String) */
	UPROPERTY(EditAnywhere, Category = "Blackboard")
	FBlackboardKeySelector TacticalActionNameKey;

	/** Log action selection to console */
	UPROPERTY(EditAnywhere, Category = "Debug")
	bool bLogActionSelection = true;

	/** Draw debug visualization */
	UPROPERTY(EditAnywhere, Category = "Debug")
	bool bDrawDebugInfo = false;

protected:
	/** Get follower component from AI controller */
	class UFollowerAgentComponent* GetFollowerComponent(UBehaviorTreeComponent& OwnerComp) const;

	/** Draw debug visualization */
	void DrawDebugActionSelection(UBehaviorTreeComponent& OwnerComp, ETacticalAction SelectedAction) const;
};
