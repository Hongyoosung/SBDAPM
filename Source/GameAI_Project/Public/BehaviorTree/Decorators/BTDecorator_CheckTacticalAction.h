// Copyright Epic Games, Inc. All Rights Reserved.

#pragma once

#include "CoreMinimal.h"
#include "BehaviorTree/BTDecorator.h"
#include "RL/RLTypes.h"
#include "BTDecorator_CheckTacticalAction.generated.h"

/**
 * BTDecorator_CheckTacticalAction
 *
 * Decorator that checks if the RL policy's selected tactical action matches
 * the required action type(s).
 *
 * Used to branch behavior tree execution based on the action selected by the
 * RL policy (e.g., execute flanking maneuver only when RL selects "FlankLeft").
 *
 * Features:
 *   - Check single tactical action or multiple actions
 *   - Observer aborts support (abort when action changes)
 *   - Invert condition (check if NOT this action)
 *   - Blackboard integration (read action from blackboard)
 *
 * Usage Example:
 *   AssaultSubtree (executed when command is Assault)
 *   ├─ [CheckTacticalAction: AggressiveAssault] AggressiveSubtree
 *   ├─ [CheckTacticalAction: CautiousAdvance] CautiousSubtree
 *   ├─ [CheckTacticalAction: FlankLeft, FlankRight] FlankSubtree
 *   └─ [Default] GenericAssaultSubtree
 *
 * Workflow:
 *   1. BTService_QueryRLPolicyPeriodic queries policy and updates blackboard
 *   2. This decorator reads tactical action from blackboard
 *   3. Checks if action matches accepted types
 *   4. Allows/blocks subtree execution accordingly
 *
 * Configuration:
 *   - Add decorator to composite node
 *   - Select tactical actions to match (can select multiple)
 *   - Set observer aborts to "Self" for reactivity when action changes
 */
UCLASS()
class GAMEAI_PROJECT_API UBTDecorator_CheckTacticalAction : public UBTDecorator
{
	GENERATED_BODY()

public:
	UBTDecorator_CheckTacticalAction();

	virtual bool CalculateRawConditionValue(UBehaviorTreeComponent& OwnerComp, uint8* NodeMemory) const override;
	virtual FString GetStaticDescription() const override;

	// ========================================
	// Configuration
	// ========================================

	/**
	 * Tactical actions to check against
	 * If current action matches ANY of these types, condition is true
	 */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Tactical Action")
	TArray<ETacticalAction> AcceptedActions;

	/**
	 * If true, condition is inverted (true when action does NOT match)
	 */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Tactical Action")
	bool bInvertCondition = false;

	/**
	 * Blackboard key containing tactical action (byte/enum)
	 * Should be populated by BTService_QueryRLPolicyPeriodic or BTTask_QueryRLPolicy
	 */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Tactical Action")
	FBlackboardKeySelector TacticalActionKey;

	/**
	 * If true, reads action directly from FollowerAgentComponent instead of blackboard
	 * Less efficient but ensures latest action is used
	 */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Tactical Action")
	bool bReadDirectlyFromComponent = false;

	/**
	 * If true, logs condition checks to console for debugging
	 */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Debug")
	bool bLogChecks = false;

private:
	/**
	 * Get tactical action from blackboard
	 */
	ETacticalAction GetTacticalActionFromBlackboard(UBehaviorTreeComponent& OwnerComp) const;

	/**
	 * Get tactical action directly from FollowerAgentComponent
	 */
	ETacticalAction GetTacticalActionFromComponent(UBehaviorTreeComponent& OwnerComp) const;
};
