// Copyright Epic Games, Inc. All Rights Reserved.

#pragma once

#include "CoreMinimal.h"
#include "BehaviorTree/BTDecorator.h"
#include "BTDecorator_CheckStrategy.generated.h"

/**
 * UBTDecorator_CheckStrategy - Behavior Tree Decorator for Strategy Checking
 *
 * This decorator checks if the current strategy in the Blackboard matches
 * the required strategy for a behavior tree subtree to execute.
 *
 * Usage:
 * - Attach this decorator to the root node of a behavior tree subtree (e.g., Attack Behavior, Flee Behavior)
 * - Set the RequiredStrategy property to match the strategy this subtree handles
 * - The decorator will allow the subtree to execute only when Blackboard.CurrentStrategy == RequiredStrategy
 *
 * Example:
 * - Attack subtree has decorator with RequiredStrategy = "Attack"
 * - Flee subtree has decorator with RequiredStrategy = "Flee"
 * - When FSM's AttackState sets Blackboard.CurrentStrategy = "Attack", only the Attack subtree runs
 *
 * This enables the strategic layer (FSM + MCTS) to control which tactical behavior
 * the Behavior Tree executes, creating a clean separation between strategy and tactics.
 */
UCLASS()
class GAMEAI_PROJECT_API UBTDecorator_CheckStrategy : public UBTDecorator
{
	GENERATED_BODY()

public:
	UBTDecorator_CheckStrategy();

protected:
	/**
	 * Called when the decorator is checked for execution.
	 * Returns true if the current strategy matches the required strategy.
	 */
	virtual bool CalculateRawConditionValue(UBehaviorTreeComponent& OwnerComp, uint8* NodeMemory) const override;

	/**
	 * Returns a description of this decorator for the Behavior Tree editor.
	 */
	virtual FString GetStaticDescription() const override;

public:
	/**
	 * The strategy name required for this subtree to execute.
	 * Must match the value set in Blackboard.CurrentStrategy by the FSM states.
	 *
	 * Common values:
	 * - "MoveTo" - Navigation/movement behavior
	 * - "Attack" - Combat/aggressive behavior
	 * - "Flee" - Retreat/evasive behavior
	 * - "Dead" - Death/disabled behavior
	 */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Strategy", meta = (DisplayName = "Required Strategy"))
	FString RequiredStrategy;

	/**
	 * The name of the Blackboard key that stores the current strategy.
	 * Default: "CurrentStrategy"
	 *
	 * This key should be set by FSM states and read by this decorator.
	 */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Strategy", meta = (DisplayName = "Strategy Key Name"))
	FName StrategyKeyName = FName("CurrentStrategy");

	/**
	 * Whether to enable debug logging for this decorator.
	 * Useful for diagnosing strategy switching issues.
	 */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Debug")
	bool bEnableDebugLog = false;
};
