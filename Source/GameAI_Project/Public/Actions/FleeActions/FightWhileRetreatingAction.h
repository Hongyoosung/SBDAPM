// Fill out your copyright notice in the Description page of Project Settings.

#pragma once

#include "CoreMinimal.h"
#include "Actions/Action.h"
#include "FightWhileRetreatingAction.generated.h"

/**
 * Strategic flee action: Fight while retreating
 *
 * This is a strategic-level action evaluated by MCTS to decide the flee approach.
 * Used when the agent should maintain offensive pressure while withdrawing.
 * The actual tactical execution (fire-move-fire pattern) is handled by the Behavior Tree.
 *
 * MCTS evaluates this action based on:
 * - Ammunition availability
 * - Enemy pursuit behavior
 * - Cover availability along retreat path
 * - Health threshold (sufficient to risk engagement)
 */
UCLASS()
class GAMEAI_PROJECT_API UFightWhileRetreatingAction : public UAction
{
	GENERATED_BODY()

public:
	UFightWhileRetreatingAction();

	/**
	 * Execute the strategic action
	 * Note: This doesn't perform actual combat - it signals MCTS selection
	 * The Behavior Tree coordinates movement and attack subtrees for execution
	 */
	virtual void ExecuteAction(UStateMachine* StateMachine) override;

	/**
	 * Get human-readable action name for debugging
	 */
	FString GetActionName() const { return TEXT("Fight While Retreating"); }
};
