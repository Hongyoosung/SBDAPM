// Fill out your copyright notice in the Description page of Project Settings.

#pragma once

#include "CoreMinimal.h"
#include "Actions/Action.h"
#include "SprintToCoverAction.generated.h"

/**
 * Strategic flee action: Sprint to nearest cover
 *
 * This is a strategic-level action evaluated by MCTS to decide the flee approach.
 * The actual tactical execution (pathfinding, sprinting) is handled by the Behavior Tree.
 *
 * MCTS evaluates this action based on:
 * - Cover availability (from observation)
 * - Distance to cover
 * - Enemy proximity
 */
UCLASS()
class GAMEAI_PROJECT_API USprintToCoverAction : public UAction
{
	GENERATED_BODY()

public:
	USprintToCoverAction();

	/**
	 * Execute the strategic action
	 * Note: This doesn't perform actual movement - it signals MCTS selection
	 * The Behavior Tree's Flee subtree handles the tactical execution
	 */
	virtual void ExecuteAction(UStateMachine* StateMachine) override;

	/**
	 * Get human-readable action name for debugging
	 */
	FString GetActionName() const { return TEXT("Sprint to Cover"); }
};
