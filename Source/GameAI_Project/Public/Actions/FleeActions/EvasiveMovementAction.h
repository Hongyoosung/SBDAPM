// Fill out your copyright notice in the Description page of Project Settings.

#pragma once

#include "CoreMinimal.h"
#include "Actions/Action.h"
#include "EvasiveMovementAction.generated.h"

/**
 * Strategic flee action: Evasive zigzag movement
 *
 * This is a strategic-level action evaluated by MCTS to decide the flee approach.
 * Used when no cover is available or when evasive movement is tactically superior.
 * The actual tactical execution (zigzag pattern) is handled by the Behavior Tree.
 *
 * MCTS evaluates this action based on:
 * - Cover availability (prefer when no cover)
 * - Open terrain presence
 * - Enemy firing patterns
 * - Stamina availability
 */
UCLASS()
class GAMEAI_PROJECT_API UEvasiveMovementAction : public UAction
{
	GENERATED_BODY()

public:
	UEvasiveMovementAction();

	/**
	 * Execute the strategic action
	 * Note: This doesn't perform actual movement - it signals MCTS selection
	 * The Behavior Tree's EvasiveMovement task handles the tactical execution
	 */
	virtual void ExecuteAction(UStateMachine* StateMachine) override;

	/**
	 * Get human-readable action name for debugging
	 */
	FString GetActionName() const { return TEXT("Evasive Movement"); }
};
