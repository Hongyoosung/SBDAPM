// Fill out your copyright notice in the Description page of Project Settings.

#include "Actions/FleeActions/EvasiveMovementAction.h"
#include "Core/StateMachine.h"

UEvasiveMovementAction::UEvasiveMovementAction()
{
	// Initialize action properties if needed
}

void UEvasiveMovementAction::ExecuteAction(UStateMachine* StateMachine)
{
	// This is a strategic action - actual execution handled by Behavior Tree
	// Just log that MCTS selected this strategy
	UE_LOG(LogTemp, Log, TEXT("FleeAction: Evasive Movement selected by MCTS"));

	// The Behavior Tree's EvasiveMovement task will handle the actual zigzag pattern
	// This action indicates that the agent should prioritize unpredictable movement
	// over rushing to cover

	// Optional: Could trigger Blueprint event for additional logic
	// StateMachine->TriggerBlueprintEvent("EvasiveMovement");
}
