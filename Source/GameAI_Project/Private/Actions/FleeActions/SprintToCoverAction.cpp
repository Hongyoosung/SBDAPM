// Fill out your copyright notice in the Description page of Project Settings.

#include "Actions/FleeActions/SprintToCoverAction.h"
#include "Core/StateMachine.h"

USprintToCoverAction::USprintToCoverAction()
{
	// Initialize action properties if needed
}

void USprintToCoverAction::ExecuteAction(UStateMachine* StateMachine)
{
	// This is a strategic action - actual execution handled by Behavior Tree
	// Just log that MCTS selected this strategy
	UE_LOG(LogTemp, Log, TEXT("FleeAction: Sprint to Cover selected by MCTS"));

	// The Behavior Tree's Flee subtree will handle the actual sprinting
	// using the CoverLocation already set on the Blackboard by FleeState

	// Optional: Could trigger Blueprint event for additional logic
	// StateMachine->TriggerBlueprintEvent("SprintToCover");
}
