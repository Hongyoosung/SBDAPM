// Fill out your copyright notice in the Description page of Project Settings.

#include "Actions/FleeActions/FightWhileRetreatingAction.h"
#include "Core/StateMachine.h"

UFightWhileRetreatingAction::UFightWhileRetreatingAction()
{
	// Initialize action properties if needed
}

void UFightWhileRetreatingAction::ExecuteAction(UStateMachine* StateMachine)
{
	// This is a strategic action - actual execution handled by Behavior Tree
	// Just log that MCTS selected this strategy
	UE_LOG(LogTemp, Log, TEXT("FleeAction: Fight While Retreating selected by MCTS"));

	// The Behavior Tree will coordinate both Attack and Flee subtrees:
	// - Find firing positions along the retreat path
	// - Execute attacks during tactical pauses
	// - Continue movement toward safety

	// This is a more complex strategy that requires BT coordination
	// between offensive and defensive behaviors

	// Optional: Could trigger Blueprint event for additional logic
	// StateMachine->TriggerBlueprintEvent("FightWhileRetreating");
}
