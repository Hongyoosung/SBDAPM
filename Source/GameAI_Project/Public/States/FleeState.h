// Fill out your copyright notice in the Description page of Project Settings.

#pragma once

#include "CoreMinimal.h"
#include "States/State.h"
#include "AI/MCTS.h"
#include "FleeState.generated.h"

/**
 * UFleeState - Strategic retreat state with MCTS-driven decision making
 *
 * This state is activated when the agent needs to flee from danger (low health, overwhelming enemies).
 * It uses MCTS to evaluate different flee strategies and activates the Flee behavior tree subtree
 * for tactical execution.
 *
 * Strategic Decisions (MCTS):
 * - Which flee strategy? (Sprint to cover, Evasive movement, Fight while retreating)
 * - Which cover location to target?
 * - When to stop fleeing and transition to another state?
 *
 * Tactical Execution (Behavior Tree):
 * - Pathfinding to cover
 * - Evasive zigzag movement
 * - Sprinting mechanics
 * - Combat while retreating
 *
 * Integration:
 * - Sets Blackboard: CurrentStrategy = "Flee"
 * - Updates Blackboard: CoverLocation, ThreatLevel
 * - BT Decorator checks strategy to activate Flee subtree
 */
UCLASS(Blueprintable)
class GAMEAI_PROJECT_API UFleeState : public UState
{
	GENERATED_BODY()

public:
	virtual void EnterState(UStateMachine* StateMachine) override;
	virtual void UpdateState(UStateMachine* StateMachine, float Reward, float DeltaTime) override;
	virtual void ExitState(UStateMachine* StateMachine) override;
	virtual TArray<UAction*> GetPossibleActions() override;

private:
	/**
	 * MCTS instance for strategic flee decision-making
	 * Evaluates flee actions and selects optimal strategy
	 */
	UPROPERTY(EditAnywhere, Category = "Flee State")
	UMCTS* MCTS;

	/**
	 * Best child node selected by MCTS
	 * Represents the optimal flee action chosen
	 */
	UPROPERTY(EditAnywhere, Category = "Flee State")
	UMCTSNode* BestChild;

	/**
	 * Array of possible flee actions for MCTS to evaluate
	 * Populated in EnterState, includes:
	 * - SprintToCoverAction
	 * - EvasiveMovementAction
	 * - FightWhileRetreatingAction
	 */
	UPROPERTY(EditAnywhere, Category = "Flee State")
	TArray<UAction*> PossibleActions;
};
