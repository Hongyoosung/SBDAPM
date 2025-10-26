// Fill out your copyright notice in the Description page of Project Settings.

#include "States/MoveToState.h"
#include "Core/StateMachine.h"
#include "Actions/Movement/MoveToAllyAction.h"
#include "Actions/Movement/MoveToAttackPositionAction.h"
#include "Actions/Movement/MoveToCoverAction.h"
#include "Actions/Movement/TacticalRepositionAction.h"
// Legacy directional actions (kept for backward compatibility)
#include "Actions/Movement/MoveForwardAction.h"
#include "Actions/Movement/MoveBackwardAction.h"
#include "Actions/Movement/MoveLeftAction.h"
#include "Actions/Movement/MoveRightAction.h"

void UMoveToState::EnterState(UStateMachine* StateMachine)
{
    UE_LOG(LogTemp, Warning, TEXT("Entered MoveToState"));

    // Set Blackboard strategy to "MoveTo" to activate MoveTo behavior tree subtree
    if (StateMachine)
    {
        StateMachine->SetCurrentStrategy(TEXT("MoveTo"));
    }

    if (MCTS == nullptr)
    {
        MCTS = NewObject<UMCTS>();
        MCTS->InitializeMCTS();
        MCTS->InitializeCurrentNodeLocate();
        PossibleActions = GetPossibleActions();
    }
    else
    {
        MCTS->InitializeCurrentNodeLocate();
	}
}

void UMoveToState::UpdateState(UStateMachine* StateMachine, float Reward, float DeltaTime)
{
	if (!MCTS)
	{
        UE_LOG(LogTemp, Error, TEXT("MCTS or CurrentNode is nullptr"));
		return;
	}

    // Run MCTS to select optimal movement strategy
    // MCTS will evaluate all possible actions (tactical + legacy) and choose the best one
    MCTS->RunMCTS(PossibleActions, StateMachine);

    // Strategic decision-making layer
    // The selected action from MCTS will set the appropriate destination
    // which the Behavior Tree will then execute

    if (StateMachine)
    {
        FObservationElement CurrentObs = StateMachine->GetCurrentObservation();

        // Log strategic context for debugging
        UE_LOG(LogTemp, Display, TEXT("MoveToState Strategic Context:"));
        UE_LOG(LogTemp, Display, TEXT("  - Health: %.1f, Shield: %.1f"), CurrentObs.Health, CurrentObs.Shield);
        UE_LOG(LogTemp, Display, TEXT("  - Enemies: %d, HasCover: %s"),
               CurrentObs.VisibleEnemyCount, CurrentObs.bHasCover ? TEXT("Yes") : TEXT("No"));
        UE_LOG(LogTemp, Display, TEXT("  - Stamina: %.1f"), CurrentObs.Stamina);

        // Provide contextual information to Blackboard for BT decorators
        // The BT will use this to determine which subtree to execute

        // Set movement mode based on context
        FString MovementMode = "Normal";

        // Determine movement mode based on observation
        if (CurrentObs.Health < 40.0f || CurrentObs.VisibleEnemyCount >= 3)
        {
            MovementMode = "Defensive";  // Prioritize cover and safety
        }
        else if (CurrentObs.Health > 80.0f && CurrentObs.Stamina > 70.0f)
        {
            MovementMode = "Aggressive";  // Can take risks for better positioning
        }
        else if (CurrentObs.VisibleEnemyCount > 0 && CurrentObs.bHasCover)
        {
            MovementMode = "Tactical";  // Balance offense and defense
        }

        // This could be set in the Blackboard for BT to read
        // StateMachine->SetBlackboardValue("MovementMode", MovementMode);

        UE_LOG(LogTemp, Display, TEXT("  - Movement Mode: %s"), *MovementMode);
    }
}


void UMoveToState::ExitState(UStateMachine* StateMachine)
{
    if (MCTS)
    {
        MCTS->Backpropagate();
        UE_LOG(LogTemp, Warning, TEXT("Exited MoveToState"));
    }
}

TArray<UAction*> UMoveToState::GetPossibleActions()
{
    TArray<UAction*> Actions;

    // NEW: Tactical movement actions for realistic 3D game behavior
    Actions.Add(NewObject<UMoveToAllyAction>(this, UMoveToAllyAction::StaticClass()));
    Actions.Add(NewObject<UMoveToAttackPositionAction>(this, UMoveToAttackPositionAction::StaticClass()));
    Actions.Add(NewObject<UMoveToCoverAction>(this, UMoveToCoverAction::StaticClass()));
    Actions.Add(NewObject<UTacticalRepositionAction>(this, UTacticalRepositionAction::StaticClass()));

    // LEGACY: Keep simple directional actions for basic movement scenarios
    // These can be removed if you only want tactical actions
    Actions.Add(NewObject<UMoveForwardAction>(this, UMoveForwardAction::StaticClass()));
    Actions.Add(NewObject<UMoveBackwardAction>(this, UMoveBackwardAction::StaticClass()));
    Actions.Add(NewObject<UMoveLeftAction>(this, UMoveLeftAction::StaticClass()));
    Actions.Add(NewObject<UMoveRightAction>(this, UMoveRightAction::StaticClass()));

    return Actions;
}