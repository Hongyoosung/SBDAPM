// Fill out your copyright notice in the Description page of Project Settings.

#include "States/MoveToState.h"
#include "Core/StateMachine.h"
#include "Actions/MoveForwardAction.h"
#include "Actions/MoveBackwardAction.h"
#include "Actions/MoveLeftAction.h"
#include "Actions/MoveRightAction.h"

void UMoveToState::EnterState(UStateMachine* StateMachine)
{
    UE_LOG(LogTemp, Warning, TEXT("Entered MoveToState"));

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

    MCTS->RunMCTS(PossibleActions, StateMachine);
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

    Actions.Add(NewObject<UMoveForwardAction>(this, UMoveForwardAction::StaticClass()));
    Actions.Add(NewObject<UMoveBackwardAction>(this, UMoveBackwardAction::StaticClass()));
    Actions.Add(NewObject<UMoveLeftAction>(this, UMoveLeftAction::StaticClass()));
    Actions.Add(NewObject<UMoveRightAction>(this, UMoveRightAction::StaticClass()));

    return Actions;
}