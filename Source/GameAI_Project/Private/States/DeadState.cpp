// Fill out your copyright notice in the Description page of Project Settings.


#include "States/DeadState.h"
#include "Core/StateMachine.h"

void UDeadState::EnterState(UStateMachine* StateMachine)
{
    Super::EnterState(StateMachine);
}

void UDeadState::UpdateState(UStateMachine* StateMachine, float Reward, float DeltaTime)
{

}

void UDeadState::ExitState(UStateMachine* StateMachine)
{

}

TArray<UAction*> UDeadState::GetPossibleActions()
{
    return TArray<UAction*>();
}
