// Fill out your copyright notice in the Description page of Project Settings.


#include "States/State.h"
#include "Core/StateMachine.h"
#include "Actions/Action.h"

void UState::EnterState(UStateMachine* StateMachine)
{
    // ���°� ���۵� �� ������ �۾�
    
}

void UState::ExitState(UStateMachine* StateMachine)
{
    // ���°� ����� �� ������ �۾�
}

void UState::UpdateState(UStateMachine* StateMachine, float Reward, float DeltaTime)
{

}

TArray<UAction*> UState::GetPossibleActions()
{
    return TArray<UAction*>();
}


