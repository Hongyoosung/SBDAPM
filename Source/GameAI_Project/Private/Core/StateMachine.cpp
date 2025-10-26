// Fill out your copyright notice in the Description page of Project Settings.


#include "Core/StateMachine.h"
#include "States/State.h"
#include "States/MoveToState.h"
#include "States/AttackState.h"
#include "States/FleeState.h"
#include "States/DeadState.h"
#include "AIController.h"
#include "GameFramework/Character.h"


UStateMachine::UStateMachine()
{
    PrimaryComponentTick.bCanEverTick = true;
    Owner = GetOwner();
}

void UStateMachine::BeginPlay()
{
    Super::BeginPlay();
    InitStateMachine();
}

void UStateMachine::InitStateMachine()
{
    // ���� ��ü ����
    MoveToState     =   NewObject<UMoveToState>(this, UMoveToState::StaticClass());
    AttackState     =   NewObject<UAttackState>(this, UAttackState::StaticClass());
    FleeState       =   NewObject<UFleeState>(this, UFleeState::StaticClass());
    DeadState       =   NewObject<UDeadState>(this, UDeadState::StaticClass());

    // �ʱ� ���� ����
    CurrentState    =   MoveToState;


    // �ʱ� ���� ����
    /*
    if (CurrentState)
    {
        CurrentState->EnterState(this);
    }
    */
}

void UStateMachine::TickComponent(float DeltaTime, ELevelTick TickType, FActorComponentTickFunction* ThisTickFunction)
{
    //Super::TickComponent(DeltaTime, TickType, ThisTickFunction);
    
    /*
    // ���� ���� ������Ʈ
    if (CurrentState)
    {
        
        // Status updates every 2 seconds
        CurrentTime = GetWorld()->GetTimeSeconds();
        if (CurrentTime - LastStateUpdateTime > 2.0f)
        {
            LastStateUpdateTime = CurrentTime;
            CurrentState->UpdateState(this, DeltaTime);
        }
    }
    */
}

void UStateMachine::ChangeState(UState* NewState)
{
    if (CurrentState != NewState)
    {
        if (CurrentState)
        {
            CurrentState->  ExitState(this);
            CurrentState =  NewState;
            CurrentState->  EnterState(this);
        }
    }
}

void UStateMachine::GetObservation(float Health, float Distance, int32 Num)
{
    this->AgentHealth = Health;
	this->DistanceToDestination = Distance;
	this->EnemiesNum = Num;
}

void UStateMachine::TriggerBlueprintEvent(const FName& EventName)
{
    if (Owner)
    {
        UFunction* Function = Owner->FindFunction(EventName);
        if (Function)
        {
            Owner->ProcessEvent(Function, nullptr);
        }
    }
}

UState* UStateMachine::GetCurrentState()
{
    return CurrentState;
}

UState* UStateMachine::GetMoveToState()
{
    return MoveToState;
}

UState* UStateMachine::GetAttackState()
{
    return AttackState;
}

UState* UStateMachine::GetFleeState()
{
    return FleeState;
}

UState* UStateMachine::GetDeadState()
{
    return DeadState;
}


