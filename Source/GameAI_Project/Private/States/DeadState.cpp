// Fill out your copyright notice in the Description page of Project Settings.

#include "States/DeadState.h"
#include "Core/StateMachine.h"
#include "AIController.h"
#include "BehaviorTree/BehaviorTreeComponent.h"

void UDeadState::EnterState(UStateMachine* StateMachine)
{
    Super::EnterState(StateMachine);

    UE_LOG(LogTemp, Warning, TEXT("Entered DeadState"));

    // Set Blackboard strategy to "Dead" to activate Death behavior tree subtree
    if (StateMachine)
    {
        StateMachine->SetCurrentStrategy(TEXT("Dead"));

        // Stop the Behavior Tree since the agent is dead
        AAIController* AIController = StateMachine->GetAIController();
        if (AIController)
        {
            UBrainComponent* BrainComp = AIController->GetBrainComponent();
            if (BrainComp)
            {
                BrainComp->StopLogic(TEXT("Agent died"));
                UE_LOG(LogTemp, Log, TEXT("DeadState: Stopped Behavior Tree"));
            }
        }

        // Set threat level to 0 (dead agents pose no threat)
        StateMachine->SetThreatLevel(0.0f);

        // Clear any targets
        StateMachine->SetTargetEnemy(nullptr);
    }
}

void UDeadState::UpdateState(UStateMachine* StateMachine, float Reward, float DeltaTime)
{
    // Dead state has no updates - agent is inactive
    // This could be extended to handle respawn logic, death animations, etc.
}

void UDeadState::ExitState(UStateMachine* StateMachine)
{
    UE_LOG(LogTemp, Warning, TEXT("Exited DeadState"));

    // If we're exiting dead state (e.g., respawn), restart the Behavior Tree
    if (StateMachine)
    {
        AAIController* AIController = StateMachine->GetAIController();
        if (AIController)
        {
            UBrainComponent* BrainComp = AIController->GetBrainComponent();
            if (BrainComp && !BrainComp->IsRunning())
            {
                BrainComp->RestartLogic();
                UE_LOG(LogTemp, Log, TEXT("DeadState: Restarted Behavior Tree after respawn"));
            }
        }
    }
}

TArray<UAction*> UDeadState::GetPossibleActions()
{
    // Dead state has no actions
    return TArray<UAction*>();
}
