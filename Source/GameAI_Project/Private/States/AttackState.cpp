// Fill out your copyright notice in the Description page of Project Settings.


#include "States/AttackState.h"
#include "Core/StateMachine.h"
#include "Actions/SkillAttackAction.h"
#include "Actions/DeafultAttackAction.h"

void UAttackState::EnterState(UStateMachine* StateMachine)
{
	UE_LOG(LogTemp, Warning, TEXT("Entered Attack State"));

	// Set Blackboard strategy to "Attack" to activate Attack behavior tree subtree
	if (StateMachine)
	{
		StateMachine->SetCurrentStrategy(TEXT("Attack"));
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

void UAttackState::UpdateState(UStateMachine* StateMachine, float Reward, float DeltaTime)
{
	if (MCTS == nullptr)
	{
		UE_LOG(LogTemp, Error, TEXT("MCTS or CurrentNode is nullptr"));
		return;
	}

	// Run MCTS to select optimal combat strategy and target
	MCTS->RunMCTS(PossibleActions, StateMachine);

	// Update Blackboard with target enemy
	// TODO: This should use MCTS output to select the best target
	// For now, we'll select the nearest enemy from observations
	if (StateMachine)
	{
		FObservationElement CurrentObs = StateMachine->GetCurrentObservation();

		// Check if we have visible enemies
		if (CurrentObs.VisibleEnemyCount > 0 && CurrentObs.NearbyEnemies.Num() > 0)
		{
			// Select the nearest enemy (first in the sorted array)
			// In a more sophisticated implementation, MCTS would help decide which enemy to prioritize
			// based on factors like: distance, health, threat level, tactical position, etc.

			// Note: We don't have the actual enemy actor reference in the observation yet
			// This is a limitation that should be addressed by storing enemy actor references
			// For now, we just update the threat level based on enemy presence
			float ThreatLevel = FMath::Clamp(CurrentObs.VisibleEnemyCount / 10.0f, 0.0f, 1.0f);
			StateMachine->SetThreatLevel(ThreatLevel);

			// TODO: Properly store and retrieve enemy actor references
			// StateMachine->SetTargetEnemy(SelectedEnemy);
		}
		else
		{
			// No enemies visible - clear target
			StateMachine->SetTargetEnemy(nullptr);
			StateMachine->SetThreatLevel(0.0f);
		}
	}
}

void UAttackState::ExitState(UStateMachine* StateMachine)
{
	if (MCTS)
	{
		MCTS->Backpropagate();
		UE_LOG(LogTemp, Warning, TEXT("Exited Attack State"));
	}
}


TArray<UAction*> UAttackState::GetPossibleActions()
{
	TArray<UAction*> Actions;

	Actions.Add(NewObject<USkillAttackAction>(this, USkillAttackAction::StaticClass()));
	Actions.Add(NewObject<UDeafultAttackAction>(this, UDeafultAttackAction::StaticClass()));


	return Actions;
}
