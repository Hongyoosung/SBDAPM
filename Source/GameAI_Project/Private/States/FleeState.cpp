// Fill out your copyright notice in the Description page of Project Settings.

#include "States/FleeState.h"
#include "Core/StateMachine.h"
#include "Actions/FleeActions/SprintToCoverAction.h"
#include "Actions/FleeActions/EvasiveMovementAction.h"
#include "Actions/FleeActions/FightWhileRetreatingAction.h"

void UFleeState::EnterState(UStateMachine* StateMachine)
{
	Super::EnterState(StateMachine);

	UE_LOG(LogTemp, Warning, TEXT("Entered FleeState"));

	// Set Blackboard strategy to "Flee" to activate Flee behavior tree subtree
	if (StateMachine)
	{
		StateMachine->SetCurrentStrategy(TEXT("Flee"));
	}

	// Initialize MCTS for flee strategy selection
	if (MCTS == nullptr)
	{
		// Create new MCTS instance for this state
		MCTS = NewObject<UMCTS>();
		MCTS->InitializeMCTS();
		MCTS->InitializeCurrentNodeLocate();

		// Populate possible flee actions for MCTS to evaluate
		PossibleActions = GetPossibleActions();

		UE_LOG(LogTemp, Log, TEXT("FleeState: Initialized MCTS with %d possible actions"), PossibleActions.Num());
	}
	else
	{
		// MCTS already exists, just reset the current node
		MCTS->InitializeCurrentNodeLocate();
		UE_LOG(LogTemp, Log, TEXT("FleeState: Reusing existing MCTS instance"));
	}
}

void UFleeState::UpdateState(UStateMachine* StateMachine, float Reward, float DeltaTime)
{
	if (!MCTS)
	{
		UE_LOG(LogTemp, Error, TEXT("FleeState: MCTS is nullptr, cannot update state"));
		return;
	}

	// Run MCTS to select optimal flee strategy
	MCTS->RunMCTS(PossibleActions, StateMachine);

	// Update Blackboard based on MCTS decision and current observations
	if (StateMachine)
	{
		FObservationElement CurrentObs = StateMachine->GetCurrentObservation();

		// Check if cover is available
		if (CurrentObs.bHasCover)
		{
			// Calculate cover location based on cover direction and distance
			FVector CurrentPosition = CurrentObs.Position;
			FVector CoverDirection3D = FVector(
				CurrentObs.CoverDirection.X,
				CurrentObs.CoverDirection.Y,
				0.0f
			);
			CoverDirection3D.Normalize();

			FVector CoverLocation = CurrentPosition + (CoverDirection3D * CurrentObs.NearestCoverDistance);

			// Update Blackboard with cover location
			StateMachine->SetCoverLocation(CoverLocation);

			UE_LOG(LogTemp, Log, TEXT("FleeState: Found cover at distance %.1f, location: %s"),
				CurrentObs.NearestCoverDistance, *CoverLocation.ToString());
		}
		else
		{
			// No cover available - the Behavior Tree will use evasive movement
			// Clear the cover location or set it to an invalid value
			StateMachine->SetCoverLocation(FVector::ZeroVector);
			UE_LOG(LogTemp, Warning, TEXT("FleeState: No cover available, BT will use evasive movement"));
		}

		// Update threat level based on visible enemies
		// This helps the Behavior Tree make tactical decisions
		float ThreatLevel = FMath::Clamp(CurrentObs.VisibleEnemyCount / 10.0f, 0.0f, 1.0f);
		StateMachine->SetThreatLevel(ThreatLevel);

		UE_LOG(LogTemp, Verbose, TEXT("FleeState: Updated - Threat: %.2f, Enemies: %d, HasCover: %s"),
			ThreatLevel, CurrentObs.VisibleEnemyCount, CurrentObs.bHasCover ? TEXT("Yes") : TEXT("No"));
	}
}

void UFleeState::ExitState(UStateMachine* StateMachine)
{
	// Backpropagate MCTS rewards to update the tree
	if (MCTS)
	{
		MCTS->Backpropagate();
		UE_LOG(LogTemp, Warning, TEXT("Exited FleeState - MCTS backpropagation complete"));
	}
	else
	{
		UE_LOG(LogTemp, Warning, TEXT("Exited FleeState - No MCTS to backpropagate"));
	}
}

TArray<UAction*> UFleeState::GetPossibleActions()
{
	TArray<UAction*> Actions;

	// Create flee-specific actions that MCTS can evaluate
	// These represent strategic choices, not actual movement
	// The Behavior Tree will handle the tactical execution

	// 1. Sprint to Cover - Prioritize reaching cover quickly
	Actions.Add(NewObject<USprintToCoverAction>(this, USprintToCoverAction::StaticClass()));

	// 2. Evasive Movement - Zigzag movement when no cover or tactical advantage
	Actions.Add(NewObject<UEvasiveMovementAction>(this, UEvasiveMovementAction::StaticClass()));

	// 3. Fight While Retreating - Maintain offensive pressure while withdrawing
	Actions.Add(NewObject<UFightWhileRetreatingAction>(this, UFightWhileRetreatingAction::StaticClass()));

	UE_LOG(LogTemp, Log, TEXT("FleeState: Generated %d possible flee actions"), Actions.Num());

	return Actions;
}
