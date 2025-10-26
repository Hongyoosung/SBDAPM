// Fill out your copyright notice in the Description page of Project Settings.

#pragma once

#include "CoreMinimal.h"
#include "Actions/Action.h"
#include "MoveToAllyAction.generated.h"

/**
 * Strategic movement action: Move toward friendly units for coordination
 *
 * This action analyzes the observation data to:
 * - Identify nearby allies (via Blackboard or perception)
 * - Calculate optimal ally position to move toward
 * - Set destination for Behavior Tree to execute
 *
 * Use cases:
 * - Regrouping with team
 * - Supporting injured allies
 * - Coordinated flanking maneuvers
 */
UCLASS()
class GAMEAI_PROJECT_API UMoveToAllyAction : public UAction
{
	GENERATED_BODY()

public:
	virtual void ExecuteAction(UStateMachine* StateMachine) override;

	/**
	 * Calculate the optimal ally position to move toward
	 * @param StateMachine The state machine component
	 * @return Destination position to move toward (or ZeroVector if no allies found)
	 */
	FVector CalculateAllyDestination(UStateMachine* StateMachine) const;

	/**
	 * Evaluate the priority of moving to an ally based on context
	 * Higher values = more urgent/beneficial
	 * @param StateMachine The state machine component
	 * @return Priority score (0.0 - 1.0)
	 */
	float EvaluateAllyMovementPriority(UStateMachine* StateMachine) const;

private:
	UPROPERTY(EditAnywhere, Category = "Move To Ally")
	float MinAllyDistance = 300.0f;  // Minimum distance to maintain from ally

	UPROPERTY(EditAnywhere, Category = "Move To Ally")
	float MaxAllySearchRadius = 2000.0f;  // Maximum search radius for allies
};
