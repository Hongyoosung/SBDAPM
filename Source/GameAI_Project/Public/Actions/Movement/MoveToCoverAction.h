// Fill out your copyright notice in the Description page of Project Settings.

#pragma once

#include "CoreMinimal.h"
#include "Actions/Action.h"
#include "MoveToCoverAction.generated.h"

/**
 * Strategic movement action: Move to nearest cover position
 *
 * This action analyzes the observation data to:
 * - Identify available cover positions
 * - Evaluate cover quality (distance from enemy, protection level)
 * - Calculate optimal cover position
 * - Set destination for Behavior Tree to execute
 *
 * Use cases:
 * - Defensive retreat under fire
 * - Repositioning during combat
 * - Healing/reloading safely
 */
UCLASS()
class GAMEAI_PROJECT_API UMoveToCoverAction : public UAction
{
	GENERATED_BODY()

public:
	virtual void ExecuteAction(UStateMachine* StateMachine) override;

	/**
	 * Calculate the best cover position to move to
	 * @param StateMachine The state machine component
	 * @return Cover destination (or ZeroVector if no cover available)
	 */
	FVector CalculateCoverDestination(UStateMachine* StateMachine) const;

	/**
	 * Evaluate the urgency of seeking cover
	 * @param StateMachine The state machine component
	 * @return Urgency score (0.0 - 1.0, higher = more urgent)
	 */
	float EvaluateCoverUrgency(UStateMachine* StateMachine) const;

	/**
	 * Check if current cover position is still safe
	 * @param StateMachine The state machine component
	 * @return true if cover is compromised/unsafe, false if still safe
	 */
	bool IsCoverCompromised(UStateMachine* StateMachine) const;

private:
	/**
	 * Fallback method to find cover using raycast data when observation doesn't have cover info
	 */
	FVector FindCoverUsingRaycasts(UStateMachine* StateMachine) const;

	UPROPERTY(EditAnywhere, Category = "Cover")
	float MaxCoverSearchRadius = 1500.0f;  // Maximum distance to search for cover

	UPROPERTY(EditAnywhere, Category = "Cover")
	float MinCoverDistance = 100.0f;  // Minimum distance from current position to consider

	UPROPERTY(EditAnywhere, Category = "Cover")
	float PreferredCoverDistance = 500.0f;  // Preferred distance to cover (not too far)
};
