// Fill out your copyright notice in the Description page of Project Settings.

#pragma once

#include "CoreMinimal.h"
#include "Actions/Action.h"
#include "TacticalRepositionAction.generated.h"

/**
 * Strategic movement action: Dynamically reposition based on combat situation
 *
 * This action analyzes the observation data to:
 * - Evaluate current position's tactical value
 * - Identify weaknesses (flanked, surrounded, poor visibility)
 * - Calculate optimal repositioning destination
 * - Adapt to changing combat dynamics
 *
 * Use cases:
 * - Evading flanking enemies
 * - Repositioning when cover is compromised
 * - Gaining better line of sight
 * - Adapting to enemy movements
 */
UCLASS()
class GAMEAI_PROJECT_API UTacticalRepositionAction : public UAction
{
	GENERATED_BODY()

public:
	virtual void ExecuteAction(UStateMachine* StateMachine) override;

	/**
	 * Evaluate the quality of the current tactical position
	 * @param StateMachine The state machine component
	 * @return Quality score (0.0 - 1.0, higher is better position)
	 */
	float EvaluateCurrentPosition(UStateMachine* StateMachine) const;

	/**
	 * Calculate optimal reposition destination based on threats and opportunities
	 * @param StateMachine The state machine component
	 * @return Reposition destination (or ZeroVector if current position is good)
	 */
	FVector CalculateRepositionDestination(UStateMachine* StateMachine) const;

	/**
	 * Determine if repositioning is necessary
	 * @param StateMachine The state machine component
	 * @return true if should reposition, false if current position is acceptable
	 */
	bool ShouldReposition(UStateMachine* StateMachine) const;

private:
	/**
	 * Find position that avoids enemy fire lines
	 */
	FVector FindSafePosition(UStateMachine* StateMachine) const;

	/**
	 * Calculate position that maximizes tactical advantage
	 * Considers: cover, line of sight, ally proximity, escape routes
	 */
	FVector CalculateOptimalTacticalPosition(UStateMachine* StateMachine) const;

	/**
	 * Analyze threat vectors from enemies
	 * @return Array of threat directions (normalized)
	 */
	TArray<FVector> AnalyzeThreatVectors(const struct FObservationElement& Observation) const;

	UPROPERTY(EditAnywhere, Category = "Tactical Reposition")
	float RepositionThreshold = 0.4f;  // Position quality below this triggers reposition

	UPROPERTY(EditAnywhere, Category = "Tactical Reposition")
	float MinRepositionDistance = 300.0f;  // Minimum distance to move when repositioning

	UPROPERTY(EditAnywhere, Category = "Tactical Reposition")
	float MaxRepositionDistance = 1000.0f;  // Maximum distance to move when repositioning

	UPROPERTY(EditAnywhere, Category = "Tactical Reposition")
	float CoverWeight = 0.4f;  // Importance of cover in position evaluation

	UPROPERTY(EditAnywhere, Category = "Tactical Reposition")
	float AllyProximityWeight = 0.2f;  // Importance of ally proximity

	UPROPERTY(EditAnywhere, Category = "Tactical Reposition")
	float EnemyAvoidanceWeight = 0.3f;  // Importance of avoiding enemies

	UPROPERTY(EditAnywhere, Category = "Tactical Reposition")
	float VisibilityWeight = 0.1f;  // Importance of good visibility/line of sight
};
