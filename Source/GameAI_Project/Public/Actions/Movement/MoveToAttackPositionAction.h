// Fill out your copyright notice in the Description page of Project Settings.

#pragma once

#include "CoreMinimal.h"
#include "Actions/Action.h"
#include "MoveToAttackPositionAction.generated.h"

/**
 * Enum for different attack position strategies
 */
UENUM(BlueprintType)
enum class EAttackPositionStrategy : uint8
{
    Flanking UMETA(DisplayName = "Flanking"),        // Move to enemy's side/rear
    HighGround UMETA(DisplayName = "High Ground"),   // Seek elevated position
    CoverBased UMETA(DisplayName = "Cover-Based"),   // Move to position with nearby cover
    DirectAssault UMETA(DisplayName = "Direct"),     // Move directly toward enemy
    Encircle UMETA(DisplayName = "Encircle")         // Surround enemy (multi-agent)
};

/**
 * Strategic movement action: Move to an optimal attack position
 *
 * This action analyzes the observation data to:
 * - Identify target enemy
 * - Evaluate tactical positions (flanking, high ground, cover)
 * - Calculate optimal attack position based on strategy
 * - Set destination for Behavior Tree to execute
 *
 * Use cases:
 * - Positioning before engaging enemy
 * - Flanking maneuvers
 * - Taking advantageous positions (high ground, cover)
 * - Coordinated assaults
 */
UCLASS()
class GAMEAI_PROJECT_API UMoveToAttackPositionAction : public UAction
{
	GENERATED_BODY()

public:
	virtual void ExecuteAction(UStateMachine* StateMachine) override;

	/**
	 * Calculate the optimal attack position based on current strategy
	 * @param StateMachine The state machine component
	 * @param Strategy The attack position strategy to use
	 * @return Destination position for attacking (or ZeroVector if invalid)
	 */
	FVector CalculateAttackPosition(UStateMachine* StateMachine, EAttackPositionStrategy Strategy) const;

	/**
	 * Select the best attack strategy based on current observations
	 * @param StateMachine The state machine component
	 * @return Best strategy for current situation
	 */
	EAttackPositionStrategy SelectBestStrategy(UStateMachine* StateMachine) const;

	/**
	 * Evaluate a potential attack position's quality
	 * @param Position The position to evaluate
	 * @param EnemyPosition The target enemy's position
	 * @param MyPosition Current agent position
	 * @param CurrentObs Current observation data
	 * @return Quality score (0.0 - 1.0, higher is better)
	 */
	float EvaluateAttackPosition(const FVector& Position, const FVector& EnemyPosition,
	                              const FVector& MyPosition, const struct FObservationElement& CurrentObs) const;

private:
	/**
	 * Find flanking position relative to enemy
	 */
	FVector CalculateFlankingPosition(const FVector& MyPosition, const FVector& EnemyPosition,
	                                   const FRotator& EnemyRotation) const;

	/**
	 * Find high ground position if available
	 */
	FVector CalculateHighGroundPosition(UStateMachine* StateMachine, const FVector& EnemyPosition) const;

	/**
	 * Find position near cover with line of sight to enemy
	 */
	FVector CalculateCoverBasedPosition(UStateMachine* StateMachine, const FVector& EnemyPosition) const;

	UPROPERTY(EditAnywhere, Category = "Attack Position")
	float OptimalAttackRange = 800.0f;  // Preferred distance from enemy

	UPROPERTY(EditAnywhere, Category = "Attack Position")
	float MinAttackRange = 300.0f;  // Minimum safe distance from enemy

	UPROPERTY(EditAnywhere, Category = "Attack Position")
	float FlankingAngleMin = 90.0f;  // Minimum angle for flanking (degrees)

	UPROPERTY(EditAnywhere, Category = "Attack Position")
	float HighGroundHeightBonus = 200.0f;  // Z-height advantage to seek
};
