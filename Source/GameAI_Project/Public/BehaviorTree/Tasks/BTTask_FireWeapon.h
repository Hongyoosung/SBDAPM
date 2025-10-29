// Copyright Epic Games, Inc. All Rights Reserved.

#pragma once

#include "CoreMinimal.h"
#include "BehaviorTree/BTTaskNode.h"
#include "BTTask_FireWeapon.generated.h"

/**
 * BTTask_FireWeapon
 *
 * Executes weapon fire at a target actor.
 * Integrates with ICombatStatsInterface for weapon state.
 * Provides rewards to RL policy based on hit/miss/kill outcomes.
 *
 * Workflow:
 *   1. Check if weapon is ready (cooldown, ammo)
 *   2. Get target from blackboard
 *   3. Check line of sight to target
 *   4. Fire weapon (raycast or projectile)
 *   5. Apply damage if hit
 *   6. Provide reward to RL policy (+10 kill, +5 damage, 0 miss)
 *   7. Return success if fired, failure if couldn't fire
 *
 * Usage:
 *   - Place in BT under combat subtrees
 *   - Assign TargetActorKey to blackboard key with enemy reference
 *   - Configure FireRate, Damage, Range, Accuracy
 *   - Task completes after single shot (use in loop for sustained fire)
 */
UCLASS()
class GAMEAI_PROJECT_API UBTTask_FireWeapon : public UBTTaskNode
{
	GENERATED_BODY()

public:
	UBTTask_FireWeapon();

	virtual EBTNodeResult::Type ExecuteTask(UBehaviorTreeComponent& OwnerComp, uint8* NodeMemory) override;
	virtual FString GetStaticDescription() const override;

	// ========================================
	// Configuration
	// ========================================

	/**
	 * Blackboard key containing the target actor to fire at
	 */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Target")
	FBlackboardKeySelector TargetActorKey;

	/**
	 * Maximum firing range (units)
	 * Weapon won't fire at targets beyond this range
	 */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Weapon")
	float MaxRange = 3000.0f;

	/**
	 * Base damage per shot
	 */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Weapon")
	float BaseDamage = 10.0f;

	/**
	 * Accuracy factor (0.0 - 1.0)
	 * 1.0 = perfect accuracy, 0.0 = always miss
	 * Accuracy decreases with distance and target movement
	 */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Weapon")
	float BaseAccuracy = 0.8f;

	/**
	 * Accuracy penalty per 1000 units of distance
	 * Example: 0.2 = -20% accuracy per 1000 units
	 */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Weapon")
	float AccuracyDistancePenalty = 0.15f;

	/**
	 * If true, requires line of sight to target before firing
	 */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Weapon")
	bool bRequireLineOfSight = true;

	/**
	 * If true, applies camera recoil/shake when firing
	 */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Effects")
	bool bApplyCameraShake = true;

	/**
	 * If true, spawns muzzle flash particle effect
	 */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Effects")
	bool bSpawnMuzzleFlash = true;

	/**
	 * If true, spawns tracer/projectile visual
	 */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Effects")
	bool bSpawnTracer = true;

	/**
	 * If true, provides reward to RL policy after firing
	 */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Reinforcement Learning")
	bool bProvideReward = true;

	/**
	 * Reward values for different outcomes
	 */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Reinforcement Learning")
	float RewardForKill = 10.0f;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Reinforcement Learning")
	float RewardForHit = 5.0f;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Reinforcement Learning")
	float RewardForMiss = 0.0f;

	/**
	 * If true, logs firing events to console for debugging
	 */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Debug")
	bool bLogFiring = false;

	/**
	 * If true, draws debug lines showing fire trajectory
	 */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Debug")
	bool bDrawDebugLines = false;

private:
	/**
	 * Calculate accuracy based on distance and target movement
	 */
	float CalculateAccuracy(float Distance, const FVector& TargetVelocity) const;

	/**
	 * Perform raycast to check line of sight
	 */
	bool HasLineOfSight(APawn* Shooter, AActor* Target) const;

	/**
	 * Execute the actual weapon fire
	 */
	bool FireWeapon(APawn* Shooter, AActor* Target, float& OutDamageDealt, bool& OutKilled);

	/**
	 * Apply damage to target
	 */
	bool ApplyDamageToTarget(AActor* Target, float Damage, APawn* Instigator, bool& OutKilled);

	/**
	 * Provide reward to RL policy
	 */
	void ProvideRewardToRLPolicy(UBehaviorTreeComponent& OwnerComp, float Reward);

	/**
	 * Spawn visual effects (muzzle flash, tracer, etc.)
	 */
	void SpawnFireEffects(APawn* Shooter, const FVector& FireStart, const FVector& FireEnd, bool bHit);
};
