// Copyright Epic Games, Inc. All Rights Reserved.

#pragma once

#include "CoreMinimal.h"
#include "BehaviorTree/BTTaskNode.h"
#include "RL/RLTypes.h"
#include "BTTask_UpdateTacticalReward.generated.h"

/**
 * Reward calculation modes for tactical reward task
 */
UENUM(BlueprintType)
enum class ERewardMode : uint8
{
	Manual UMETA(DisplayName = "Manual (Specify Value)"),
	FromBlackboard UMETA(DisplayName = "From Blackboard"),
	AutoCalculate UMETA(DisplayName = "Auto-Calculate")
};

/**
 * Behavior Tree Task: Update Tactical Reward
 *
 * Provides reward feedback to the follower's RL policy based on
 * recent events and tactical outcomes.
 *
 * Reward calculation uses:
 * - Combat events (kills, damage dealt/taken)
 * - Tactical positioning (cover, formation)
 * - Command execution (progress, success)
 * - Support actions (assist allies, share ammo)
 *
 * Requirements:
 * - Actor must have UFollowerAgentComponent
 * - FollowerAgentComponent must have TacticalPolicy initialized
 *
 * Usage:
 * 1. Add this task to your Behavior Tree after combat/tactical actions
 * 2. Configure reward type (manual, auto-calculate, or from blackboard)
 * 3. Task will provide reward to RL policy for experience collection
 */
UCLASS()
class GAMEAI_PROJECT_API UBTTask_UpdateTacticalReward : public UBTTaskNode
{
	GENERATED_BODY()

public:
	UBTTask_UpdateTacticalReward();

	virtual EBTNodeResult::Type ExecuteTask(UBehaviorTreeComponent& OwnerComp, uint8* NodeMemory) override;
	virtual FString GetStaticDescription() const override;

	//--------------------------------------------------------------------------
	// CONFIGURATION
	//--------------------------------------------------------------------------

	/** Reward calculation mode */
	UPROPERTY(EditAnywhere, Category = "Reward")
	ERewardMode RewardMode = ERewardMode::AutoCalculate;

	/** Manual reward value (used if RewardMode = Manual) */
	UPROPERTY(EditAnywhere, Category = "Reward", meta = (EditCondition = "RewardMode == ERewardMode::Manual"))
	float ManualReward = 0.0f;

	/** Blackboard key to read reward from (used if RewardMode = FromBlackboard) */
	UPROPERTY(EditAnywhere, Category = "Reward", meta = (EditCondition = "RewardMode == ERewardMode::FromBlackboard"))
	FBlackboardKeySelector RewardBlackboardKey;

	/** Is this a terminal state? (episode end) */
	UPROPERTY(EditAnywhere, Category = "Reward")
	bool bTerminalState = false;

	/** Blackboard key for terminal state flag (optional) */
	UPROPERTY(EditAnywhere, Category = "Reward")
	FBlackboardKeySelector TerminalStateKey;

	/** Auto-calculate combat rewards (kills, damage) */
	UPROPERTY(EditAnywhere, Category = "Reward|AutoCalculate")
	bool bCalculateCombatRewards = true;

	/** Auto-calculate tactical rewards (cover, formation) */
	UPROPERTY(EditAnywhere, Category = "Reward|AutoCalculate")
	bool bCalculateTacticalRewards = true;

	/** Auto-calculate command rewards (progress, success) */
	UPROPERTY(EditAnywhere, Category = "Reward|AutoCalculate")
	bool bCalculateCommandRewards = true;

	/** Log reward to console */
	UPROPERTY(EditAnywhere, Category = "Debug")
	bool bLogReward = true;

protected:
	/** Get follower component from AI controller */
	class UFollowerAgentComponent* GetFollowerComponent(UBehaviorTreeComponent& OwnerComp) const;

	/** Auto-calculate reward based on agent state */
	float CalculateReward(UBehaviorTreeComponent& OwnerComp, UFollowerAgentComponent* FollowerComp) const;

	/** Calculate combat-related rewards */
	float CalculateCombatReward(UBehaviorTreeComponent& OwnerComp, UFollowerAgentComponent* FollowerComp) const;

	/** Calculate tactical positioning rewards */
	float CalculateTacticalReward(UBehaviorTreeComponent& OwnerComp, UFollowerAgentComponent* FollowerComp) const;

	/** Calculate command execution rewards */
	float CalculateCommandReward(UBehaviorTreeComponent& OwnerComp, UFollowerAgentComponent* FollowerComp) const;
};
