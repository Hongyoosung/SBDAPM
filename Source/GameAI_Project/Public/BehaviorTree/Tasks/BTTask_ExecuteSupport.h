// Copyright Epic Games, Inc. All Rights Reserved.

#pragma once

#include "CoreMinimal.h"
#include "BehaviorTree/BTTaskNode.h"
#include "RL/RLTypes.h"
#include "Team/TeamTypes.h"
#include "BTTask_ExecuteSupport.generated.h"

/**
 * Behavior Tree Task: Execute Support
 *
 * Executes support tactics based on RL policy selection.
 * This task handles support and team-oriented actions including:
 * - Covering fire (suppress enemies to help allies)
 * - Reload (tactical reload when safe)
 * - Use ability (use special abilities/items)
 * - Rescue ally (move to assist ally in danger)
 *
 * Requirements:
 * - Actor must have UFollowerAgentComponent
 * - Current command must be support-type (RescueAlly, ProvideSupport, Regroup, ShareAmmo)
 * - FollowerAgentComponent must have TacticalPolicy initialized
 *
 * Execution Flow:
 * 1. Query RL policy for tactical action
 * 2. Identify ally in need or threat to suppress
 * 3. Execute selected support tactic
 * 4. Provide reward feedback based on ally assistance
 *
 * Blackboard Keys:
 * - Input: "CurrentCommand" (FStrategicCommand) - Strategic command from leader
 * - Input: "AllyToSupport" (AActor) - Ally requiring support
 * - Input: "SupportLocation" (Vector) - Location to provide support from
 * - Output: "TacticalAction" (Enum) - Selected tactical action
 * - Output: "SupportTarget" (AActor) - Current support target (ally or enemy)
 * - Output: "ActionProgress" (Float) - Progress of current action (0-1)
 *
 * Usage:
 * 1. Add to Behavior Tree under [CommandType == Support] decorator
 * 2. Configure Blackboard keys
 * 3. Set support parameters (assist radius, reload threshold, etc.)
 */
UCLASS()
class GAMEAI_PROJECT_API UBTTask_ExecuteSupport : public UBTTaskNode
{
	GENERATED_BODY()

public:
	UBTTask_ExecuteSupport();

	virtual EBTNodeResult::Type ExecuteTask(UBehaviorTreeComponent& OwnerComp, uint8* NodeMemory) override;
	virtual EBTNodeResult::Type AbortTask(UBehaviorTreeComponent& OwnerComp, uint8* NodeMemory) override;
	virtual void TickTask(UBehaviorTreeComponent& OwnerComp, uint8* NodeMemory, float DeltaSeconds) override;
	virtual FString GetStaticDescription() const override;

	//--------------------------------------------------------------------------
	// CONFIGURATION
	//--------------------------------------------------------------------------

	/** Query RL policy every N seconds (0 = only once at start) */
	UPROPERTY(EditAnywhere, Category = "Support|RL", meta = (ClampMin = "0.0", ClampMax = "10.0"))
	float RLQueryInterval = 2.5f;

	/** Maximum distance to provide support (cm) */
	UPROPERTY(EditAnywhere, Category = "Support|Tactical", meta = (ClampMin = "500.0", ClampMax = "5000.0"))
	float MaxSupportRange = 2500.0f;

	/** Ideal distance to ally when providing support (cm) */
	UPROPERTY(EditAnywhere, Category = "Support|Tactical", meta = (ClampMin = "200.0", ClampMax = "2000.0"))
	float OptimalSupportDistance = 800.0f;

	/** Ammunition threshold to trigger reload (0-1) */
	UPROPERTY(EditAnywhere, Category = "Support|Combat", meta = (ClampMin = "0.0", ClampMax = "1.0"))
	float ReloadThreshold = 0.3f;

	/** Fire rate multiplier for covering fire */
	UPROPERTY(EditAnywhere, Category = "Support|Combat", meta = (ClampMin = "0.5", ClampMax = "3.0"))
	float CoveringFireRateMultiplier = 1.8f;

	/** Accuracy modifier for covering fire (lower = suppressive) */
	UPROPERTY(EditAnywhere, Category = "Support|Combat", meta = (ClampMin = "0.1", ClampMax = "1.0"))
	float CoveringFireAccuracy = 0.4f;

	/** Health threshold to consider ally in danger (0-1) */
	UPROPERTY(EditAnywhere, Category = "Support|Tactical", meta = (ClampMin = "0.0", ClampMax = "1.0"))
	float AllyDangerHealthThreshold = 0.4f;

	/** Reward for providing covering fire */
	UPROPERTY(EditAnywhere, Category = "Support|Reward")
	float CoveringFireReward = 4.0f;

	/** Reward for successfully reloading safely */
	UPROPERTY(EditAnywhere, Category = "Support|Reward")
	float SafeReloadReward = 2.0f;

	/** Reward for rescuing ally */
	UPROPERTY(EditAnywhere, Category = "Support|Reward")
	float RescueAllyReward = 10.0f;

	/** Penalty for failing to reach ally in time */
	UPROPERTY(EditAnywhere, Category = "Support|Reward")
	float FailedRescuePenalty = -8.0f;

	//--------------------------------------------------------------------------
	// BLACKBOARD KEYS
	//--------------------------------------------------------------------------

	/** Current strategic command from leader */
	UPROPERTY(EditAnywhere, Category = "Blackboard")
	FBlackboardKeySelector CurrentCommandKey;

	/** Ally requiring support */
	UPROPERTY(EditAnywhere, Category = "Blackboard")
	FBlackboardKeySelector AllyToSupportKey;

	/** Location to provide support from */
	UPROPERTY(EditAnywhere, Category = "Blackboard")
	FBlackboardKeySelector SupportLocationKey;

	/** Selected tactical action (output) */
	UPROPERTY(EditAnywhere, Category = "Blackboard")
	FBlackboardKeySelector TacticalActionKey;

	/** Current support target - ally or enemy (output) */
	UPROPERTY(EditAnywhere, Category = "Blackboard")
	FBlackboardKeySelector SupportTargetKey;

	/** Action progress 0-1 (output) */
	UPROPERTY(EditAnywhere, Category = "Blackboard")
	FBlackboardKeySelector ActionProgressKey;

	//--------------------------------------------------------------------------
	// DEBUG
	//--------------------------------------------------------------------------

	/** Log tactical actions to console */
	UPROPERTY(EditAnywhere, Category = "Debug")
	bool bLogActions = true;

	/** Draw debug visualization */
	UPROPERTY(EditAnywhere, Category = "Debug")
	bool bDrawDebugInfo = false;

protected:
	/** Task memory structure */
	struct FBTExecuteSupportMemory
	{
		ETacticalAction CurrentTactic = ETacticalAction::ProvideCoveringFire;
		float TimeInCurrentTactic = 0.0f;
		float TimeSinceLastRLQuery = 0.0f;
		AActor* AllyBeingSupported = nullptr;
		AActor* ThreatToSuppress = nullptr;
		bool bReloadInProgress = false;
		bool bAllyRescued = false;
		float DistanceToAlly = 0.0f;
		float InitialAllyHealth = 1.0f;
		int32 CoveringFireShots = 0;
		int32 ThreatsNeutralized = 0;
	};

	virtual uint16 GetInstanceMemorySize() const override
	{
		return sizeof(FBTExecuteSupportMemory);
	}

	/** Get follower component from AI controller */
	class UFollowerAgentComponent* GetFollowerComponent(UBehaviorTreeComponent& OwnerComp) const;

	/** Query RL policy for new tactical action */
	ETacticalAction QueryTacticalAction(UBehaviorTreeComponent& OwnerComp) const;

	/** Execute selected tactical action */
	void ExecuteTacticalAction(UBehaviorTreeComponent& OwnerComp, FBTExecuteSupportMemory* Memory, ETacticalAction Action, float DeltaSeconds);

	/** Execute covering fire tactic */
	void ExecuteCoveringFire(UBehaviorTreeComponent& OwnerComp, FBTExecuteSupportMemory* Memory, float DeltaSeconds);

	/** Execute reload tactic */
	void ExecuteReload(UBehaviorTreeComponent& OwnerComp, FBTExecuteSupportMemory* Memory, float DeltaSeconds);

	/** Execute use ability tactic */
	void ExecuteUseAbility(UBehaviorTreeComponent& OwnerComp, FBTExecuteSupportMemory* Memory, float DeltaSeconds);

	/** Execute rescue/assist ally tactic */
	void ExecuteRescueAlly(UBehaviorTreeComponent& OwnerComp, FBTExecuteSupportMemory* Memory, float DeltaSeconds);

	/** Get ally to support from blackboard or find nearest ally in danger */
	AActor* GetAllyToSupport(UBehaviorTreeComponent& OwnerComp) const;

	/** Find nearest ally in danger */
	AActor* FindNearestAllyInDanger(UBehaviorTreeComponent& OwnerComp) const;

	/** Find threats engaging specified ally */
	TArray<AActor*> FindThreatsEngagingAlly(UBehaviorTreeComponent& OwnerComp, AActor* Ally) const;

	/** Get support location from blackboard */
	FVector GetSupportLocation(UBehaviorTreeComponent& OwnerComp) const;

	/** Get ally health percentage */
	float GetAllyHealthPercentage(AActor* Ally) const;

	/** Calculate tactical reward based on support performance */
	float CalculateTacticalReward(UBehaviorTreeComponent& OwnerComp, const FBTExecuteSupportMemory* Memory) const;

	/** Update action progress in blackboard */
	void UpdateActionProgress(UBehaviorTreeComponent& OwnerComp, float Progress) const;

	/** Draw debug visualization */
	void DrawDebugInfo(UBehaviorTreeComponent& OwnerComp, const FBTExecuteSupportMemory* Memory) const;

	/** Check if support is complete or should abort */
	bool ShouldCompleteSupport(UBehaviorTreeComponent& OwnerComp, const FBTExecuteSupportMemory* Memory) const;

	/** Move to support position */
	void MoveToSupportPosition(UBehaviorTreeComponent& OwnerComp, const FVector& Destination, float DeltaSeconds);

	/** Provide covering fire at threat */
	void ProvideCoveringFireAtThreat(UBehaviorTreeComponent& OwnerComp, AActor* Threat, float AccuracyModifier, float FireRateMultiplier);
};
