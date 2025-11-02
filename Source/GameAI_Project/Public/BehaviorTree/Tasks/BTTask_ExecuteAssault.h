// Copyright Epic Games, Inc. All Rights Reserved.

#pragma once

#include "CoreMinimal.h"
#include "BehaviorTree/BTTaskNode.h"
#include "RL/RLTypes.h"
#include "Team/TeamTypes.h"
#include "BTTask_ExecuteAssault.generated.h"

/**
 * Behavior Tree Task: Execute Assault
 *
 * Executes assault tactics based on RL policy selection.
 * This task handles offensive combat maneuvers including:
 * - Aggressive assault (direct attack)
 * - Cautious advance (measured approach)
 * - Flanking maneuvers (left/right)
 * - Suppressive fire
 *
 * Requirements:
 * - Actor must have UFollowerAgentComponent
 * - Current command must be assault-type (Assault, Flank, Suppress, Charge)
 * - FollowerAgentComponent must have TacticalPolicy initialized
 *
 * Execution Flow:
 * 1. Query RL policy for tactical action
 * 2. Execute selected tactic (movement, aiming, firing)
 * 3. Monitor action completion or interruption
 * 4. Provide reward feedback based on outcome
 *
 * Blackboard Keys:
 * - Input: "CurrentCommand" (FStrategicCommand) - Strategic command from leader
 * - Input: "TargetActor" (AActor) - Enemy to attack
 * - Input: "TargetLocation" (Vector) - Location to assault
 * - Output: "TacticalAction" (Enum) - Selected tactical action
 * - Output: "ActionProgress" (Float) - Progress of current action (0-1)
 *
 * Usage:
 * 1. Add to Behavior Tree under [CommandType == Assault] decorator
 * 2. Configure Blackboard keys
 * 3. Set tactical parameters (approach distance, fire rate, etc.)
 */
UCLASS()
class GAMEAI_PROJECT_API UBTTask_ExecuteAssault : public UBTTaskNode
{
	GENERATED_BODY()

public:
	UBTTask_ExecuteAssault();

	virtual EBTNodeResult::Type ExecuteTask(UBehaviorTreeComponent& OwnerComp, uint8* NodeMemory) override;
	virtual EBTNodeResult::Type AbortTask(UBehaviorTreeComponent& OwnerComp, uint8* NodeMemory) override;
	virtual void TickTask(UBehaviorTreeComponent& OwnerComp, uint8* NodeMemory, float DeltaSeconds) override;
	virtual FString GetStaticDescription() const override;

	//--------------------------------------------------------------------------
	// CONFIGURATION
	//--------------------------------------------------------------------------

	/** Query RL policy every N seconds (0 = only once at start) */
	UPROPERTY(EditAnywhere, Category = "Assault|RL", meta = (ClampMin = "0.0", ClampMax = "10.0"))
	float RLQueryInterval = 2.0f;

	/** Ideal engagement distance for assault (cm) */
	UPROPERTY(EditAnywhere, Category = "Assault|Tactical", meta = (ClampMin = "100.0", ClampMax = "5000.0"))
	float OptimalEngagementDistance = 1500.0f;

	/** Maximum distance to pursue target (cm) */
	UPROPERTY(EditAnywhere, Category = "Assault|Tactical", meta = (ClampMin = "500.0", ClampMax = "10000.0"))
	float MaxPursuitDistance = 3000.0f;

	/** Movement speed multiplier for assault */
	UPROPERTY(EditAnywhere, Category = "Assault|Tactical", meta = (ClampMin = "0.5", ClampMax = "2.0"))
	float AssaultSpeedMultiplier = 1.2f;

	/** Fire rate multiplier during assault */
	UPROPERTY(EditAnywhere, Category = "Assault|Combat", meta = (ClampMin = "0.5", ClampMax = "3.0"))
	float FireRateMultiplier = 1.5f;

	/** Accuracy modifier for suppressive fire (lower = less accurate) */
	UPROPERTY(EditAnywhere, Category = "Assault|Combat", meta = (ClampMin = "0.1", ClampMax = "1.0"))
	float SuppressiveAccuracyModifier = 0.3f;

	/** Distance to move when flanking (cm) */
	UPROPERTY(EditAnywhere, Category = "Assault|Tactical", meta = (ClampMin = "200.0", ClampMax = "2000.0"))
	float FlankingDistance = 800.0f;

	/** Reward for successfully closing distance to enemy */
	UPROPERTY(EditAnywhere, Category = "Assault|Reward")
	float ClosingDistanceReward = 2.0f;

	/** Reward for hitting enemy during assault */
	UPROPERTY(EditAnywhere, Category = "Assault|Reward")
	float CombatHitReward = 5.0f;

	/** Penalty for taking damage during assault */
	UPROPERTY(EditAnywhere, Category = "Assault|Reward")
	float DamageTakenPenalty = -3.0f;

	//--------------------------------------------------------------------------
	// BLACKBOARD KEYS
	//--------------------------------------------------------------------------

	/** Current strategic command from leader */
	UPROPERTY(EditAnywhere, Category = "Blackboard")
	FBlackboardKeySelector CurrentCommandKey;

	/** Target enemy actor */
	UPROPERTY(EditAnywhere, Category = "Blackboard")
	FBlackboardKeySelector TargetActorKey;

	/** Target assault location */
	UPROPERTY(EditAnywhere, Category = "Blackboard")
	FBlackboardKeySelector TargetLocationKey;

	/** Selected tactical action (output) */
	UPROPERTY(EditAnywhere, Category = "Blackboard")
	FBlackboardKeySelector TacticalActionKey;

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
	struct FBTExecuteAssaultMemory
	{
		ETacticalAction CurrentTactic = ETacticalAction::AggressiveAssault;
		float TimeInCurrentTactic = 0.0f;
		float TimeSinceLastRLQuery = 0.0f;
		float InitialDistanceToTarget = 0.0f;
		float LastDistanceToTarget = 0.0f;
		int32 HitsLanded = 0;
		int32 DamageTaken = 0;
		bool bHasTarget = false;
		FVector FlankDestination = FVector::ZeroVector;
		FVector LastPosition = FVector::ZeroVector;
	};

	virtual uint16 GetInstanceMemorySize() const override
	{
		return sizeof(FBTExecuteAssaultMemory);
	}

	/** Get follower component from AI controller */
	class UFollowerAgentComponent* GetFollowerComponent(UBehaviorTreeComponent& OwnerComp) const;

	/** Query RL policy for new tactical action */
	ETacticalAction QueryTacticalAction(UBehaviorTreeComponent& OwnerComp) const;

	/** Execute selected tactical action */
	void ExecuteTacticalAction(UBehaviorTreeComponent& OwnerComp, FBTExecuteAssaultMemory* Memory, ETacticalAction Action, float DeltaSeconds);

	/** Execute aggressive assault tactic */
	void ExecuteAggressiveAssault(UBehaviorTreeComponent& OwnerComp, FBTExecuteAssaultMemory* Memory, float DeltaSeconds);

	/** Execute cautious advance tactic */
	void ExecuteCautiousAdvance(UBehaviorTreeComponent& OwnerComp, FBTExecuteAssaultMemory* Memory, float DeltaSeconds);

	/** Execute flanking maneuver (left or right) */
	void ExecuteFlankingManeuver(UBehaviorTreeComponent& OwnerComp, FBTExecuteAssaultMemory* Memory, bool bFlankLeft, float DeltaSeconds);

	/** Execute suppressive fire */
	void ExecuteSuppressiveFire(UBehaviorTreeComponent& OwnerComp, FBTExecuteAssaultMemory* Memory, float DeltaSeconds);

	/** Get target actor from blackboard */
	AActor* GetTargetActor(UBehaviorTreeComponent& OwnerComp) const;

	/** Get target location from blackboard */
	FVector GetTargetLocation(UBehaviorTreeComponent& OwnerComp) const;

	/** Calculate tactical reward based on performance */
	float CalculateTacticalReward(UBehaviorTreeComponent& OwnerComp, const FBTExecuteAssaultMemory* Memory) const;

	/** Update action progress in blackboard */
	void UpdateActionProgress(UBehaviorTreeComponent& OwnerComp, float Progress) const;

	/** Draw debug visualization */
	void DrawDebugInfo(UBehaviorTreeComponent& OwnerComp, const FBTExecuteAssaultMemory* Memory) const;

	/** Check if assault is complete or should abort */
	bool ShouldCompleteAssault(UBehaviorTreeComponent& OwnerComp, const FBTExecuteAssaultMemory* Memory) const;

	/** Move toward target with specified speed */
	void MoveTowardTarget(UBehaviorTreeComponent& OwnerComp, const FVector& Destination, float SpeedMultiplier, float DeltaSeconds);

	/** Attempt to fire weapon at target */
	void FireAtTarget(UBehaviorTreeComponent& OwnerComp, AActor* Target, float AccuracyModifier, float FireRateMultiplier);

	/** Calculate flanking position relative to target */
	FVector CalculateFlankingPosition(UBehaviorTreeComponent& OwnerComp, AActor* Target, bool bFlankLeft) const;
};
