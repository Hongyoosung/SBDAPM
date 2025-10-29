// Copyright Epic Games, Inc. All Rights Reserved.

#pragma once

#include "CoreMinimal.h"
#include "BehaviorTree/BTService.h"
#include "BTService_SyncCommandToBlackboard.generated.h"

/**
 * BTService_SyncCommandToBlackboard
 *
 * Periodically syncs the follower's current strategic command from
 * FollowerAgentComponent to the Blackboard.
 *
 * This allows BT decorators and tasks to check the current command type
 * and respond accordingly (e.g., switch to assault subtree when command is "Assault").
 *
 * Synced Data:
 *   - CommandType (EStrategicCommandType as byte)
 *   - CommandTarget (AActor*)
 *   - CommandPriority (int32)
 *   - TimeSinceCommand (float)
 *   - IsCommandValid (bool)
 *
 * Usage:
 *   - Add to root composite node or high-level selector
 *   - Configure update interval (default: 0.5s)
 *   - Set blackboard keys for synced data
 *   - Use BTDecorator_CheckCommandType to branch based on command
 */
UCLASS()
class GAMEAI_PROJECT_API UBTService_SyncCommandToBlackboard : public UBTService
{
	GENERATED_BODY()

public:
	UBTService_SyncCommandToBlackboard();

	virtual void TickNode(UBehaviorTreeComponent& OwnerComp, uint8* NodeMemory, float DeltaSeconds) override;
	virtual FString GetStaticDescription() const override;

	// ========================================
	// Blackboard Keys
	// ========================================

	/**
	 * Blackboard key to store command type (byte)
	 * Will store EStrategicCommandType as uint8
	 */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Blackboard")
	FBlackboardKeySelector CommandTypeKey;

	/**
	 * Blackboard key to store command target actor (object)
	 */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Blackboard")
	FBlackboardKeySelector CommandTargetKey;

	/**
	 * Blackboard key to store command priority (int)
	 */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Blackboard")
	FBlackboardKeySelector CommandPriorityKey;

	/**
	 * Blackboard key to store time since command was received (float)
	 */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Blackboard")
	FBlackboardKeySelector TimeSinceCommandKey;

	/**
	 * Blackboard key to store whether command is valid (bool)
	 * False if no command received yet or command expired
	 */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Blackboard")
	FBlackboardKeySelector IsCommandValidKey;

	// ========================================
	// Configuration
	// ========================================

	/**
	 * If true, clears command from blackboard if FollowerAgentComponent not found
	 */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Config")
	bool bClearOnNoFollowerComponent = true;

	/**
	 * If true, logs command sync events to console for debugging
	 */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Debug")
	bool bLogSync = false;
};
