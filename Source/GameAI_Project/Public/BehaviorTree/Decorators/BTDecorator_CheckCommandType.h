// Copyright Epic Games, Inc. All Rights Reserved.

#pragma once

#include "CoreMinimal.h"
#include "BehaviorTree/BTDecorator.h"
#include "Team/TeamTypes.h"
#include "BTDecorator_CheckCommandType.generated.h"

/**
 * BTDecorator_CheckCommandType
 *
 * **FOLLOWER-ONLY DECORATOR**
 * Team leaders issue commands but do not receive them.
 * This decorator should only be used in BT_FollowerAgent.
 *
 * Decorator that checks if the follower's current strategic command matches
 * the required command type(s).
 *
 * Used to branch behavior tree execution based on the command received from
 * the team leader (e.g., execute assault subtree only when command is "Assault").
 *
 * Requirements:
 *   - Actor must have UFollowerAgentComponent (if not using blackboard)
 *   - BTService_SyncCommandToBlackboard should be active (if using blackboard)
 *   - Use with BT_FollowerAgent behavior tree
 *
 * Features:
 *   - Check single command type or multiple types
 *   - Observer aborts support (abort when command changes)
 *   - Invert condition (check if NOT this command type)
 *   - Blackboard integration (read command from blackboard or directly from component)
 *
 * Usage Example:
 *   Root Selector
 *   ├─ [CheckCommandType: Assault] AssaultSubtree
 *   ├─ [CheckCommandType: Defend] DefendSubtree
 *   ├─ [CheckCommandType: Support] SupportSubtree
 *   └─ [CheckCommandType: Retreat] RetreatSubtree
 *
 * Configuration:
 *   - Add decorator to composite node in BT_FollowerAgent
 *   - Select command types to match (can select multiple)
 *   - Set observer aborts to "Self" or "Lower Priority" for reactivity
 */
UCLASS()
class GAMEAI_PROJECT_API UBTDecorator_CheckCommandType : public UBTDecorator
{
	GENERATED_BODY()

public:
	UBTDecorator_CheckCommandType();

	virtual bool CalculateRawConditionValue(UBehaviorTreeComponent& OwnerComp, uint8* NodeMemory) const override;
	virtual FString GetStaticDescription() const override;

	// ========================================
	// Configuration
	// ========================================

	/**
	 * Command types to check against
	 * If current command matches ANY of these types, condition is true
	 */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Command")
	TArray<EStrategicCommandType> AcceptedCommandTypes;

	/**
	 * If true, condition is inverted (true when command does NOT match)
	 */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Command")
	bool bInvertCondition = false;

	/**
	 * If true, reads command type from blackboard key instead of FollowerAgentComponent
	 * Useful when using BTService_SyncCommandToBlackboard
	 */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Command")
	bool bUseBlackboard = true;

	/**
	 * Blackboard key containing command type (byte/enum)
	 * Only used if bUseBlackboard is true
	 */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Command", meta = (EditCondition = "bUseBlackboard"))
	FBlackboardKeySelector CommandTypeKey;

	/**
	 * If true, also checks that command is valid/active
	 * Prevents execution on expired or invalid commands
	 */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Command")
	bool bRequireValidCommand = true;

	/**
	 * Blackboard key containing command validity (bool)
	 * Only used if bRequireValidCommand and bUseBlackboard are true
	 */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Command", meta = (EditCondition = "bRequireValidCommand && bUseBlackboard"))
	FBlackboardKeySelector IsCommandValidKey;

private:
	/**
	 * Get command type from FollowerAgentComponent
	 */
	EStrategicCommandType GetCommandTypeFromComponent(UBehaviorTreeComponent& OwnerComp) const;

	/**
	 * Get command type from blackboard
	 */
	EStrategicCommandType GetCommandTypeFromBlackboard(UBehaviorTreeComponent& OwnerComp) const;

	/**
	 * Check if command is valid
	 */
	bool IsCommandValid(UBehaviorTreeComponent& OwnerComp) const;
};
