// Copyright Epic Games, Inc. All Rights Reserved.

#pragma once

#include "CoreMinimal.h"
#include "StateTreeConditionBase.h"
#include "RL/RLTypes.h"
#include "StateTree/FollowerStateTreeSchema.h"
#include "STCondition_CheckTacticalAction.generated.h"


/**
 * State Tree Condition: Check Tactical Action
 *
 * Checks if the current tactical action (from RL policy) matches the required action(s).
 * Used to control substates or task execution based on RL policy decisions.
 *
 * Example:
 * - Execute cover-finding logic when TacticalAction == SeekCover
 * - Execute flanking logic when TacticalAction IN [FlankLeft, FlankRight]
 *
 * Replaces: BTDecorator_CheckTacticalAction
 */


USTRUCT()
struct GAMEAI_PROJECT_API FSTCondition_CheckTacticalActionInstanceData
{
	GENERATED_BODY()

	/** Current tactical action (bound from context) */
	UPROPERTY(EditAnywhere, Category = "Input")
	ETacticalAction CurrentTacticalAction = ETacticalAction::DefensiveHold;

	/**
	 * Tactical actions to check against.
	 * If current action matches ANY of these, condition is true.
	 */
	UPROPERTY(EditAnywhere, Category = "Condition")
	TArray<ETacticalAction> AcceptedActions;

	/** If true, condition is inverted */
	UPROPERTY(EditAnywhere, Category = "Condition")
	bool bInvertCondition = false;
};

USTRUCT(meta = (DisplayName = "Check Tactical Action"))
struct GAMEAI_PROJECT_API FSTCondition_CheckTacticalAction : public FStateTreeConditionBase
{
	GENERATED_BODY()

	using FInstanceDataType = FSTCondition_CheckTacticalActionInstanceData;

	FSTCondition_CheckTacticalAction() = default;

	virtual const UStruct* GetInstanceDataType() const override { return FInstanceDataType::StaticStruct(); }
	virtual bool TestCondition(FStateTreeExecutionContext& Context) const override;
};