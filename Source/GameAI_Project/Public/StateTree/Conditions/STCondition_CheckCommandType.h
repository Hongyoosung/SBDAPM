// Copyright Epic Games, Inc. All Rights Reserved.

#pragma once

#include "CoreMinimal.h"
#include "StateTreeConditionBase.h"
#include "StateTree/FollowerStateTreeSchema.h"
#include "Team/TeamTypes.h"
#include "STCondition_CheckCommandType.generated.h"

/**
 * State Tree Condition: Check Command Type
 *
 * Checks if the current strategic command matches the required type(s).
 * Used to control state transitions based on leader commands.
 *
 * Example:
 * - Transition to "Assault" state when CommandType == Assault
 * - Transition to "Defend" state when CommandType IN [Defend, HoldPosition, TakeCover]
 *
 * Replaces: BTDecorator_CheckCommandType
 */

USTRUCT()
struct GAMEAI_PROJECT_API FSTCondition_CheckCommandTypeInstanceData
{
	GENERATED_BODY()

	//--------------------------------------------------------------------------
	// INPUT BINDINGS (bind these in StateTree editor)
	//--------------------------------------------------------------------------

	/**
	 * Current command type - bind to FollowerContext.CurrentCommand.CommandType
	 */
	UPROPERTY(EditAnywhere, Category = "Input")
	EStrategicCommandType CurrentCommandType = EStrategicCommandType::None;

	/**
	 * Is command valid - bind to FollowerContext.bIsCommandValid
	 */
	UPROPERTY(EditAnywhere, Category = "Input")
	bool bIsCommandValid = false;

	//--------------------------------------------------------------------------
	// CONFIGURATION
	//--------------------------------------------------------------------------

	/**
	 * Command types to check against.
	 * If current command matches ANY of these types, condition is true.
	 */
	UPROPERTY(EditAnywhere, Category = "Parameter")
	TArray<EStrategicCommandType> AcceptedCommandTypes;

	/** If true, condition is inverted (true when command does NOT match) */
	UPROPERTY(EditAnywhere, Category = "Parameter")
	bool bInvertCondition = false;

	/** If true, also checks that command is valid/active */
	UPROPERTY(EditAnywhere, Category = "Parameter")
	bool bRequireValidCommand = true;
};

USTRUCT(meta = (DisplayName = "Check Command Type"))
struct GAMEAI_PROJECT_API FSTCondition_CheckCommandType : public FStateTreeConditionBase
{
	GENERATED_BODY()

	using FInstanceDataType = FSTCondition_CheckCommandTypeInstanceData;

	FSTCondition_CheckCommandType() = default;

	virtual const UStruct* GetInstanceDataType() const override { return FInstanceDataType::StaticStruct(); }
	virtual bool TestCondition(FStateTreeExecutionContext& Context) const override;
};