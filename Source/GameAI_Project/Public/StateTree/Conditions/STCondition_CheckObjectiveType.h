// Copyright Epic Games, Inc. All Rights Reserved.

#pragma once

#include "CoreMinimal.h"
#include "StateTreeConditionBase.h"
#include "StateTree/FollowerStateTreeSchema.h"
#include "Team/Objective.h"
#include "STCondition_CheckObjectiveType.generated.h"

/**
 * State Tree Condition: Check Objective Type
 *
 * Checks if the current objective matches the required type(s).
 * Used to control state transitions based on leader-assigned objectives.
 *
 * Example:
 * - Transition to "Assault" state when ObjectiveType == Eliminate
 * - Transition to "Defend" state when ObjectiveType IN [DefendObjective, CaptureObjective]
 *
 * Replaces: STCondition_CheckCommandType (v2.0)
 */

USTRUCT()
struct GAMEAI_PROJECT_API FSTCondition_CheckObjectiveTypeInstanceData
{
	GENERATED_BODY()

	//--------------------------------------------------------------------------
	// INPUT BINDINGS (bind these in StateTree editor)
	//--------------------------------------------------------------------------

	/**
	 * Current objective - bind to FollowerContext.CurrentObjective
	 */
	UPROPERTY(EditAnywhere, Category = "Input")
	TObjectPtr<UObjective> CurrentObjective = nullptr;

	/**
	 * Is objective valid - bind to FollowerContext.bHasActiveObjective
	 */
	UPROPERTY(EditAnywhere, Category = "Input")
	bool bHasActiveObjective = false;

	//--------------------------------------------------------------------------
	// CONFIGURATION
	//--------------------------------------------------------------------------

	/**
	 * Objective types to check against.
	 * If current objective matches ANY of these types, condition is true.
	 */
	UPROPERTY(EditAnywhere, Category = "Parameter")
	TArray<EObjectiveType> AcceptedObjectiveTypes;

	/** If true, condition is inverted (true when objective does NOT match) */
	UPROPERTY(EditAnywhere, Category = "Parameter")
	bool bInvertCondition = false;

	/** If true, also checks that objective is active */
	UPROPERTY(EditAnywhere, Category = "Parameter")
	bool bRequireActiveObjective = true;
};

USTRUCT(meta = (DisplayName = "Check Objective Type"))
struct GAMEAI_PROJECT_API FSTCondition_CheckObjectiveType : public FStateTreeConditionBase
{
	GENERATED_BODY()

	using FInstanceDataType = FSTCondition_CheckObjectiveTypeInstanceData;

	FSTCondition_CheckObjectiveType() = default;

	virtual const UStruct* GetInstanceDataType() const override { return FInstanceDataType::StaticStruct(); }
	virtual bool TestCondition(FStateTreeExecutionContext& Context) const override;
};
