// Copyright Epic Games, Inc. All Rights Reserved.

#pragma once

#include "CoreMinimal.h"
#include "StateTreeConditionBase.h"
#include "StateTree/FollowerStateTreeSchema.h"
#include "STCondition_IsAlive.generated.h"


/**
 * State Tree Condition: Is Alive
 *
 * Checks if the follower is currently alive.
 * Used to transition to "Dead" state or prevent execution when dead.
 */


USTRUCT()
struct GAMEAI_PROJECT_API FSTCondition_IsAliveInstanceData
{
	GENERATED_BODY()

	/** Is alive (bound from context) */
	UPROPERTY(EditAnywhere, Category = "Input")
	bool bIsAlive = true;

	/** If true, checks if dead instead of alive */
	UPROPERTY(EditAnywhere, Category = "Condition")
	bool bCheckIfDead = false;
};

USTRUCT(meta = (DisplayName = "Is Alive"))
struct GAMEAI_PROJECT_API FSTCondition_IsAlive : public FStateTreeConditionBase
{
	GENERATED_BODY()

	using FInstanceDataType = FSTCondition_IsAliveInstanceData;

	FSTCondition_IsAlive() = default;

	virtual const UStruct* GetInstanceDataType() const override { return FInstanceDataType::StaticStruct(); }
	virtual bool TestCondition(FStateTreeExecutionContext& Context) const override;
};