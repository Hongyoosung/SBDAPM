// Copyright Epic Games, Inc. All Rights Reserved.

#pragma once

#include "CoreMinimal.h"
#include "StateTreeConditionBase.h"
#include "StateTree/FollowerStateTreeSchema.h"
#include "STCondition_HasObjective.generated.h"

class UObjective;

/**
 * State Tree Condition: Has Objective
 *
 * Simple condition that checks if the follower has an active objective.
 * Used to control transitions between ExecuteObjective and Idle states.
 *
 * Returns true when: CurrentObjective != null && bHasActiveObjective == true
 */

USTRUCT()
struct GAMEAI_PROJECT_API FSTCondition_HasObjectiveInstanceData
{
	GENERATED_BODY()

	/** Current objective (bind to FollowerContext.CurrentObjective) */
	UPROPERTY(EditAnywhere, Category = "Input")
	TObjectPtr<UObjective> CurrentObjective = nullptr;

	/** Has active objective (bind to FollowerContext.bHasActiveObjective) */
	UPROPERTY(EditAnywhere, Category = "Input")
	bool bHasActiveObjective = false;

	/** If true, inverts the condition (true when NO objective) */
	UPROPERTY(EditAnywhere, Category = "Parameter")
	bool bInvertCondition = false;
};

USTRUCT(meta = (DisplayName = "Has Objective"))
struct GAMEAI_PROJECT_API FSTCondition_HasObjective : public FStateTreeConditionBase
{
	GENERATED_BODY()

	using FInstanceDataType = FSTCondition_HasObjectiveInstanceData;

	FSTCondition_HasObjective() = default;

	virtual const UStruct* GetInstanceDataType() const override { return FInstanceDataType::StaticStruct(); }
	virtual bool TestCondition(FStateTreeExecutionContext& Context) const override;
};
