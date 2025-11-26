// Copyright Epic Games, Inc. All Rights Reserved.

#pragma once

#include "CoreMinimal.h"
#include "StateTreeEvaluatorBase.h"
#include "StateTree/FollowerStateTreeSchema.h"
#include "Team/Objective.h"
#include "STEvaluator_SyncObjective.generated.h"

/**
 * State Tree Evaluator: Sync Objective
 *
 * Syncs assigned objective from FollowerAgentComponent to State Tree context.
 * Runs every tick to detect objective changes from team leader.
 *
 * When objective changes, this can trigger state transitions via conditions.
 *
 * Replaces: STEvaluator_SyncCommand (v2.0)
 */


USTRUCT()
struct GAMEAI_PROJECT_API FSTEvaluator_SyncObjectiveInstanceData
{
	GENERATED_BODY()

	//--------------------------------------------------------------------------
	// CONTEXT BINDING (UE 5.6 - auto-binds to FollowerContext from schema)
	//--------------------------------------------------------------------------

	/**
	 * Shared context struct - automatically bound by StateTree
	 * Contains all agent state, objectives, and components
	 */
	UPROPERTY(EditAnywhere, Category = "Context")
	FFollowerStateTreeContext Context;

	//--------------------------------------------------------------------------
	// CONFIGURATION
	//--------------------------------------------------------------------------

	/** Log objective changes */
	UPROPERTY(EditAnywhere, Category = "Parameter")
	bool bLogObjectiveChanges = true;

	//--------------------------------------------------------------------------
	// RUNTIME STATE
	//--------------------------------------------------------------------------

	/** Last objective type (for change detection) */
	UPROPERTY()
	EObjectiveType LastObjectiveType = EObjectiveType::Eliminate;

	/** Last objective pointer (for change detection) */
	UPROPERTY()
	TObjectPtr<UObjective> LastObjective = nullptr;
};

USTRUCT(meta = (DisplayName = "Sync Objective", BlueprintType))
struct GAMEAI_PROJECT_API FSTEvaluator_SyncObjective : public FStateTreeEvaluatorBase
{
	GENERATED_BODY()

	using FInstanceDataType = FSTEvaluator_SyncObjectiveInstanceData;

	FSTEvaluator_SyncObjective() = default;

	virtual const UStruct* GetInstanceDataType() const override { return FInstanceDataType::StaticStruct(); }

	virtual void TreeStart(FStateTreeExecutionContext& Context) const override;
	virtual void Tick(FStateTreeExecutionContext& Context, float DeltaTime) const override;
	virtual void TreeStop(FStateTreeExecutionContext& Context) const override;
};
