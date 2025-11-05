// Copyright Epic Games, Inc. All Rights Reserved.

#pragma once

#include "CoreMinimal.h"
#include "StateTreeEvaluatorBase.h"
#include "StateTree/FollowerStateTreeSchema.h"
#include "Team/TeamTypes.h"
#include "STEvaluator_SyncCommand.generated.h"

/**
 * State Tree Evaluator: Sync Command
 *
 * Syncs strategic command from FollowerAgentComponent to State Tree context.
 * Runs every tick to detect command changes from team leader.
 *
 * When command changes, this can trigger state transitions via conditions.
 *
 * Replaces: BTService_SyncCommandToBlackboard
 */


USTRUCT()
struct GAMEAI_PROJECT_API FSTEvaluator_SyncCommandInstanceData
{
	GENERATED_BODY()

	/** Follower component (bound from context) */
	UPROPERTY(EditAnywhere, Category = "Input")
	TObjectPtr<UFollowerAgentComponent> FollowerComponent = nullptr;

	/** Current command (output - bound to context) */
	UPROPERTY(EditAnywhere, Category = "Output")
	FStrategicCommand CurrentCommand;

	/** Is command valid (output - bound to context) */
	UPROPERTY(EditAnywhere, Category = "Output")
	bool bIsCommandValid = false;

	/** Time since command (output - bound to context) */
	UPROPERTY(EditAnywhere, Category = "Output")
	float TimeSinceCommand = 0.0f;

	/** Log command changes */
	UPROPERTY(EditAnywhere, Category = "Debug")
	bool bLogCommandChanges = true;

	/** Last command type (for change detection) */
	UPROPERTY()
	EStrategicCommandType LastCommandType = EStrategicCommandType::None;
};

USTRUCT(meta = (DisplayName = "Sync Command"))
struct GAMEAI_PROJECT_API FSTEvaluator_SyncCommand : public FStateTreeEvaluatorBase
{
	GENERATED_BODY()

	using FInstanceDataType = FSTEvaluator_SyncCommandInstanceData;

	FSTEvaluator_SyncCommand() = default;

	virtual const UStruct* GetInstanceDataType() const override { return FInstanceDataType::StaticStruct(); }

	virtual void TreeStart(FStateTreeExecutionContext& Context) const override;
	virtual void Tick(FStateTreeExecutionContext& Context, float DeltaTime) const override;
	virtual void TreeStop(FStateTreeExecutionContext& Context) const override;
};