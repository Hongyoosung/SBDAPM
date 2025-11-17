// Copyright Epic Games, Inc. All Rights Reserved.

#include "StateTree/Evaluators/STEvaluator_SyncCommand.h"
#include "StateTree/FollowerStateTreeContext.h"
#include "Team/FollowerAgentComponent.h"

void FSTEvaluator_SyncCommand::TreeStart(FStateTreeExecutionContext& Context) const
{
	FInstanceDataType& InstanceData = Context.GetInstanceData(*this);

	// Initialize last command type from context's follower component
	if (InstanceData.Context.FollowerComponent)
	{
		InstanceData.LastCommandType = InstanceData.Context.FollowerComponent->GetCurrentCommand().CommandType;
	}
}

void FSTEvaluator_SyncCommand::Tick(FStateTreeExecutionContext& Context, float DeltaTime) const
{
	FInstanceDataType& InstanceData = Context.GetInstanceData(*this);

	if (!InstanceData.Context.FollowerComponent)
	{
		return;
	}

	// Get current command from follower component
	FStrategicCommand NewCommand = InstanceData.Context.FollowerComponent->GetCurrentCommand();

	// Update context outputs (shared with all tasks/evaluators)
	InstanceData.Context.CurrentCommand = NewCommand;
	InstanceData.Context.bIsCommandValid = InstanceData.Context.FollowerComponent->IsCommandValid();
	InstanceData.Context.TimeSinceCommand = InstanceData.Context.FollowerComponent->GetTimeSinceLastCommand();

	// Log command changes
	if (InstanceData.bLogCommandChanges && NewCommand.CommandType != InstanceData.LastCommandType)
	{
		UE_LOG(LogTemp, Log, TEXT("STEvaluator_SyncCommand: Command changed from '%s' to '%s'"),
			*UEnum::GetValueAsString(InstanceData.LastCommandType),
			*UEnum::GetValueAsString(NewCommand.CommandType));

		InstanceData.LastCommandType = NewCommand.CommandType;
	}
}

void FSTEvaluator_SyncCommand::TreeStop(FStateTreeExecutionContext& Context) const
{
	// No cleanup needed
}
