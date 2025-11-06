// Copyright Epic Games, Inc. All Rights Reserved.

#include "StateTree/Evaluators/STEvaluator_SyncCommand.h"
#include "StateTree/FollowerStateTreeContext.h"
#include "Team/FollowerAgentComponent.h"

void FSTEvaluator_SyncCommand::TreeStart(FStateTreeExecutionContext& Context) const
{
	FInstanceDataType& InstanceData = Context.GetInstanceData(*this);

	// Initialize last command type
	if (InstanceData.FollowerComponent)
	{
		InstanceData.LastCommandType = InstanceData.FollowerComponent->GetCurrentCommand().CommandType;
	}
}

void FSTEvaluator_SyncCommand::Tick(FStateTreeExecutionContext& Context, float DeltaTime) const
{
	FInstanceDataType& InstanceData = Context.GetInstanceData(*this);

	if (!InstanceData.FollowerComponent)
	{
		return;
	}

	// Get current command from follower component
	FStrategicCommand NewCommand = InstanceData.FollowerComponent->GetCurrentCommand();

	// Update outputs
	InstanceData.CurrentCommand = NewCommand;
	InstanceData.bIsCommandValid = InstanceData.FollowerComponent->IsCommandValid();
	InstanceData.TimeSinceCommand = InstanceData.FollowerComponent->GetTimeSinceLastCommand();

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
