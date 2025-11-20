// Copyright Epic Games, Inc. All Rights Reserved.

#include "StateTree/Evaluators/STEvaluator_SyncCommand.h"
#include "StateTree/FollowerStateTreeContext.h"
#include "Team/FollowerAgentComponent.h"

void FSTEvaluator_SyncCommand::TreeStart(FStateTreeExecutionContext& Context) const
{
	FInstanceDataType& InstanceData = Context.GetInstanceData(*this);

	// Initialize context outputs from follower component
	// CRITICAL: This must happen in TreeStart so conditions can evaluate during initial state selection
	if (InstanceData.Context.FollowerComponent)
	{
		FStrategicCommand CurrentCommand = InstanceData.Context.FollowerComponent->GetCurrentCommand();
		bool bCommandValid = InstanceData.Context.FollowerComponent->IsCommandValid();
		float TimeSinceCommand = InstanceData.Context.FollowerComponent->GetTimeSinceLastCommand();

		// Set context outputs (shared with conditions/tasks)
		InstanceData.Context.CurrentCommand = CurrentCommand;
		InstanceData.Context.bIsCommandValid = bCommandValid;
		InstanceData.Context.TimeSinceCommand = TimeSinceCommand;

		// Sync PrimaryTarget from command
		InstanceData.Context.PrimaryTarget = CurrentCommand.TargetActor;

		// Initialize tracking variable
		InstanceData.LastCommandType = CurrentCommand.CommandType;

		UE_LOG(LogTemp, Warning, TEXT("[SYNC COMMAND] TreeStart: Initialized context - Type=%s, Valid=%d, Time=%.2f"),
			*UEnum::GetValueAsString(CurrentCommand.CommandType),
			bCommandValid,
			TimeSinceCommand);
	}
	else
	{
		UE_LOG(LogTemp, Error, TEXT("[SYNC COMMAND] TreeStart: ❌ FollowerComponent is null!"));
	}
}

void FSTEvaluator_SyncCommand::Tick(FStateTreeExecutionContext& Context, float DeltaTime) const
{
	FInstanceDataType& InstanceData = Context.GetInstanceData(*this);

	if (!InstanceData.Context.FollowerComponent)
	{
		UE_LOG(LogTemp, Error, TEXT("[SYNC COMMAND] ❌ FollowerComponent is null!"));
		return;
	}

	// Get current command from follower component
	FStrategicCommand NewCommand = InstanceData.Context.FollowerComponent->GetCurrentCommand();
	bool bNewCommandValid = InstanceData.Context.FollowerComponent->IsCommandValid();
	float NewTimeSinceCommand = InstanceData.Context.FollowerComponent->GetTimeSinceLastCommand();

	// Update context outputs (shared with all tasks/evaluators)
	InstanceData.Context.CurrentCommand = NewCommand;
	InstanceData.Context.bIsCommandValid = bNewCommandValid;
	InstanceData.Context.TimeSinceCommand = NewTimeSinceCommand;


	// Log command changes (Warning level for visibility)
	if (NewCommand.CommandType != InstanceData.LastCommandType)
	{
		FString TargetInfo = NewCommand.TargetActor ?
			FString::Printf(TEXT("Target: %s"), *NewCommand.TargetActor->GetName()) :
			TEXT("Target: None");

		UE_LOG(LogTemp, Warning, TEXT("[SYNC COMMAND] 📝 Command changed: '%s' → '%s', Valid=%d, %s"),
			*UEnum::GetValueAsString(InstanceData.LastCommandType),
			*UEnum::GetValueAsString(NewCommand.CommandType),
			bNewCommandValid,
			*TargetInfo);

		InstanceData.LastCommandType = NewCommand.CommandType;
	}

	// Periodic verbose logging (every 60 ticks)
	static int32 TickCounter = 0;
	if (++TickCounter % 60 == 0)
	{
		UE_LOG(LogTemp, Verbose, TEXT("[SYNC COMMAND] Current: Type=%s, Valid=%d, Time=%.2f"),
			*UEnum::GetValueAsString(NewCommand.CommandType),
			bNewCommandValid,
			NewTimeSinceCommand);
	}
}

void FSTEvaluator_SyncCommand::TreeStop(FStateTreeExecutionContext& Context) const
{
	// No cleanup needed
}
