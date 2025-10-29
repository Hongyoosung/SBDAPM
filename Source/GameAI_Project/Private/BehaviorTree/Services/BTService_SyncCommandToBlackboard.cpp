// Copyright Epic Games, Inc. All Rights Reserved.

#include "BehaviorTree/Services/BTService_SyncCommandToBlackboard.h"
#include "BehaviorTree/BlackboardComponent.h"
#include "AIController.h"
#include "GameFramework/Pawn.h"
#include "Team/FollowerAgentComponent.h"
#include "Team/TeamTypes.h"

UBTService_SyncCommandToBlackboard::UBTService_SyncCommandToBlackboard()
{
	NodeName = "Sync Command to Blackboard";
	Interval = 0.5f;  // Update every 0.5 seconds
	RandomDeviation = 0.1f;
	bNotifyBecomeRelevant = true;
}

void UBTService_SyncCommandToBlackboard::TickNode(UBehaviorTreeComponent& OwnerComp, uint8* NodeMemory, float DeltaSeconds)
{
	Super::TickNode(OwnerComp, NodeMemory, DeltaSeconds);

	// Get AI controller and pawn
	AAIController* AIController = OwnerComp.GetAIOwner();
	if (!AIController)
	{
		return;
	}

	APawn* ControlledPawn = AIController->GetPawn();
	if (!ControlledPawn)
	{
		return;
	}

	// Get blackboard component
	UBlackboardComponent* BlackboardComp = OwnerComp.GetBlackboardComponent();
	if (!BlackboardComp)
	{
		return;
	}

	// Get FollowerAgentComponent
	UFollowerAgentComponent* FollowerComp = ControlledPawn->FindComponentByClass<UFollowerAgentComponent>();
	if (!FollowerComp)
	{
		if (bClearOnNoFollowerComponent)
		{
			// Clear command data from blackboard
			if (IsCommandValidKey.SelectedKeyName != NAME_None)
			{
				BlackboardComp->SetValueAsBool(IsCommandValidKey.SelectedKeyName, false);
			}

			if (bLogSync)
			{
				UE_LOG(LogTemp, Warning, TEXT("BTService_SyncCommandToBlackboard: No FollowerAgentComponent found on %s"),
					*ControlledPawn->GetName());
			}
		}
		return;
	}

	// Get current command
	FStrategicCommand CurrentCommand = FollowerComp->GetCurrentCommand();
	bool bHasValidCommand = FollowerComp->HasActiveCommand();

	// Sync command type
	if (CommandTypeKey.SelectedKeyName != NAME_None)
	{
		uint8 CommandTypeByte = static_cast<uint8>(CurrentCommand.CommandType);
		BlackboardComp->SetValueAsEnum(CommandTypeKey.SelectedKeyName, CommandTypeByte);
	}

	// Sync command target
	if (CommandTargetKey.SelectedKeyName != NAME_None)
	{
		BlackboardComp->SetValueAsObject(CommandTargetKey.SelectedKeyName, CurrentCommand.TargetActor);
	}

	// Sync command priority
	if (CommandPriorityKey.SelectedKeyName != NAME_None)
	{
		BlackboardComp->SetValueAsInt(CommandPriorityKey.SelectedKeyName, CurrentCommand.Priority);
	}

	// Sync time since command
	if (TimeSinceCommandKey.SelectedKeyName != NAME_None)
	{
		float TimeSinceCommand = FollowerComp->GetTimeSinceLastCommand();
		BlackboardComp->SetValueAsFloat(TimeSinceCommandKey.SelectedKeyName, TimeSinceCommand);
	}

	// Sync command validity
	if (IsCommandValidKey.SelectedKeyName != NAME_None)
	{
		BlackboardComp->SetValueAsBool(IsCommandValidKey.SelectedKeyName, bHasValidCommand);
	}

	// Log sync
	if (bLogSync)
	{
		FString CommandTypeName = UEnum::GetValueAsString(CurrentCommand.CommandType);
		FString TargetName = CurrentCommand.TargetActor ? CurrentCommand.TargetActor->GetName() : TEXT("None");

		UE_LOG(LogTemp, Verbose, TEXT("BTService_SyncCommandToBlackboard: Synced command %s (Target: %s, Valid: %s)"),
			*CommandTypeName, *TargetName, bHasValidCommand ? TEXT("Yes") : TEXT("No"));
	}
}

FString UBTService_SyncCommandToBlackboard::GetStaticDescription() const
{
	TArray<FString> Keys;

	if (CommandTypeKey.SelectedKeyName != NAME_None)
		Keys.Add(FString::Printf(TEXT("Type->%s"), *CommandTypeKey.SelectedKeyName.ToString()));

	if (CommandTargetKey.SelectedKeyName != NAME_None)
		Keys.Add(FString::Printf(TEXT("Target->%s"), *CommandTargetKey.SelectedKeyName.ToString()));

	if (TimeSinceCommandKey.SelectedKeyName != NAME_None)
		Keys.Add(FString::Printf(TEXT("Time->%s"), *TimeSinceCommandKey.SelectedKeyName.ToString()));

	if (IsCommandValidKey.SelectedKeyName != NAME_None)
		Keys.Add(FString::Printf(TEXT("Valid->%s"), *IsCommandValidKey.SelectedKeyName.ToString()));

	FString KeysString = FString::Join(Keys, TEXT(", "));

	return FString::Printf(TEXT("Sync command: %s"), *KeysString);
}
