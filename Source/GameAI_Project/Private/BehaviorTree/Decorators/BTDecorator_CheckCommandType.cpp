// Copyright Epic Games, Inc. All Rights Reserved.

#include "BehaviorTree/Decorators/BTDecorator_CheckCommandType.h"
#include "BehaviorTree/BlackboardComponent.h"
#include "AIController.h"
#include "GameFramework/Pawn.h"
#include "Team/FollowerAgentComponent.h"

UBTDecorator_CheckCommandType::UBTDecorator_CheckCommandType()
{
	NodeName = "Check Command Type";
	bNotifyBecomeRelevant = true;
	bNotifyTick = false;

	// Enable observer aborts for reactivity
	FlowAbortMode = EBTFlowAbortMode::None;
}

bool UBTDecorator_CheckCommandType::CalculateRawConditionValue(UBehaviorTreeComponent& OwnerComp, uint8* NodeMemory) const
{
	// Get current command type
	EStrategicCommandType CurrentCommandType;

	if (bUseBlackboard)
	{
		CurrentCommandType = GetCommandTypeFromBlackboard(OwnerComp);
	}
	else
	{
		CurrentCommandType = GetCommandTypeFromComponent(OwnerComp);
	}

	// Check if command is valid (if required)
	if (bRequireValidCommand)
	{
		bool bIsValid = IsCommandValid(OwnerComp);
		if (!bIsValid)
		{
			return bInvertCondition; // Return inverted result if command invalid
		}
	}

	// Check if current command matches any accepted types
	bool bMatches = AcceptedCommandTypes.Contains(CurrentCommandType);

	// Apply inversion if needed
	return bInvertCondition ? !bMatches : bMatches;
}

FString UBTDecorator_CheckCommandType::GetStaticDescription() const
{
	FString CommandTypesStr;

	if (AcceptedCommandTypes.Num() == 0)
	{
		CommandTypesStr = TEXT("None");
	}
	else if (AcceptedCommandTypes.Num() == 1)
	{
		CommandTypesStr = UEnum::GetValueAsString(AcceptedCommandTypes[0]);
	}
	else
	{
		TArray<FString> TypeNames;
		for (EStrategicCommandType CommandType : AcceptedCommandTypes)
		{
			TypeNames.Add(UEnum::GetValueAsString(CommandType));
		}
		CommandTypesStr = FString::Join(TypeNames, TEXT(" OR "));
	}

	FString InvertStr = bInvertCondition ? TEXT("NOT ") : TEXT("");
	FString SourceStr = bUseBlackboard ? TEXT(" (BB)") : TEXT(" (Direct)");

	return FString::Printf(TEXT("%s%s%s"), *InvertStr, *CommandTypesStr, *SourceStr);
}

EStrategicCommandType UBTDecorator_CheckCommandType::GetCommandTypeFromComponent(UBehaviorTreeComponent& OwnerComp) const
{
	// Get AI controller and pawn
	AAIController* AIController = OwnerComp.GetAIOwner();
	if (!AIController)
	{
		return EStrategicCommandType::None;
	}

	APawn* ControlledPawn = AIController->GetPawn();
	if (!ControlledPawn)
	{
		return EStrategicCommandType::None;
	}

	// Get FollowerAgentComponent
	UFollowerAgentComponent* FollowerComp = ControlledPawn->FindComponentByClass<UFollowerAgentComponent>();
	if (!FollowerComp)
	{
		return EStrategicCommandType::None;
	}

	// Get current command type
	FStrategicCommand CurrentCommand = FollowerComp->GetCurrentCommand();
	return CurrentCommand.CommandType;
}

EStrategicCommandType UBTDecorator_CheckCommandType::GetCommandTypeFromBlackboard(UBehaviorTreeComponent& OwnerComp) const
{
	// Get blackboard component
	UBlackboardComponent* BlackboardComp = OwnerComp.GetBlackboardComponent();
	if (!BlackboardComp)
	{
		return EStrategicCommandType::None;
	}

	// Check if key is set
	if (CommandTypeKey.SelectedKeyName == NAME_None)
	{
		return EStrategicCommandType::None;
	}

	// Get command type from blackboard
	uint8 CommandTypeByte = BlackboardComp->GetValueAsEnum(CommandTypeKey.SelectedKeyName);
	return static_cast<EStrategicCommandType>(CommandTypeByte);
}

bool UBTDecorator_CheckCommandType::IsCommandValid(UBehaviorTreeComponent& OwnerComp) const
{
	if (bUseBlackboard)
	{
		// Check validity from blackboard key
		if (IsCommandValidKey.SelectedKeyName != NAME_None)
		{
			UBlackboardComponent* BlackboardComp = OwnerComp.GetBlackboardComponent();
			if (BlackboardComp)
			{
				return BlackboardComp->GetValueAsBool(IsCommandValidKey.SelectedKeyName);
			}
		}

		// If no validity key specified, assume valid if command type is not None
		EStrategicCommandType CommandType = GetCommandTypeFromBlackboard(OwnerComp);
		return CommandType != EStrategicCommandType::None;
	}
	else
	{
		// Check validity directly from FollowerAgentComponent
		AAIController* AIController = OwnerComp.GetAIOwner();
		if (!AIController)
		{
			return false;
		}

		APawn* ControlledPawn = AIController->GetPawn();
		if (!ControlledPawn)
		{
			return false;
		}

		UFollowerAgentComponent* FollowerComp = ControlledPawn->FindComponentByClass<UFollowerAgentComponent>();
		if (!FollowerComp)
		{
			return false;
		}

		return FollowerComp->HasActiveCommand();
	}
}
