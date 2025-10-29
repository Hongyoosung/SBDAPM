// Copyright Epic Games, Inc. All Rights Reserved.

#include "BehaviorTree/Decorators/BTDecorator_CheckTacticalAction.h"
#include "BehaviorTree/BlackboardComponent.h"
#include "AIController.h"
#include "GameFramework/Pawn.h"
#include "Team/FollowerAgentComponent.h"

UBTDecorator_CheckTacticalAction::UBTDecorator_CheckTacticalAction()
{
	NodeName = "Check Tactical Action";
	bNotifyBecomeRelevant = true;
	bNotifyTick = false;

	// Enable observer aborts for reactivity when action changes
	FlowAbortMode = EBTFlowAbortMode::None;
}

bool UBTDecorator_CheckTacticalAction::CalculateRawConditionValue(UBehaviorTreeComponent& OwnerComp, uint8* NodeMemory) const
{
	// Get current tactical action
	ETacticalAction CurrentAction;

	if (bReadDirectlyFromComponent)
	{
		CurrentAction = GetTacticalActionFromComponent(OwnerComp);
	}
	else
	{
		CurrentAction = GetTacticalActionFromBlackboard(OwnerComp);
	}

	// Check if current action matches any accepted actions
	bool bMatches = AcceptedActions.Contains(CurrentAction);

	// Apply inversion if needed
	bool bResult = bInvertCondition ? !bMatches : bMatches;

	// Log check
	if (bLogChecks)
	{
		FString CurrentActionName = UEnum::GetValueAsString(CurrentAction);
		FString ResultStr = bResult ? TEXT("PASS") : TEXT("FAIL");
		UE_LOG(LogTemp, Verbose, TEXT("BTDecorator_CheckTacticalAction: Current action '%s' - %s"),
			*CurrentActionName, *ResultStr);
	}

	return bResult;
}

FString UBTDecorator_CheckTacticalAction::GetStaticDescription() const
{
	FString ActionsStr;

	if (AcceptedActions.Num() == 0)
	{
		ActionsStr = TEXT("None");
	}
	else if (AcceptedActions.Num() == 1)
	{
		ActionsStr = UEnum::GetValueAsString(AcceptedActions[0]);
	}
	else
	{
		TArray<FString> ActionNames;
		for (ETacticalAction Action : AcceptedActions)
		{
			ActionNames.Add(UEnum::GetValueAsString(Action));
		}
		ActionsStr = FString::Join(ActionNames, TEXT(" OR "));
	}

	FString InvertStr = bInvertCondition ? TEXT("NOT ") : TEXT("");
	FString SourceStr = bReadDirectlyFromComponent ? TEXT(" (Direct)") : TEXT(" (BB)");

	return FString::Printf(TEXT("%s%s%s"), *InvertStr, *ActionsStr, *SourceStr);
}

ETacticalAction UBTDecorator_CheckTacticalAction::GetTacticalActionFromBlackboard(UBehaviorTreeComponent& OwnerComp) const
{
	// Get blackboard component
	UBlackboardComponent* BlackboardComp = OwnerComp.GetBlackboardComponent();
	if (!BlackboardComp)
	{
		return ETacticalAction::DefensiveHold; // Default fallback
	}

	// Check if key is set
	if (TacticalActionKey.SelectedKeyName == NAME_None)
	{
		if (bLogChecks)
		{
			UE_LOG(LogTemp, Warning, TEXT("BTDecorator_CheckTacticalAction: No blackboard key specified"));
		}
		return ETacticalAction::DefensiveHold;
	}

	// Get tactical action from blackboard
	uint8 ActionByte = BlackboardComp->GetValueAsEnum(TacticalActionKey.SelectedKeyName);
	return static_cast<ETacticalAction>(ActionByte);
}

ETacticalAction UBTDecorator_CheckTacticalAction::GetTacticalActionFromComponent(UBehaviorTreeComponent& OwnerComp) const
{
	// Get AI controller and pawn
	AAIController* AIController = OwnerComp.GetAIOwner();
	if (!AIController)
	{
		return ETacticalAction::DefensiveHold;
	}

	APawn* ControlledPawn = AIController->GetPawn();
	if (!ControlledPawn)
	{
		return ETacticalAction::DefensiveHold;
	}

	// Get FollowerAgentComponent
	UFollowerAgentComponent* FollowerComp = ControlledPawn->FindComponentByClass<UFollowerAgentComponent>();
	if (!FollowerComp)
	{
		if (bLogChecks)
		{
			UE_LOG(LogTemp, Warning, TEXT("BTDecorator_CheckTacticalAction: No FollowerAgentComponent found on %s"),
				*ControlledPawn->GetName());
		}
		return ETacticalAction::DefensiveHold;
	}

	// Get last tactical action from component
	return FollowerComp->LastTacticalAction;
}
