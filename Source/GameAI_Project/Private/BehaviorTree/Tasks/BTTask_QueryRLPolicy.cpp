// Copyright Epic Games, Inc. All Rights Reserved.

#include "BehaviorTree/Tasks/BTTask_QueryRLPolicy.h"
#include "Team/FollowerAgentComponent.h"
#include "RL/RLPolicyNetwork.h"
#include "BehaviorTree/BlackboardComponent.h"
#include "BehaviorTree/BehaviorTreeComponent.h"
#include "AIController.h"
#include "DrawDebugHelpers.h"

UBTTask_QueryRLPolicy::UBTTask_QueryRLPolicy()
{
	NodeName = "Query RL Policy";
	bNotifyTick = false;

	// Set default blackboard keys
	TacticalActionKey.SelectedKeyName = "TacticalAction";
	TacticalActionNameKey.SelectedKeyName = "TacticalActionName";
}

EBTNodeResult::Type UBTTask_QueryRLPolicy::ExecuteTask(UBehaviorTreeComponent& OwnerComp, uint8* NodeMemory)
{
	// Get follower component
	UFollowerAgentComponent* FollowerComp = GetFollowerComponent(OwnerComp);
	if (!FollowerComp)
	{
		UE_LOG(LogTemp, Error, TEXT("BTTask_QueryRLPolicy: No FollowerAgentComponent found"));
		return EBTNodeResult::Failed;
	}

	// Query RL policy for tactical action
	ETacticalAction SelectedAction = FollowerComp->QueryRLPolicy();

	// Get blackboard
	UBlackboardComponent* Blackboard = OwnerComp.GetBlackboardComponent();
	if (!Blackboard)
	{
		UE_LOG(LogTemp, Error, TEXT("BTTask_QueryRLPolicy: No Blackboard found"));
		return EBTNodeResult::Failed;
	}

	// Update blackboard with selected action
	if (TacticalActionKey.SelectedKeyType == UBlackboardKeyType_Enum::StaticClass())
	{
		Blackboard->SetValueAsEnum(TacticalActionKey.SelectedKeyName, static_cast<uint8>(SelectedAction));
	}

	// Update blackboard with action name (optional)
	if (TacticalActionNameKey.SelectedKeyType == UBlackboardKeyType_String::StaticClass())
	{
		FString ActionName = URLPolicyNetwork::GetActionName(SelectedAction);
		Blackboard->SetValueAsString(TacticalActionNameKey.SelectedKeyName, ActionName);
	}

	// Log action selection
	if (bLogActionSelection)
	{
		UE_LOG(LogTemp, Log, TEXT("BTTask_QueryRLPolicy: Selected action '%s' for %s"),
			*URLPolicyNetwork::GetActionName(SelectedAction),
			*FollowerComp->GetOwner()->GetName());
	}

	// Draw debug
	if (bDrawDebugInfo)
	{
		DrawDebugActionSelection(OwnerComp, SelectedAction);
	}

	return EBTNodeResult::Succeeded;
}

FString UBTTask_QueryRLPolicy::GetStaticDescription() const
{
	return FString::Printf(TEXT("Query RL policy and store result in '%s'"),
		*TacticalActionKey.SelectedKeyName.ToString());
}

UFollowerAgentComponent* UBTTask_QueryRLPolicy::GetFollowerComponent(UBehaviorTreeComponent& OwnerComp) const
{
	AAIController* AIController = OwnerComp.GetAIOwner();
	if (!AIController)
	{
		return nullptr;
	}

	APawn* ControlledPawn = AIController->GetPawn();
	if (!ControlledPawn)
	{
		return nullptr;
	}

	return ControlledPawn->FindComponentByClass<UFollowerAgentComponent>();
}

void UBTTask_QueryRLPolicy::DrawDebugActionSelection(UBehaviorTreeComponent& OwnerComp, ETacticalAction SelectedAction) const
{
	AAIController* AIController = OwnerComp.GetAIOwner();
	if (!AIController || !AIController->GetPawn())
	{
		return;
	}

	UWorld* World = AIController->GetWorld();
	if (!World)
	{
		return;
	}

	FVector PawnLocation = AIController->GetPawn()->GetActorLocation();
	FString ActionText = FString::Printf(TEXT("RL Action:\n%s"),
		*URLPolicyNetwork::GetActionName(SelectedAction));

	DrawDebugString(World, PawnLocation + FVector(0, 0, 150), ActionText, nullptr, FColor::Green, 2.0f, true);
}
