// Copyright Epic Games, Inc. All Rights Reserved.

#include "StateTree/Tasks/STTask_QueryRLPolicy.h"
#include "StateTree/FollowerStateTreeSchema.h"
#include "RL/RLPolicyNetwork.h"
#include "Team/FollowerAgentComponent.h"
#include "AIController.h"
#include "DrawDebugHelpers.h"

EStateTreeRunStatus FSTTask_QueryRLPolicy::EnterState(FStateTreeExecutionContext& Context, const FStateTreeTransitionResult& Transition) const
{
	FInstanceDataType& InstanceData = Context.GetInstanceData(*this);

	// Validate inputs
	if (!InstanceData.FollowerComponent)
	{
		UE_LOG(LogTemp, Error, TEXT("STTask_QueryRLPolicy: FollowerComponent is null"));
		return EStateTreeRunStatus::Failed;
	}

	// Query the policy
	ETacticalAction SelectedAction = QueryPolicy(Context);

	// Output selected action
	InstanceData.SelectedAction = SelectedAction;

	// Log if enabled
	if (InstanceData.bLogActionSelection)
	{
		UE_LOG(LogTemp, Log, TEXT("STTask_QueryRLPolicy: Selected action '%s' for '%s'"),
			*UEnum::GetValueAsString(SelectedAction),
			*InstanceData.FollowerComponent->GetOwner()->GetName());
	}

	// Draw debug if enabled
	if (InstanceData.bDrawDebugInfo && InstanceData.FollowerComponent->GetOwner())
	{
		AActor* Owner = InstanceData.FollowerComponent->GetOwner();
		FVector Location = Owner->GetActorLocation();
		FString ActionName = UEnum::GetValueAsString(SelectedAction);
		DrawDebugString(Owner->GetWorld(), Location + FVector(0, 0, 100),
			*ActionName, nullptr, FColor::Cyan, 2.0f, true);
	}

	// Task completes immediately
	return EStateTreeRunStatus::Succeeded;
}

void FSTTask_QueryRLPolicy::ExitState(FStateTreeExecutionContext& Context, const FStateTreeTransitionResult& Transition) const
{
	// No cleanup needed
}

ETacticalAction FSTTask_QueryRLPolicy::QueryPolicy(FStateTreeExecutionContext& Context) const
{
	FInstanceDataType& InstanceData = Context.GetInstanceData(*this);

	// Check if using RL policy
	if (InstanceData.bUseRLPolicy && InstanceData.TacticalPolicy && InstanceData.TacticalPolicy->IsReady())
	{
		// Query RL policy network using public API
		ETacticalAction SelectedAction = InstanceData.TacticalPolicy->SelectAction(InstanceData.CurrentObservation);
		return SelectedAction;
	}
	else
	{
		// Fallback to rule-based
		return GetFallbackAction(Context);
	}
}

ETacticalAction FSTTask_QueryRLPolicy::GetFallbackAction(FStateTreeExecutionContext& Context) const
{
	FInstanceDataType& InstanceData = Context.GetInstanceData(*this);

	// Simple rule-based fallback based on command type
	switch (InstanceData.CurrentCommand.CommandType)
	{
	case EStrategicCommandType::Assault:
		return InstanceData.CurrentObservation.AgentHealth > 0.5f
			? ETacticalAction::AggressiveAssault
			: ETacticalAction::CautiousAdvance;

	case EStrategicCommandType::Defend:
		return InstanceData.bInCover
			? ETacticalAction::DefensiveHold
			: ETacticalAction::SeekCover;

	case EStrategicCommandType::Support:
		return ETacticalAction::ProvideCoveringFire;

	case EStrategicCommandType::Retreat:
		return ETacticalAction::TacticalRetreat;

	case EStrategicCommandType::MoveTo:
		return ETacticalAction::Sprint;

	default:
		return ETacticalAction::DefensiveHold;
	}
}
