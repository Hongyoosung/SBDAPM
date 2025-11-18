// Copyright Epic Games, Inc. All Rights Reserved.

#include "StateTree/Tasks/STTask_QueryRLPolicy.h"
#include "StateTree/FollowerStateTreeContext.h"
#include "RL/RLPolicyNetwork.h"
#include "Team/FollowerAgentComponent.h"
#include "AIController.h"
#include "DrawDebugHelpers.h"

EStateTreeRunStatus FSTTask_QueryRLPolicy::EnterState(FStateTreeExecutionContext& Context, const FStateTreeTransitionResult& Transition) const
{
	FInstanceDataType& InstanceData = Context.GetInstanceData(*this);

	// Validate context components
	if (!InstanceData.Context.FollowerComponent)
	{
		UE_LOG(LogTemp, Error, TEXT("STTask_QueryRLPolicy: FollowerComponent is null in context"));
		return EStateTreeRunStatus::Failed;
	}

	// Query the policy
	ETacticalAction SelectedAction = QueryPolicy(Context);

	// Write selected action to context (so other tasks can read it)
	InstanceData.Context.CurrentTacticalAction = SelectedAction;
	InstanceData.Context.TimeInTacticalAction = 0.0f;

	// Log if enabled
	if (InstanceData.bLogActionSelection)
	{
		UE_LOG(LogTemp, Log, TEXT("STTask_QueryRLPolicy: Selected action '%s' for '%s'"),
			*UEnum::GetValueAsString(SelectedAction),
			*InstanceData.Context.FollowerComponent->GetOwner()->GetName());
	}

	// Draw debug if enabled
	if (InstanceData.bDrawDebugInfo && InstanceData.Context.FollowerComponent->GetOwner())
	{
		AActor* Owner = InstanceData.Context.FollowerComponent->GetOwner();
		FVector Location = Owner->GetActorLocation();
		FString ActionName = UEnum::GetValueAsString(SelectedAction);
		DrawDebugString(Owner->GetWorld(), Location + FVector(0, 0, 100),
			*ActionName, nullptr, FColor::Cyan, 2.0f, true);
	}

	// Task completes immediately - execution tasks handle their own RL query intervals
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
	if (InstanceData.bUseRLPolicy && InstanceData.Context.TacticalPolicy && InstanceData.Context.TacticalPolicy->IsReady())
	{
		// Query RL policy network using current observation from context
		ETacticalAction SelectedAction = InstanceData.Context.TacticalPolicy->SelectAction(InstanceData.Context.CurrentObservation);
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

	// Simple rule-based fallback based on command type from context
	switch (InstanceData.Context.CurrentCommand.CommandType)
	{
	case EStrategicCommandType::Assault:
		return InstanceData.Context.CurrentObservation.AgentHealth > 0.5f
			? ETacticalAction::AggressiveAssault
			: ETacticalAction::CautiousAdvance;

	case EStrategicCommandType::Defend:
		return InstanceData.Context.bInCover
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
