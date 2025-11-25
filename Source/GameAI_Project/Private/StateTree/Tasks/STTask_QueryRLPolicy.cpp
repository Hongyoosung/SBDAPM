// Copyright Epic Games, Inc. All Rights Reserved.

#include "StateTree/Tasks/STTask_QueryRLPolicy.h"
#include "StateTree/FollowerStateTreeContext.h"
#include "StateTree/FollowerStateTreeComponent.h"
#include "RL/RLPolicyNetwork.h"
#include "Team/FollowerAgentComponent.h"
#include "AIController.h"
#include "DrawDebugHelpers.h"

EStateTreeRunStatus FSTTask_QueryRLPolicy::EnterState(FStateTreeExecutionContext& Context, const FStateTreeTransitionResult& Transition) const
{
	FInstanceDataType& InstanceData = Context.GetInstanceData(*this);

	if (!InstanceData.StateTreeComp)
	{
		UE_LOG(LogTemp, Error, TEXT("STTask_QueryRLPolicy: StateTreeComp is null!"));
		return EStateTreeRunStatus::Failed;
	}

	FFollowerStateTreeContext& SharedContext = InstanceData.StateTreeComp->GetSharedContext();

	// Reset timer state
	InstanceData.TimeSinceLastQuery = 0.0f;
	InstanceData.bHasQueriedOnce = false;

	// Validate context components
	if (!SharedContext.FollowerComponent)
	{
		UE_LOG(LogTemp, Error, TEXT("STTask_QueryRLPolicy: FollowerComponent is null in context"));
		return EStateTreeRunStatus::Failed;
	}

	// Query the policy immediately on entry
	FTacticalAction SelectedAction = QueryPolicy(Context);

	// Write selected action to SHARED context (so other tasks can read it)
	SharedContext.CurrentAtomicAction = SelectedAction;
	SharedContext.TimeInTacticalAction = 0.0f;
	InstanceData.bHasQueriedOnce = true;

	// ALWAYS log to diagnose binding issue
	UE_LOG(LogTemp, Warning, TEXT("🔍 [QUERY RL] '%s': WROTE action to SHARED context (Context addr: %p)"),
		*SharedContext.FollowerComponent->GetOwner()->GetName(),
		&SharedContext);

	// Log if enabled
	if (InstanceData.bLogActionSelection)
	{
		UE_LOG(LogTemp, Log, TEXT("STTask_QueryRLPolicy: Selected action for '%s' - Move(%.2f, %.2f) Look(%.2f, %.2f) Fire=%d"),
			*SharedContext.FollowerComponent->GetOwner()->GetName(),
			SelectedAction.MoveDirection.X, SelectedAction.MoveDirection.Y,
			SelectedAction.LookDirection.X, SelectedAction.LookDirection.Y,
			SelectedAction.bFire);
	}

	// Draw debug if enabled
	if (InstanceData.bDrawDebugInfo && SharedContext.FollowerComponent->GetOwner())
	{
		AActor* Owner = SharedContext.FollowerComponent->GetOwner();
		FVector Location = Owner->GetActorLocation();
		FString ActionInfo = FString::Printf(TEXT("Move(%.1f,%.1f) Fire:%d"),
			SelectedAction.MoveDirection.X, SelectedAction.MoveDirection.Y, SelectedAction.bFire);
		DrawDebugString(Owner->GetWorld(), Location + FVector(0, 0, 100),
			*ActionInfo, nullptr, FColor::Cyan, 2.0f, true);
	}

	// If interval is 0, complete immediately (one-shot query)
	if (InstanceData.QueryInterval <= 0.0f)
	{
		return EStateTreeRunStatus::Succeeded;
	}

	// Otherwise keep running to query at intervals
	return EStateTreeRunStatus::Running;
}

EStateTreeRunStatus FSTTask_QueryRLPolicy::Tick(FStateTreeExecutionContext& Context, const float DeltaTime) const
{
	FInstanceDataType& InstanceData = Context.GetInstanceData(*this);
	FFollowerStateTreeContext& SharedContext = InstanceData.StateTreeComp->GetSharedContext();

	// Accumulate time
	InstanceData.TimeSinceLastQuery += DeltaTime;

	// Check if it's time to query again
	if (InstanceData.TimeSinceLastQuery >= InstanceData.QueryInterval)
	{
		InstanceData.TimeSinceLastQuery = 0.0f;

		// Query the policy
		FTacticalAction SelectedAction = QueryPolicy(Context);

		// Update SHARED context
		SharedContext.CurrentTacticalAction = SelectedAction;
		// Note: Don't reset TimeInTacticalAction here - let execution tasks track that

		// Log if enabled
		if (InstanceData.bLogActionSelection)
		{
			UE_LOG(LogTemp, Log, TEXT("STTask_QueryRLPolicy: Re-queried action '%s' for '%s'"),
				*UEnum::GetValueAsString(SelectedAction),
				*SharedContext.FollowerComponent->GetOwner()->GetName());
		}

		// Draw debug if enabled
		if (InstanceData.bDrawDebugInfo && SharedContext.FollowerComponent->GetOwner())
		{
			AActor* Owner = SharedContext.FollowerComponent->GetOwner();
			FVector Location = Owner->GetActorLocation();
			FString ActionName = UEnum::GetValueAsString(SelectedAction);
			DrawDebugString(Owner->GetWorld(), Location + FVector(0, 0, 100),
				*ActionName, nullptr, FColor::Cyan, 2.0f, true);
		}
	}

	// Keep running to continue querying at intervals
	return EStateTreeRunStatus::Running;
}

void FSTTask_QueryRLPolicy::ExitState(FStateTreeExecutionContext& Context, const FStateTreeTransitionResult& Transition) const
{
	// No cleanup needed
}

FTacticalAction FSTTask_QueryRLPolicy::QueryPolicy(FStateTreeExecutionContext& Context) const
{
	FInstanceDataType& InstanceData = Context.GetInstanceData(*this);
	FFollowerStateTreeContext& SharedContext = InstanceData.StateTreeComp->GetSharedContext();

	// Check if using RL policy
	if (InstanceData.bUseRLPolicy && SharedContext.TacticalPolicy && SharedContext.TacticalPolicy->IsReady())
	{
		UE_LOG(LogTemp, Warning, TEXT("STTask_QueryRLPolicy: Querying RL policy for '%s'"),
			*SharedContext.FollowerComponent->GetOwner()->GetName());
		// Query RL policy network using current observation from context
		ETacticalAction SelectedAction = SharedContext.TacticalPolicy->SelectAction(SharedContext.CurrentObservation);
		return SelectedAction;
	}
	else
	{
		UE_LOG(LogTemp, Warning, TEXT("STTask_QueryRLPolicy: RL policy not used or not ready for '%s', falling back to rule-based action"),
			*SharedContext.FollowerComponent->GetOwner()->GetName());
		// Fallback to rule-based
		return GetFallbackAction(Context);
	}
}

FTacticalAction FSTTask_QueryRLPolicy::GetFallbackAction(FStateTreeExecutionContext& Context) const
{
	FInstanceDataType& InstanceData = Context.GetInstanceData(*this);
	FFollowerStateTreeContext& SharedContext = InstanceData.StateTreeComp->GetSharedContext();

	// Simple rule-based fallback based on command type from context
	switch (SharedContext.CurrentCommand.CommandType)
	{
	case EStrategicCommandType::Assault:
		return SharedContext.CurrentObservation.AgentHealth > 0.5f
			? ETacticalAction::AggressiveAssault
			: ETacticalAction::CautiousAdvance;

	case EStrategicCommandType::Defend:
		return SharedContext.bInCover
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
