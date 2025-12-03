// Copyright Epic Games, Inc. All Rights Reserved.

#include "StateTree/Evaluators/STEvaluator_SyncObjective.h"
#include "StateTree/FollowerStateTreeContext.h"
#include "StateTree/FollowerStateTreeComponent.h"
#include "Team/FollowerAgentComponent.h"

void FSTEvaluator_SyncObjective::TreeStart(FStateTreeExecutionContext& Context) const
{
	FInstanceDataType& InstanceData = Context.GetInstanceData(*this);

	if (!InstanceData.StateTreeComp)
	{
		UE_LOG(LogTemp, Error, TEXT("[SYNC OBJECTIVE] TreeStart: ❌ StateTreeComp is null!"));
		return;
	}

	// Get SHARED context from component (not a local copy!)
	FFollowerStateTreeContext& SharedContext = InstanceData.StateTreeComp->GetSharedContext();

	// Initialize context outputs from follower component
	// CRITICAL: This must happen in TreeStart so conditions can evaluate during initial state selection
	if (SharedContext.FollowerComponent)
	{
		UObjective* CurrentObjective = SharedContext.FollowerComponent->GetCurrentObjective();
		bool bHasObjective = CurrentObjective != nullptr && CurrentObjective->IsActive();

		// Set context outputs (shared with conditions/tasks)
		SharedContext.CurrentObjective = CurrentObjective;
		SharedContext.bHasActiveObjective = bHasObjective;

		// Sync PrimaryTarget from objective
		if (CurrentObjective)
		{
			SharedContext.PrimaryTarget = CurrentObjective->TargetActor;

			// Initialize tracking variables
			InstanceData.LastObjectiveType = CurrentObjective->Type;
			InstanceData.LastObjective = CurrentObjective;

			UE_LOG(LogTemp, Warning, TEXT("[SYNC OBJECTIVE] TreeStart: Initialized SharedContext - Type=%s, Active=%d, Target=%s"),
				*UEnum::GetValueAsString(CurrentObjective->Type),
				bHasObjective,
				CurrentObjective->TargetActor ? *CurrentObjective->TargetActor->GetName() : TEXT("None"));
		}
		else
		{
			InstanceData.LastObjective = nullptr;
			UE_LOG(LogTemp, Warning, TEXT("[SYNC OBJECTIVE] TreeStart: No objective assigned"));
		}
	}
	else
	{
		UE_LOG(LogTemp, Error, TEXT("[SYNC OBJECTIVE] TreeStart: ❌ FollowerComponent is null!"));
	}
}

void FSTEvaluator_SyncObjective::Tick(FStateTreeExecutionContext& Context, float DeltaTime) const
{
	FInstanceDataType& InstanceData = Context.GetInstanceData(*this);

	if (!InstanceData.StateTreeComp)
	{
		UE_LOG(LogTemp, Error, TEXT("[SYNC OBJECTIVE] Tick: ❌ StateTreeComp is null!"));
		return;
	}

	// Get SHARED context from component (not a local copy!)
	FFollowerStateTreeContext& SharedContext = InstanceData.StateTreeComp->GetSharedContext();

	if (!SharedContext.FollowerComponent)
	{
		UE_LOG(LogTemp, Error, TEXT("[SYNC OBJECTIVE] Tick: ❌ FollowerComponent is null!"));
		return;
	}

	// Get current objective from follower component
	UObjective* NewObjective = SharedContext.FollowerComponent->GetCurrentObjective();
	bool bHasNewObjective = NewObjective != nullptr && NewObjective->IsActive();

	// Update context outputs (shared with all tasks/evaluators)
	SharedContext.CurrentObjective = NewObjective;
	SharedContext.bHasActiveObjective = bHasNewObjective;

	// Update primary target from objective
	if (NewObjective)
	{
		SharedContext.PrimaryTarget = NewObjective->TargetActor;
	}
	else
	{
		SharedContext.PrimaryTarget = nullptr;
	}

	// Log objective changes (Warning level for visibility)
	if (NewObjective != InstanceData.LastObjective)
	{
		if (NewObjective)
		{
			FString TargetInfo = NewObjective->TargetActor ?
				FString::Printf(TEXT("Target: %s"), *NewObjective->TargetActor->GetName()) :
				TEXT("Target: None");

			FString OldTypeStr = InstanceData.LastObjective ?
				UEnum::GetValueAsString(InstanceData.LastObjective->Type) :
				TEXT("None");

			UE_LOG(LogTemp, Warning, TEXT("[SYNC OBJECTIVE] 📝 Objective changed: '%s' → '%s', Active=%d, %s"),
				*OldTypeStr,
				*UEnum::GetValueAsString(NewObjective->Type),
				bHasNewObjective,
				*TargetInfo);

			InstanceData.LastObjectiveType = NewObjective->Type;
		}
		else
		{
			UE_LOG(LogTemp, Warning, TEXT("[SYNC OBJECTIVE] 📝 Objective cleared (was: %s)"),
				InstanceData.LastObjective ? *UEnum::GetValueAsString(InstanceData.LastObjective->Type) : TEXT("None"));
		}

		InstanceData.LastObjective = NewObjective;
	}

	// Periodic verbose logging (every 60 ticks)
	static int32 TickCounter = 0;
	if (++TickCounter % 60 == 0)
	{
		UE_LOG(LogTemp, Warning, TEXT("[SYNC OBJECTIVE] Tick #%d: Type=%s, Active=%d"),
			TickCounter,
			NewObjective ? *UEnum::GetValueAsString(NewObjective->Type) : TEXT("NULL"),
			bHasNewObjective);
	}
}

void FSTEvaluator_SyncObjective::TreeStop(FStateTreeExecutionContext& Context) const
{
	// No cleanup needed
}
