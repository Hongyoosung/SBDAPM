// Copyright Epic Games, Inc. All Rights Reserved.

#include "StateTree/Tasks/STTask_ExecuteSupport.h"
#include "StateTree/FollowerStateTreeSchema.h"
#include "Team/FollowerAgentComponent.h"
#include "AIController.h"
#include "GameFramework/Pawn.h"
#include "DrawDebugHelpers.h"

EStateTreeRunStatus FSTTask_ExecuteSupport::EnterState(FStateTreeExecutionContext& Context, const FStateTreeTransitionResult& Transition) const
{
	FInstanceDataType& InstanceData = Context.GetInstanceData(*this);

	// Validate inputs
	if (!InstanceData.Context.FollowerComponent || !InstanceData.Context.AIController)
	{
		UE_LOG(LogTemp, Error, TEXT("STTask_ExecuteSupport: Invalid inputs (missing component/controller)"));
		return EStateTreeRunStatus::Failed;
	}

	// Reset timers
	InstanceData.TimeSinceLastRLQuery = 0.0f;
	InstanceData.Context.TimeInTacticalAction = 0.0f;
	InstanceData.Context.ActionProgress = 0.0f;

	UE_LOG(LogTemp, Log, TEXT("STTask_ExecuteSupport: Starting support with tactic: %d"),
		static_cast<int32>(InstanceData.Context.CurrentTacticalAction));

	return EStateTreeRunStatus::Running;
}

EStateTreeRunStatus FSTTask_ExecuteSupport::Tick(FStateTreeExecutionContext& Context, float DeltaTime) const
{
	FInstanceDataType& InstanceData = Context.GetInstanceData(*this);

	// Check if should abort
	if (!InstanceData.Context.bIsAlive || !InstanceData.Context.bIsCommandValid)
	{
		return EStateTreeRunStatus::Succeeded;
	}

	// Update timers
	InstanceData.TimeSinceLastRLQuery += DeltaTime;
	InstanceData.Context.TimeInTacticalAction += DeltaTime;

	// Re-query RL policy if interval elapsed
	if (InstanceData.RLQueryInterval > 0.0f && InstanceData.TimeSinceLastRLQuery >= InstanceData.RLQueryInterval)
	{
		// This would trigger a transition back to QueryRLPolicy state
		InstanceData.TimeSinceLastRLQuery = 0.0f;
	}

	// Execute current tactical action
	ExecuteTacticalAction(Context, DeltaTime);

	// Calculate and provide reward
	float Reward = 0.0f;

	// Reward for providing support
	if (InstanceData.Context.CurrentTacticalAction == ETacticalAction::ProvideCoveringFire ||
		InstanceData.Context.CurrentTacticalAction == ETacticalAction::SuppressiveFire)
	{
		if (InstanceData.Context.bHasLOS && InstanceData.Context.bWeaponReady)
		{
			Reward += FTacticalRewards::COVERING_FIRE * DeltaTime;
		}
	}

	// Reward for following support command
	if (InstanceData.Context.bIsCommandValid)
	{
		Reward += FTacticalRewards::FOLLOW_COMMAND * DeltaTime;
	}

	if (Reward != 0.0f && InstanceData.Context.FollowerComponent)
	{
		InstanceData.Context.FollowerComponent->ProvideReward(Reward, false);
	}

	return EStateTreeRunStatus::Running;
}

void FSTTask_ExecuteSupport::ExitState(FStateTreeExecutionContext& Context, const FStateTreeTransitionResult& Transition) const
{
	FInstanceDataType& InstanceData = Context.GetInstanceData(*this);

	UE_LOG(LogTemp, Log, TEXT("STTask_ExecuteSupport: Exiting support (time in action: %.1fs)"),
		InstanceData.Context.TimeInTacticalAction);
}

void FSTTask_ExecuteSupport::ExecuteTacticalAction(FStateTreeExecutionContext& Context, float DeltaTime) const
{
	FInstanceDataType& InstanceData = Context.GetInstanceData(*this);

	switch (InstanceData.Context.CurrentTacticalAction)
	{
	case ETacticalAction::ProvideCoveringFire:
		ExecuteProvideCoveringFire(Context, DeltaTime);
		break;

	case ETacticalAction::SuppressiveFire:
		ExecuteSuppressiveFire(Context, DeltaTime);
		break;

	case ETacticalAction::Reload:
		ExecuteReload(Context, DeltaTime);
		break;

	case ETacticalAction::UseAbility:
		ExecuteUseAbility(Context, DeltaTime);
		break;

	default:
		// Default to providing covering fire
		ExecuteProvideCoveringFire(Context, DeltaTime);
		break;
	}
}

void FSTTask_ExecuteSupport::ExecuteProvideCoveringFire(FStateTreeExecutionContext& Context, float DeltaTime) const
{
	FInstanceDataType& InstanceData = Context.GetInstanceData(*this);

	APawn* Pawn = InstanceData.Context.AIController ? InstanceData.Context.AIController->GetPawn() : nullptr;
	if (!Pawn) return;

	// Stay in position and provide covering fire
	if (InstanceData.Context.AIController)
	{
		InstanceData.Context.AIController->StopMovement();
	}

	// Focus on primary target or sweep visible enemies
	if (InstanceData.Context.PrimaryTarget && InstanceData.Context.bWeaponReady)
	{
		if (InstanceData.Context.AIController)
		{
			InstanceData.Context.AIController->SetFocus(InstanceData.Context.PrimaryTarget);
		}
	}
	else if (InstanceData.Context.VisibleEnemies.Num() > 0)
	{
		// Rotate through visible enemies to suppress area
		int32 TargetIndex = static_cast<int32>(InstanceData.Context.TimeInTacticalAction) % InstanceData.Context.VisibleEnemies.Num();
		if (InstanceData.Context.VisibleEnemies.IsValidIndex(TargetIndex) &&
			InstanceData.Context.VisibleEnemies[TargetIndex])
		{
			if (InstanceData.Context.AIController)
			{
				InstanceData.Context.AIController->SetFocus(InstanceData.Context.VisibleEnemies[TargetIndex]);
			}
		}
	}

	InstanceData.Context.bIsMoving = false;
}

void FSTTask_ExecuteSupport::ExecuteSuppressiveFire(FStateTreeExecutionContext& Context, float DeltaTime) const
{
	FInstanceDataType& InstanceData = Context.GetInstanceData(*this);

	APawn* Pawn = InstanceData.Context.AIController ? InstanceData.Context.AIController->GetPawn() : nullptr;
	if (!Pawn) return;

	// Similar to covering fire but with higher volume, lower accuracy intent
	// Stop movement
	if (InstanceData.Context.AIController)
	{
		InstanceData.Context.AIController->StopMovement();
	}

	// Prioritize suppressing multiple enemies
	if (InstanceData.Context.VisibleEnemies.Num() > 0)
	{
		// Rapidly cycle through targets for suppression
		int32 TargetIndex = (static_cast<int32>(InstanceData.Context.TimeInTacticalAction * 2.0f)) % InstanceData.Context.VisibleEnemies.Num();
		if (InstanceData.Context.VisibleEnemies.IsValidIndex(TargetIndex) &&
			InstanceData.Context.VisibleEnemies[TargetIndex])
		{
			if (InstanceData.Context.AIController)
			{
				InstanceData.Context.AIController->SetFocus(InstanceData.Context.VisibleEnemies[TargetIndex]);
			}
		}
	}
	else if (InstanceData.Context.PrimaryTarget)
	{
		if (InstanceData.Context.AIController)
		{
			InstanceData.Context.AIController->SetFocus(InstanceData.Context.PrimaryTarget);
		}
	}

	InstanceData.Context.bIsMoving = false;
}

void FSTTask_ExecuteSupport::ExecuteReload(FStateTreeExecutionContext& Context, float DeltaTime) const
{
	FInstanceDataType& InstanceData = Context.GetInstanceData(*this);

	APawn* Pawn = InstanceData.Context.AIController ? InstanceData.Context.AIController->GetPawn() : nullptr;
	if (!Pawn) return;

	// Stop movement and seek cover if possible
	if (InstanceData.Context.AIController)
	{
		InstanceData.Context.AIController->StopMovement();
		InstanceData.Context.AIController->ClearFocus(EAIFocusPriority::Gameplay);
	}

	// If not in cover and under fire, try to get to cover
	if (!InstanceData.Context.bInCover && InstanceData.Context.bUnderFire)
	{
		if (InstanceData.Context.NearestCoverLocation != FVector::ZeroVector)
		{
			if (InstanceData.Context.AIController)
			{
				InstanceData.Context.AIController->MoveToLocation(
					InstanceData.Context.NearestCoverLocation, 50.0f);
			}
			InstanceData.Context.bIsMoving = true;
			return;
		}
	}

	// Reload action progress (simulated 2-second reload)
	InstanceData.Context.ActionProgress = FMath::Min(1.0f, InstanceData.Context.TimeInTacticalAction / 2.0f);

	// Mark weapon as not ready during reload
	InstanceData.Context.bWeaponReady = (InstanceData.Context.ActionProgress >= 1.0f);
	InstanceData.Context.bIsMoving = false;

	UE_LOG(LogTemp, VeryVerbose, TEXT("STTask_ExecuteSupport: Reloading (%.1f%% complete)"),
		InstanceData.Context.ActionProgress * 100.0f);
}

void FSTTask_ExecuteSupport::ExecuteUseAbility(FStateTreeExecutionContext& Context, float DeltaTime) const
{
	FInstanceDataType& InstanceData = Context.GetInstanceData(*this);

	APawn* Pawn = InstanceData.Context.AIController ? InstanceData.Context.AIController->GetPawn() : nullptr;
	if (!Pawn) return;

	// Placeholder for ability usage
	// Specific ability logic would be implemented based on available abilities
	// For now, maintain position and signal ability usage intent

	if (InstanceData.Context.AIController)
	{
		InstanceData.Context.AIController->StopMovement();
	}

	// Simulate ability cooldown/activation (3-second ability)
	InstanceData.Context.ActionProgress = FMath::Min(1.0f, InstanceData.Context.TimeInTacticalAction / 3.0f);
	InstanceData.Context.bIsMoving = false;

	UE_LOG(LogTemp, VeryVerbose, TEXT("STTask_ExecuteSupport: Using ability (%.1f%% complete)"),
		InstanceData.Context.ActionProgress * 100.0f);
}
