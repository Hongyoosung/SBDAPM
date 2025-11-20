// Copyright Epic Games, Inc. All Rights Reserved.

#include "StateTree/Tasks/STTask_ExecuteMove.h"
#include "StateTree/FollowerStateTreeContext.h"
#include "Team/FollowerAgentComponent.h"
#include "AIController.h"
#include "NavigationSystem.h"
#include "GameFramework/Pawn.h"
#include "GameFramework/CharacterMovementComponent.h"
#include "DrawDebugHelpers.h"
#include "Kismet/GameplayStatics.h"

EStateTreeRunStatus FSTTask_ExecuteMove::EnterState(FStateTreeExecutionContext& Context, const FStateTreeTransitionResult& Transition) const
{
	FInstanceDataType& InstanceData = Context.GetInstanceData(*this);

	// Validate inputs
	if (!InstanceData.Context.FollowerComponent || !InstanceData.Context.AIController)
	{
		UE_LOG(LogTemp, Error, TEXT("STTask_ExecuteMove: Invalid inputs (missing component/controller)"));
		return EStateTreeRunStatus::Failed;
	}

	APawn* Pawn = InstanceData.Context.AIController->GetPawn();
	if (!Pawn)
	{
		UE_LOG(LogTemp, Error, TEXT("STTask_ExecuteMove: No pawn controlled"));
		return EStateTreeRunStatus::Failed;
	}

	// Initialize destination based on command
	InstanceData.CurrentDestination = GetDestinationForCommand(Context);
	InstanceData.LastPosition = Pawn->GetActorLocation();
	InstanceData.bHasReachedDestination = false;
	InstanceData.TotalDistanceTraveled = 0.0f;
	InstanceData.CurrentWaypointIndex = 0;

	UE_LOG(LogTemp, Log, TEXT("STTask_ExecuteMove: Starting movement to %s (Command: %s)"),
		*InstanceData.CurrentDestination.ToString(),
		*UEnum::GetValueAsString(InstanceData.Context.CurrentCommand.CommandType));

	return EStateTreeRunStatus::Running;
}

EStateTreeRunStatus FSTTask_ExecuteMove::Tick(FStateTreeExecutionContext& Context, float DeltaTime) const
{
	FInstanceDataType& InstanceData = Context.GetInstanceData(*this);

	// Check if should abort
	if (ShouldCompleteMovement(Context))
	{
		return EStateTreeRunStatus::Succeeded;
	}

	// Update timers
	InstanceData.Context.TimeInTacticalAction += DeltaTime;

	// Track distance traveled
	APawn* Pawn = InstanceData.Context.AIController->GetPawn();
	if (Pawn)
	{
		float DistanceThisFrame = FVector::Dist(Pawn->GetActorLocation(), InstanceData.LastPosition);
		InstanceData.TotalDistanceTraveled += DistanceThisFrame;
		InstanceData.LastPosition = Pawn->GetActorLocation();
	}

	// Execute current tactical action
	ExecuteTacticalAction(Context, DeltaTime);

	// Calculate and provide reward
	float Reward = CalculateMovementReward(Context, DeltaTime);
	if (Reward != 0.0f && InstanceData.Context.FollowerComponent)
	{
		InstanceData.Context.FollowerComponent->ProvideReward(Reward, false);
	}

	return EStateTreeRunStatus::Running;
}

void FSTTask_ExecuteMove::ExitState(FStateTreeExecutionContext& Context, const FStateTreeTransitionResult& Transition) const
{
	FInstanceDataType& InstanceData = Context.GetInstanceData(*this);

	if (InstanceData.Context.AIController)
	{
		InstanceData.Context.AIController->StopMovement();
	}

	UE_LOG(LogTemp, Log, TEXT("STTask_ExecuteMove: Exiting movement (distance traveled: %.1fcm, reached: %s)"),
		InstanceData.TotalDistanceTraveled,
		InstanceData.bHasReachedDestination ? TEXT("Yes") : TEXT("No"));
}

void FSTTask_ExecuteMove::ExecuteTacticalAction(FStateTreeExecutionContext& Context, float DeltaTime) const
{
	FInstanceDataType& InstanceData = Context.GetInstanceData(*this);

	switch (InstanceData.Context.CurrentTacticalAction)
	{
	case ETacticalAction::Sprint:
		ExecuteSprint(Context, DeltaTime);
		break;

	case ETacticalAction::Crouch:
		ExecuteCrouch(Context, DeltaTime);
		break;

	case ETacticalAction::Patrol:
		ExecutePatrol(Context, DeltaTime);
		break;

	case ETacticalAction::CautiousAdvance:
		ExecuteCautiousAdvance(Context, DeltaTime);
		break;

	case ETacticalAction::FlankLeft:
		ExecuteFlankManeuver(Context, DeltaTime, true);
		break;

	case ETacticalAction::FlankRight:
		ExecuteFlankManeuver(Context, DeltaTime, false);
		break;

	default:
		// Default to patrol movement
		ExecutePatrol(Context, DeltaTime);
		break;
	}
}

void FSTTask_ExecuteMove::ExecuteSprint(FStateTreeExecutionContext& Context, float DeltaTime) const
{
	FInstanceDataType& InstanceData = Context.GetInstanceData(*this);

	// Sprint: Fast movement to destination
	MoveToLocation(Context, InstanceData.CurrentDestination, InstanceData.SprintSpeedMultiplier);
}

void FSTTask_ExecuteMove::ExecuteCrouch(FStateTreeExecutionContext& Context, float DeltaTime) const
{
	FInstanceDataType& InstanceData = Context.GetInstanceData(*this);

	// Crouch: Slow stealthy movement
	MoveToLocation(Context, InstanceData.CurrentDestination, InstanceData.CrouchSpeedMultiplier);
}

void FSTTask_ExecuteMove::ExecutePatrol(FStateTreeExecutionContext& Context, float DeltaTime) const
{
	FInstanceDataType& InstanceData = Context.GetInstanceData(*this);

	// Patrol: Standard speed movement
	if (InstanceData.Context.CurrentCommand.CommandType == EStrategicCommandType::Patrol)
	{
		// Check if reached current waypoint
		APawn* Pawn = InstanceData.Context.AIController->GetPawn();
		if (Pawn && FVector::Dist(Pawn->GetActorLocation(), InstanceData.CurrentDestination) < InstanceData.WaypointAcceptanceRadius)
		{
			// Move to next waypoint
			InstanceData.CurrentDestination = GetNextPatrolWaypoint(Context);
			InstanceData.CurrentWaypointIndex++;

			UE_LOG(LogTemp, Log, TEXT("STTask_ExecuteMove: Reached waypoint, moving to next (Index: %d)"),
				InstanceData.CurrentWaypointIndex);
		}
	}

	MoveToLocation(Context, InstanceData.CurrentDestination, InstanceData.PatrolSpeedMultiplier);
}

void FSTTask_ExecuteMove::ExecuteCautiousAdvance(FStateTreeExecutionContext& Context, float DeltaTime) const
{
	FInstanceDataType& InstanceData = Context.GetInstanceData(*this);

	// Cautious advance: Slower movement, checking for cover
	// TODO: Implement cover-to-cover movement logic
	MoveToLocation(Context, InstanceData.CurrentDestination, 0.8f);
}

void FSTTask_ExecuteMove::ExecuteFlankManeuver(FStateTreeExecutionContext& Context, float DeltaTime, bool bFlankLeft) const
{
	FInstanceDataType& InstanceData = Context.GetInstanceData(*this);

	APawn* Pawn = InstanceData.Context.AIController->GetPawn();
	if (!Pawn) return;

	// Calculate flank position
	FVector ToDestination = (InstanceData.CurrentDestination - Pawn->GetActorLocation()).GetSafeNormal();
	FVector FlankDirection = FVector::CrossProduct(ToDestination, FVector::UpVector);
	if (!bFlankLeft)
	{
		FlankDirection = -FlankDirection;
	}

	FVector FlankPosition = InstanceData.CurrentDestination + FlankDirection * InstanceData.FlankOffsetDistance;

	MoveToLocation(Context, FlankPosition, 1.0f);
}

FVector FSTTask_ExecuteMove::GetDestinationForCommand(FStateTreeExecutionContext& Context) const
{
	FInstanceDataType& InstanceData = Context.GetInstanceData(*this);

	// Priority 1: Command's target location
	if (InstanceData.Context.CurrentCommand.TargetLocation != FVector::ZeroVector)
	{
		return InstanceData.Context.CurrentCommand.TargetLocation;
	}

	// Priority 2: Command's target actor
	if (InstanceData.Context.CurrentCommand.TargetActor)
	{
		return InstanceData.Context.CurrentCommand.TargetActor->GetActorLocation();
	}

	// Priority 3: Primary target (for Advance command)
	if (InstanceData.Context.CurrentCommand.CommandType == EStrategicCommandType::Advance && InstanceData.Context.PrimaryTarget)
	{
		return InstanceData.Context.PrimaryTarget->GetActorLocation();
	}

	// Priority 4: Patrol waypoint
	if (InstanceData.Context.CurrentCommand.CommandType == EStrategicCommandType::Patrol)
	{
		return GetNextPatrolWaypoint(Context);
	}

	// Fallback: Current position (hold)
	APawn* Pawn = InstanceData.Context.AIController->GetPawn();
	return Pawn ? Pawn->GetActorLocation() : FVector::ZeroVector;
}

FVector FSTTask_ExecuteMove::GetNextPatrolWaypoint(FStateTreeExecutionContext& Context) const
{
	FInstanceDataType& InstanceData = Context.GetInstanceData(*this);

	// TODO: Implement proper patrol waypoint system
	// For now, simple back-and-forth patrol

	if (!InstanceData.Context.CurrentCommand.TargetLocation.IsZero())
	{
		// Simple 2-point patrol between command location and origin
		if (InstanceData.CurrentWaypointIndex % 2 == 0)
		{
			return InstanceData.Context.CurrentCommand.TargetLocation;
		}
		else
		{
			APawn* Pawn = InstanceData.Context.AIController->GetPawn();
			if (Pawn)
			{
				// Return to start position (approximation)
				return Pawn->GetActorLocation() + FVector(500.0f, 0.0f, 0.0f);
			}
		}
	}

	// Fallback: Random point nearby
	APawn* Pawn = InstanceData.Context.AIController->GetPawn();
	if (Pawn)
	{
		FVector RandomOffset = FMath::VRand() * FMath::RandRange(500.0f, 1000.0f);
		RandomOffset.Z = 0.0f; // Keep on same plane
		return Pawn->GetActorLocation() + RandomOffset;
	}

	return FVector::ZeroVector;
}

float FSTTask_ExecuteMove::CalculateMovementReward(FStateTreeExecutionContext& Context, float DeltaTime) const
{
	FInstanceDataType& InstanceData = Context.GetInstanceData(*this);

	float Reward = 0.0f;

	// Reward for distance traveled (encourage movement)
	float DistanceReward = (InstanceData.TotalDistanceTraveled / 100.0f) * 2.0f; // +2.0 per meter
	Reward += DistanceReward * DeltaTime;

	// Reward for reaching waypoint (one-time)
	if (InstanceData.bHasReachedDestination && InstanceData.Context.CurrentCommand.CommandType == EStrategicCommandType::Patrol)
	{
		Reward += 3.0f; // Waypoint reached
	}

	// Penalty for being stationary
	APawn* Pawn = InstanceData.Context.AIController->GetPawn();
	if (Pawn && Pawn->GetVelocity().SizeSquared() < 10.0f)
	{
		Reward -= 1.0f * DeltaTime; // Small penalty for not moving
	}

	return Reward;
}

bool FSTTask_ExecuteMove::ShouldCompleteMovement(FStateTreeExecutionContext& Context) const
{
	FInstanceDataType& InstanceData = Context.GetInstanceData(*this);

	// Abort if dead
	if (!InstanceData.Context.bIsAlive)
	{
		return true;
	}

	// Abort if command changed/invalid
	if (!InstanceData.Context.bIsCommandValid)
	{
		return true;
	}

	// Complete if destination reached (for non-patrol commands)
	if (InstanceData.Context.CurrentCommand.CommandType != EStrategicCommandType::Patrol)
	{
		APawn* Pawn = InstanceData.Context.AIController->GetPawn();
		if (Pawn && FVector::Dist(Pawn->GetActorLocation(), InstanceData.CurrentDestination) < InstanceData.WaypointAcceptanceRadius)
		{
			InstanceData.bHasReachedDestination = true;
			return true;
		}
	}

	// Continue moving
	return false;
}

void FSTTask_ExecuteMove::MoveToLocation(FStateTreeExecutionContext& Context, const FVector& Destination, float SpeedMultiplier) const
{
	FInstanceDataType& InstanceData = Context.GetInstanceData(*this);

	if (!InstanceData.Context.AIController) return;

	// Move to destination
	InstanceData.Context.AIController->MoveToLocation(Destination, InstanceData.WaypointAcceptanceRadius);

	// Adjust movement speed via CharacterMovementComponent (if available)
	APawn* Pawn = InstanceData.Context.AIController->GetPawn();
	if (Pawn)
	{
		if (UCharacterMovementComponent* MovementComp = Pawn->FindComponentByClass<UCharacterMovementComponent>())
		{
			// Note: This modifies max walk speed directly
			// You may want to store original speed and restore in ExitState
			float BaseSpeed = 600.0f; // Default UE character speed
			MovementComp->MaxWalkSpeed = BaseSpeed * SpeedMultiplier;
		}
	}
}
