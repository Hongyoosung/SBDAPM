// Copyright Epic Games, Inc. All Rights Reserved.

#include "BehaviorTree/BTTask_EvasiveMovement.h"
#include "AIController.h"
#include "NavigationSystem.h"
#include "DrawDebugHelpers.h"

UBTTask_EvasiveMovement::UBTTask_EvasiveMovement()
{
	NodeName = "Evasive Movement";
	bNotifyTick = true; // We need tick to track duration and change directions
}

EBTNodeResult::Type UBTTask_EvasiveMovement::ExecuteTask(UBehaviorTreeComponent& OwnerComp, uint8* NodeMemory)
{
	AAIController* AIController = OwnerComp.GetAIOwner();
	if (!AIController)
	{
		UE_LOG(LogTemp, Error, TEXT("BTTask_EvasiveMovement: No AI Controller found"));
		return EBTNodeResult::Failed;
	}

	APawn* ControlledPawn = AIController->GetPawn();
	if (!ControlledPawn)
	{
		UE_LOG(LogTemp, Error, TEXT("BTTask_EvasiveMovement: No controlled pawn found"));
		return EBTNodeResult::Failed;
	}

	// Reset timers
	ElapsedTime = 0.0f;
	LastMoveTime = 0.0f;

	UE_LOG(LogTemp, Log, TEXT("BTTask_EvasiveMovement: Starting evasive movement for %.1f seconds"), Duration);

	// Task will continue in TickTask
	return EBTNodeResult::InProgress;
}

void UBTTask_EvasiveMovement::TickTask(UBehaviorTreeComponent& OwnerComp, uint8* NodeMemory, float DeltaSeconds)
{
	AAIController* AIController = OwnerComp.GetAIOwner();
	if (!AIController)
	{
		FinishLatentTask(OwnerComp, EBTNodeResult::Failed);
		return;
	}

	APawn* ControlledPawn = AIController->GetPawn();
	if (!ControlledPawn)
	{
		FinishLatentTask(OwnerComp, EBTNodeResult::Failed);
		return;
	}

	ElapsedTime += DeltaSeconds;
	LastMoveTime += DeltaSeconds;

	// Check if duration has elapsed
	if (ElapsedTime >= Duration)
	{
		UE_LOG(LogTemp, Log, TEXT("BTTask_EvasiveMovement: Completed evasive movement after %.1f seconds"), ElapsedTime);
		FinishLatentTask(OwnerComp, EBTNodeResult::Succeeded);
		return;
	}

	// Change direction every MoveInterval seconds
	if (LastMoveTime >= MoveInterval)
	{
		LastMoveTime = 0.0f;

		// Calculate random evasive direction
		FVector CurrentLocation = ControlledPawn->GetActorLocation();
		FVector ForwardVector = ControlledPawn->GetActorForwardVector();
		FVector RightVector = ControlledPawn->GetActorRightVector();

		// Random zigzag: alternate left/right with some randomness
		// Angle range: -90 to 90 degrees from forward
		float RandomAngle = FMath::RandRange(-90.0f, 90.0f);
		FVector RandomDirection = ForwardVector.RotateAngleAxis(RandomAngle, FVector::UpVector);
		RandomDirection.Normalize();

		FVector TargetLocation = CurrentLocation + (RandomDirection * MoveDistance);

		// Use navigation system to ensure we move to a valid location
		UNavigationSystemV1* NavSys = UNavigationSystemV1::GetCurrent(ControlledPawn->GetWorld());
		if (NavSys)
		{
			FNavLocation NavLocation;
			// Project the target location onto the navmesh
			if (NavSys->ProjectPointToNavigation(TargetLocation, NavLocation, FVector(500.0f, 500.0f, 500.0f)))
			{
				// Move to the valid navigation location
				AIController->MoveToLocation(NavLocation.Location);

				if (bDrawDebug)
				{
					// Draw a yellow line showing the movement path
					DrawDebugLine(ControlledPawn->GetWorld(), CurrentLocation, NavLocation.Location,
						FColor::Yellow, false, MoveInterval, 0, 2.0f);

					// Draw an orange sphere at the target location
					DrawDebugSphere(ControlledPawn->GetWorld(), NavLocation.Location, 50.0f, 8,
						FColor::Orange, false, MoveInterval);

					// Draw direction arrow
					DrawDebugDirectionalArrow(ControlledPawn->GetWorld(), CurrentLocation,
						CurrentLocation + (RandomDirection * 200.0f), 20.0f, FColor::Cyan, false, MoveInterval, 0, 2.0f);
				}

				UE_LOG(LogTemp, Verbose, TEXT("BTTask_EvasiveMovement: Moving to evasive position (angle: %.1f deg, distance: %.1f)"),
					RandomAngle, FVector::Dist(CurrentLocation, NavLocation.Location));
			}
			else
			{
				// If we can't project to navmesh, try moving in a different direction
				// Just log a warning and wait for the next interval
				UE_LOG(LogTemp, Warning, TEXT("BTTask_EvasiveMovement: Failed to find valid navigation point at %s"),
					*TargetLocation.ToString());
			}
		}
		else
		{
			// No navigation system - move directly (may not be valid)
			AIController->MoveToLocation(TargetLocation);
			UE_LOG(LogTemp, Warning, TEXT("BTTask_EvasiveMovement: No navigation system, moving directly"));
		}
	}
}

FString UBTTask_EvasiveMovement::GetStaticDescription() const
{
	return FString::Printf(TEXT("Zigzag for %.1fs (change every %.1fs, distance %.0f)"), Duration, MoveInterval, MoveDistance);
}
