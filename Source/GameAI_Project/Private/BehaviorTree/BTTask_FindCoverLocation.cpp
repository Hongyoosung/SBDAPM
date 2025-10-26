// Copyright Epic Games, Inc. All Rights Reserved.

#include "BehaviorTree/BTTask_FindCoverLocation.h"
#include "BehaviorTree/BlackboardComponent.h"
#include "AIController.h"
#include "Kismet/GameplayStatics.h"
#include "DrawDebugHelpers.h"

UBTTask_FindCoverLocation::UBTTask_FindCoverLocation()
{
	NodeName = "Find Cover Location";
	bNotifyTick = false; // This is a one-shot task, doesn't need tick
}

EBTNodeResult::Type UBTTask_FindCoverLocation::ExecuteTask(UBehaviorTreeComponent& OwnerComp, uint8* NodeMemory)
{
	AAIController* AIController = OwnerComp.GetAIOwner();
	if (!AIController)
	{
		UE_LOG(LogTemp, Error, TEXT("BTTask_FindCoverLocation: No AI Controller found"));
		return EBTNodeResult::Failed;
	}

	APawn* ControlledPawn = AIController->GetPawn();
	if (!ControlledPawn)
	{
		UE_LOG(LogTemp, Error, TEXT("BTTask_FindCoverLocation: No controlled pawn found"));
		return EBTNodeResult::Failed;
	}

	FVector CoverLocation;
	if (FindBestCover(ControlledPawn, CoverLocation))
	{
		// Write cover location to Blackboard
		UBlackboardComponent* BlackboardComp = OwnerComp.GetBlackboardComponent();
		if (BlackboardComp)
		{
			BlackboardComp->SetValueAsVector(CoverLocationKey, CoverLocation);
			UE_LOG(LogTemp, Log, TEXT("BTTask_FindCoverLocation: Found cover at %s"), *CoverLocation.ToString());

			if (bDrawDebug)
			{
				// Draw a bright green sphere at the selected cover location
				DrawDebugSphere(ControlledPawn->GetWorld(), CoverLocation, 150.0f, 16, FColor::Green, false, 5.0f, 0, 5.0f);

				// Draw a line from the agent to the cover
				DrawDebugLine(ControlledPawn->GetWorld(), ControlledPawn->GetActorLocation(), CoverLocation,
					FColor::Cyan, false, 5.0f, 0, 3.0f);
			}

			return EBTNodeResult::Succeeded;
		}
	}

	UE_LOG(LogTemp, Warning, TEXT("BTTask_FindCoverLocation: No cover found within radius %.1f"), SearchRadius);
	return EBTNodeResult::Failed;
}

bool UBTTask_FindCoverLocation::FindBestCover(APawn* ControlledPawn, FVector& OutCoverLocation)
{
	UWorld* World = ControlledPawn->GetWorld();
	if (!World)
	{
		return false;
	}

	// Find all cover actors within the world
	TArray<AActor*> CoverActors;
	UGameplayStatics::GetAllActorsWithTag(World, CoverTag, CoverActors);

	if (CoverActors.Num() == 0)
	{
		UE_LOG(LogTemp, Warning, TEXT("BTTask_FindCoverLocation: No actors with tag '%s' found in the world"), *CoverTag.ToString());
		return false;
	}

	FVector AgentLocation = ControlledPawn->GetActorLocation();
	float BestScore = -1.0f;
	FVector BestCoverLocation = FVector::ZeroVector;
	bool bFoundCover = false;

	for (AActor* CoverActor : CoverActors)
	{
		if (!CoverActor)
		{
			continue;
		}

		FVector CoverLocation = CoverActor->GetActorLocation();
		float Distance = FVector::Dist(AgentLocation, CoverLocation);

		// Only consider cover within search radius
		if (Distance <= SearchRadius)
		{
			// Simple scoring: prefer closer cover
			// TODO: Factor in enemy positions for better scoring
			// - Prefer cover that puts distance between agent and enemies
			// - Prefer cover with good line-of-sight blocking
			// - Consider escape routes from the cover position
			float Score = SearchRadius - Distance;

			if (Score > BestScore)
			{
				BestScore = Score;
				BestCoverLocation = CoverLocation;
				bFoundCover = true;

				if (bDrawDebug)
				{
					// Draw candidate cover locations in yellow
					DrawDebugSphere(World, CoverLocation, 100.0f, 12, FColor::Yellow, false, 3.0f);
				}
			}
		}
		else if (bDrawDebug)
		{
			// Draw out-of-range cover in red
			DrawDebugSphere(World, CoverLocation, 80.0f, 8, FColor::Red, false, 2.0f);
		}
	}

	if (bFoundCover)
	{
		OutCoverLocation = BestCoverLocation;
		UE_LOG(LogTemp, Log, TEXT("BTTask_FindCoverLocation: Selected cover at distance %.1f (Score: %.1f)"),
			FVector::Dist(AgentLocation, BestCoverLocation), BestScore);
		return true;
	}

	return false;
}

FString UBTTask_FindCoverLocation::GetStaticDescription() const
{
	return FString::Printf(TEXT("Find cover within %.1f units"), SearchRadius);
}
