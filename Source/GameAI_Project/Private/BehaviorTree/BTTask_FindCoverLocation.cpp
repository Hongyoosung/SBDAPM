// Copyright Epic Games, Inc. All Rights Reserved.

#include "BehaviorTree/BTTask_FindCoverLocation.h"
#include "BehaviorTree/BlackboardComponent.h"
#include "AIController.h"
#include "Kismet/GameplayStatics.h"
#include "DrawDebugHelpers.h"
#include "EnvironmentQuery/EnvQueryManager.h"
#include "EnvironmentQuery/EnvQuery.h"

UBTTask_FindCoverLocation::UBTTask_FindCoverLocation()
{
	NodeName = "Find Cover Location (EQS)";
	bNotifyTick = false;
	bNotifyTaskFinished = true;
}

EBTNodeResult::Type UBTTask_FindCoverLocation::ExecuteTask(UBehaviorTreeComponent& OwnerComp, uint8* NodeMemory)
{
	AAIController* AIController = OwnerComp.GetAIOwner();
	if (!AIController)
	{
		UE_LOG(LogTemp, Error, TEXT("BTTask_FindCoverLocation: No AI Controller"));
		return EBTNodeResult::Failed;
	}

	APawn* ControlledPawn = AIController->GetPawn();
	if (!ControlledPawn)
	{
		UE_LOG(LogTemp, Error, TEXT("BTTask_FindCoverLocation: No pawn"));
		return EBTNodeResult::Failed;
	}

	// Try EQS first if query is set
	if (CoverQuery)
	{
		UEnvQueryManager* QueryManager = UEnvQueryManager::GetCurrent(ControlledPawn->GetWorld());
		if (QueryManager)
		{
			FEnvQueryRequest QueryRequest(CoverQuery, ControlledPawn);
			QueryRequestID = QueryRequest.Execute(
				EEnvQueryRunMode::SingleResult,
				this,
				&UBTTask_FindCoverLocation::OnQueryFinished);

			if (QueryRequestID != INDEX_NONE)
			{
				CachedBehaviorTreeComp = &OwnerComp;
				return EBTNodeResult::InProgress;
			}
		}

		UE_LOG(LogTemp, Warning, TEXT("BTTask_FindCoverLocation: EQS query failed to start"));
	}

	// Fallback to legacy search
	if (bUseLegacySearch)
	{
		FVector CoverLocation;
		if (FindBestCover(ControlledPawn, CoverLocation))
		{
			UBlackboardComponent* BlackboardComp = OwnerComp.GetBlackboardComponent();
			if (BlackboardComp)
			{
				BlackboardComp->SetValueAsVector(CoverLocationKey, CoverLocation);
				UE_LOG(LogTemp, Log, TEXT("BTTask_FindCoverLocation: Found cover (legacy) at %s"), *CoverLocation.ToString());

				if (bDrawDebug)
				{
					DrawDebugSphere(ControlledPawn->GetWorld(), CoverLocation, 150.0f, 16, FColor::Green, false, 5.0f);
					DrawDebugLine(ControlledPawn->GetWorld(), ControlledPawn->GetActorLocation(), CoverLocation,
						FColor::Cyan, false, 5.0f, 0, 3.0f);
				}

				return EBTNodeResult::Succeeded;
			}
		}
	}

	UE_LOG(LogTemp, Warning, TEXT("BTTask_FindCoverLocation: No cover found"));
	return EBTNodeResult::Failed;
}

EBTNodeResult::Type UBTTask_FindCoverLocation::AbortTask(UBehaviorTreeComponent& OwnerComp, uint8* NodeMemory)
{
	if (QueryRequestID != INDEX_NONE)
	{
		UEnvQueryManager* QueryManager = UEnvQueryManager::GetCurrent(OwnerComp.GetWorld());
		if (QueryManager)
		{
			QueryManager->AbortQuery(QueryRequestID);
		}
		QueryRequestID = INDEX_NONE;
	}

	return EBTNodeResult::Aborted;
}

void UBTTask_FindCoverLocation::OnQueryFinished(TSharedPtr<FEnvQueryResult> Result)
{
	QueryRequestID = INDEX_NONE;

	if (!CachedBehaviorTreeComp.IsValid())
	{
		return;
	}

	UBehaviorTreeComponent* OwnerComp = CachedBehaviorTreeComp.Get();
	UBlackboardComponent* BlackboardComp = OwnerComp->GetBlackboardComponent();

	if (Result->IsSuccessful() && BlackboardComp)
	{
		FVector CoverLocation = Result->GetItemAsLocation(0);
		BlackboardComp->SetValueAsVector(CoverLocationKey, CoverLocation);

		if (bDrawDebug && OwnerComp->GetAIOwner())
		{
			APawn* Pawn = OwnerComp->GetAIOwner()->GetPawn();
			if (Pawn)
			{
				DrawDebugSphere(Pawn->GetWorld(), CoverLocation, 150.0f, 16, FColor::Green, false, 5.0f);
				DrawDebugLine(Pawn->GetWorld(), Pawn->GetActorLocation(), CoverLocation,
					FColor::Cyan, false, 5.0f, 0, 3.0f);
			}
		}

		UE_LOG(LogTemp, Log, TEXT("BTTask_FindCoverLocation: EQS found cover at %s"), *CoverLocation.ToString());
		FinishLatentTask(*OwnerComp, EBTNodeResult::Succeeded);
	}
	else
	{
		UE_LOG(LogTemp, Warning, TEXT("BTTask_FindCoverLocation: EQS query failed"));
		FinishLatentTask(*OwnerComp, EBTNodeResult::Failed);
	}
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
