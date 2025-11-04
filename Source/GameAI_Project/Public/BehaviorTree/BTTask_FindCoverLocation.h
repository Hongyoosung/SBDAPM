// Copyright Epic Games, Inc. All Rights Reserved.

#pragma once

#include "CoreMinimal.h"
#include "BehaviorTree/BTTaskNode.h"
#include "EnvironmentQuery/EnvQueryTypes.h"
#include "BTTask_FindCoverLocation.generated.h"

class UEnvQuery;

/**
 * BTTask_FindCoverLocation - Finds a suitable cover location using EQS and writes it to the Blackboard.
 *
 * This task uses Environment Query System (EQS) to search for cover within a specified radius
 * and evaluates positions based on:
 * - Distance from enemies (further = safer)
 * - Availability of line-of-sight blocking
 * - Distance from agent (closer = faster to reach)
 * - Navigability (must be reachable)
 *
 * Usage:
 * - Add this task to the Flee behavior subtree
 * - Assign an EQS query asset (e.g., EQS_FindCover)
 * - The task will write the best cover location to the Blackboard
 * - Use with a MoveTo task to navigate the agent to cover
 *
 * Returns:
 * - Success: If valid cover is found within SearchRadius
 * - Failed: If no cover is available or accessible
 */
UCLASS()
class GAMEAI_PROJECT_API UBTTask_FindCoverLocation : public UBTTaskNode
{
	GENERATED_BODY()

public:
	UBTTask_FindCoverLocation();

protected:
	virtual EBTNodeResult::Type ExecuteTask(UBehaviorTreeComponent& OwnerComp, uint8* NodeMemory) override;
	virtual EBTNodeResult::Type AbortTask(UBehaviorTreeComponent& OwnerComp, uint8* NodeMemory) override;

	virtual FString GetStaticDescription() const override;

	/** Called when EQS query finishes */
	void OnQueryFinished(TSharedPtr<FEnvQueryResult> Result);

public:
	/** EQS query to run for finding cover */
	UPROPERTY(EditAnywhere, Category = "EQS")
	UEnvQuery* CoverQuery = nullptr;

	/** Blackboard key to write the cover location to */
	UPROPERTY(EditAnywhere, Category = "Blackboard")
	FName CoverLocationKey = FName("CoverLocation");

	/** Whether to draw debug visualization */
	UPROPERTY(EditAnywhere, Category = "Debug")
	bool bDrawDebug = false;

	/** Use legacy tag-based search if EQS query is not set */
	UPROPERTY(EditAnywhere, Category = "Cover|Legacy")
	bool bUseLegacySearch = true;

	/** [Legacy] Maximum search radius for cover (in cm) */
	UPROPERTY(EditAnywhere, Category = "Cover|Legacy", meta = (ClampMin = "100.0", ClampMax = "5000.0", EditCondition = "bUseLegacySearch"))
	float SearchRadius = 1500.0f;

	/** [Legacy] Tag used to identify cover objects */
	UPROPERTY(EditAnywhere, Category = "Cover|Legacy", meta = (EditCondition = "bUseLegacySearch"))
	FName CoverTag = FName("Cover");

private:
	/** EQS query request ID */
	int32 QueryRequestID = INDEX_NONE;

	/** Cached behavior tree component */
	TWeakObjectPtr<UBehaviorTreeComponent> CachedBehaviorTreeComp;

	/** Legacy cover finding (fallback) */
	bool FindBestCover(APawn* ControlledPawn, FVector& OutCoverLocation);
};
