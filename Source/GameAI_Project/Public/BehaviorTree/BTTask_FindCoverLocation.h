// Copyright Epic Games, Inc. All Rights Reserved.

#pragma once

#include "CoreMinimal.h"
#include "BehaviorTree/BTTaskNode.h"
#include "BTTask_FindCoverLocation.generated.h"

/**
 * BTTask_FindCoverLocation - Finds a suitable cover location and writes it to the Blackboard.
 *
 * This task searches for cover within a specified radius and evaluates
 * positions based on:
 * - Distance from enemies (further = safer)
 * - Availability of line-of-sight blocking
 * - Distance from agent (closer = faster to reach)
 *
 * Usage:
 * - Add this task to the Flee behavior subtree
 * - Configure SearchRadius and CoverTag in the Behavior Tree editor
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
	/**
	 * Called when this task is executed.
	 * Searches for cover and writes the location to Blackboard.
	 */
	virtual EBTNodeResult::Type ExecuteTask(UBehaviorTreeComponent& OwnerComp, uint8* NodeMemory) override;

	/**
	 * Returns a description of this task for the Behavior Tree editor.
	 */
	virtual FString GetStaticDescription() const override;

public:
	/**
	 * Maximum search radius for cover (in cm)
	 * Default: 1500 cm (15 meters)
	 */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Cover", meta = (ClampMin = "100.0", ClampMax = "5000.0"))
	float SearchRadius = 1500.0f;

	/**
	 * Blackboard key to write the cover location to
	 * This should be a Vector key in your Blackboard asset
	 */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Blackboard")
	FName CoverLocationKey = FName("CoverLocation");

	/**
	 * Tag used to identify cover objects in the world
	 * Cover objects should be tagged with this name in the editor
	 * Default: "Cover"
	 */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Cover")
	FName CoverTag = FName("Cover");

	/**
	 * Whether to draw debug visualization
	 * Shows cover candidates and the selected cover location
	 */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Debug")
	bool bDrawDebug = false;

private:
	/**
	 * Find the best cover location based on distance and safety
	 * @param ControlledPawn - The pawn seeking cover
	 * @param OutCoverLocation - Output: The best cover location found
	 * @return true if cover was found, false otherwise
	 */
	bool FindBestCover(APawn* ControlledPawn, FVector& OutCoverLocation);
};
