// Copyright Epic Games, Inc. All Rights Reserved.

#pragma once

#include "CoreMinimal.h"
#include "BehaviorTree/BTTaskNode.h"
#include "BTTask_EvasiveMovement.generated.h"

/**
 * BTTask_EvasiveMovement - Performs evasive zigzag movement to avoid enemy fire.
 *
 * This task makes the agent move in a zigzag pattern by randomly
 * offsetting movement directions. Useful when no cover is available
 * or as a fallback flee behavior.
 *
 * The task executes for a specified duration, changing direction at
 * regular intervals to create unpredictable movement that's harder
 * for enemies to track.
 *
 * Usage:
 * - Add this task to the Flee behavior subtree (typically as a fallback when FindCover fails)
 * - Configure Duration (how long to evade) and MoveDistance (how far each zigzag)
 * - The task will return Success after completing the full duration
 *
 * Returns:
 * - InProgress: While evasive movement is ongoing
 * - Succeeded: After completing the full duration
 * - Failed: If the AI controller or pawn is invalid
 */
UCLASS()
class GAMEAI_PROJECT_API UBTTask_EvasiveMovement : public UBTTaskNode
{
	GENERATED_BODY()

public:
	UBTTask_EvasiveMovement();

protected:
	/**
	 * Called when this task starts executing.
	 * Initializes the evasive movement timers.
	 */
	virtual EBTNodeResult::Type ExecuteTask(UBehaviorTreeComponent& OwnerComp, uint8* NodeMemory) override;

	/**
	 * Called every frame while the task is active.
	 * Handles the zigzag movement logic and duration tracking.
	 */
	virtual void TickTask(UBehaviorTreeComponent& OwnerComp, uint8* NodeMemory, float DeltaSeconds) override;

	/**
	 * Returns a description of this task for the Behavior Tree editor.
	 */
	virtual FString GetStaticDescription() const override;

public:
	/**
	 * Duration of evasive movement (in seconds)
	 * How long the agent should perform zigzag movement before completing
	 * Default: 2.0 seconds
	 */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Movement", meta = (ClampMin = "0.5", ClampMax = "10.0"))
	float Duration = 2.0f;

	/**
	 * Distance for each evasive move (in cm)
	 * How far the agent moves in each zigzag direction change
	 * Default: 300 cm (3 meters)
	 */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Movement", meta = (ClampMin = "100.0", ClampMax = "1000.0"))
	float MoveDistance = 300.0f;

	/**
	 * Interval between direction changes (in seconds)
	 * How often the agent changes zigzag direction
	 * Default: 0.5 seconds
	 */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Movement", meta = (ClampMin = "0.1", ClampMax = "2.0"))
	float MoveInterval = 0.5f;

	/**
	 * Whether to draw debug visualization
	 * Shows the zigzag path and movement targets
	 */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Debug")
	bool bDrawDebug = false;

private:
	/** Tracks total elapsed time for duration check */
	float ElapsedTime;

	/** Tracks time since last direction change */
	float LastMoveTime;
};
