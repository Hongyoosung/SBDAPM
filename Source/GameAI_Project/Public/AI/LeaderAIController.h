// Copyright Epic Games, Inc. All Rights Reserved.

#pragma once

#include "CoreMinimal.h"
#include "AIController.h"
#include "LeaderAIController.generated.h"

class UTeamLeaderComponent;

/**
 * Leader AI Controller
 *
 * This controller manages team leader agents.
 *
 * Architecture:
 * - TeamLeaderComponent runs event-driven MCTS in background thread
 * - Issues strategic commands to followers
 * - No tactical execution (no State Tree/BT needed)
 * - Aggregates team observations
 *
 * Setup:
 * 1. Assign this controller to your leader pawn/character
 * 2. Ensure LeaderCharacter has TeamLeaderComponent
 * 3. Register followers with TeamLeaderComponent
 * 4. MCTS runs automatically on strategic events
 */
UCLASS()
class GAMEAI_PROJECT_API ALeaderAIController : public AAIController
{
	GENERATED_BODY()

public:
	ALeaderAIController();

protected:
	virtual void BeginPlay() override;
	virtual void OnPossess(APawn* InPawn) override;
	virtual void OnUnPossess() override;
	virtual void Tick(float DeltaTime) override;

public:
	//--------------------------------------------------------------------------
	// CONFIGURATION
	//--------------------------------------------------------------------------

	/** Enable debug visualization */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "AI|Debug")
	bool bEnableDebugDrawing = false;

	//--------------------------------------------------------------------------
	// COMPONENTS
	//--------------------------------------------------------------------------

	/** Get team leader component from controlled pawn */
	UFUNCTION(BlueprintPure, Category = "AI|Team Leader")
	UTeamLeaderComponent* GetTeamLeaderComponent() const;

	//--------------------------------------------------------------------------
	// TEAM MANAGEMENT
	//--------------------------------------------------------------------------

	/** Get number of registered followers */
	UFUNCTION(BlueprintPure, Category = "AI|Team Leader")
	int32 GetFollowerCount() const;

	/** Is MCTS currently running? */
	UFUNCTION(BlueprintPure, Category = "AI|Team Leader")
	bool IsMCTSRunning() const;

	/** Get last MCTS decision time (ms) */
	UFUNCTION(BlueprintPure, Category = "AI|Team Leader")
	float GetLastMCTSDecisionTime() const;

	//--------------------------------------------------------------------------
	// DEBUGGING
	//--------------------------------------------------------------------------

	/** Draw debug info for team leader */
	UFUNCTION(BlueprintCallable, Category = "AI|Debug")
	void DrawDebugInfo();

private:
	/** Cached reference to team leader component */
	UPROPERTY()
	UTeamLeaderComponent* CachedTeamLeaderComponent = nullptr;

	/** Initialize components on possession */
	void InitializeComponents();
};
