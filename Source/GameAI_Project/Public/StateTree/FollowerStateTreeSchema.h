// Copyright Epic Games, Inc. All Rights Reserved.

#pragma once

#include "CoreMinimal.h"
#include "StateTreeExecutionContext.h"
#include "Observation/ObservationElement.h"
#include "RL/RLTypes.h"
#include "Team/TeamTypes.h"
#include "FollowerStateTreeSchema.generated.h"

class UFollowerAgentComponent;
class URLPolicyNetwork;
class AAIController;
class UTeamLeaderComponent;

/**
 * State Tree Shared Context for Follower Agents
 *
 * Contains all data shared between State Tree states, tasks, evaluators, and conditions.
 * Replaces both Blackboard and component polling for better performance.
 *
 * This schema is optimized for command-driven tactical execution:
 * 1. Leader issues strategic command
 * 2. State Tree transitions to appropriate state
 * 3. RL policy selects tactical action within state
 * 4. State executes action using observation data
 */
USTRUCT(BlueprintType)
struct GAMEAI_PROJECT_API FFollowerStateTreeContext
{
	GENERATED_BODY()

	//--------------------------------------------------------------------------
	// CORE COMPONENTS (Set during initialization, read-only during execution)
	//--------------------------------------------------------------------------

	/** Follower agent component reference */
	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "Components")
	TObjectPtr<UFollowerAgentComponent> FollowerComponent = nullptr;

	/** AI Controller controlling this follower */
	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "Components")
	TObjectPtr<AAIController> AIController = nullptr;

	/** Team leader component reference */
	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "Components")
	TObjectPtr<UTeamLeaderComponent> TeamLeader = nullptr;

	/** RL policy network for tactical decisions */
	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "Components")
	TObjectPtr<URLPolicyNetwork> TacticalPolicy = nullptr;

	//--------------------------------------------------------------------------
	// STRATEGIC COMMAND (Updated when leader issues new command)
	//--------------------------------------------------------------------------

	/** Current strategic command from team leader */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Command")
	FStrategicCommand CurrentCommand;

	/** Is current command still valid? */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Command")
	bool bIsCommandValid = false;

	/** Time since command was received (seconds) */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Command")
	float TimeSinceCommand = 0.0f;

	//--------------------------------------------------------------------------
	// TACTICAL STATE (Updated by RL policy and execution tasks)
	//--------------------------------------------------------------------------

	/** Last tactical action selected by RL policy */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Tactical")
	ETacticalAction CurrentTacticalAction = ETacticalAction::DefensiveHold;

	/** Time in current tactical action (seconds) */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Tactical")
	float TimeInTacticalAction = 0.0f;

	/** Progress of current action (0-1) */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Tactical")
	float ActionProgress = 0.0f;

	//--------------------------------------------------------------------------
	// OBSERVATION DATA (Updated every tick by evaluator)
	//--------------------------------------------------------------------------

	/** Current environmental observation (71 features) */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Observation")
	FObservationElement CurrentObservation;

	/** Previous observation (for experience collection) */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Observation")
	FObservationElement PreviousObservation;

	//--------------------------------------------------------------------------
	// TARGET TRACKING (Updated by perception/combat systems)
	//--------------------------------------------------------------------------

	/** Visible enemy actors */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Targets")
	TArray<TObjectPtr<AActor>> VisibleEnemies;

	/** Primary target for combat */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Targets")
	TObjectPtr<AActor> PrimaryTarget = nullptr;

	/** Distance to primary target (cm) */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Targets")
	float DistanceToPrimaryTarget = 0.0f;

	//--------------------------------------------------------------------------
	// COVER SYSTEM (Updated by tactical tasks)
	//--------------------------------------------------------------------------

	/** Current cover actor being used */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Cover")
	TObjectPtr<AActor> CurrentCover = nullptr;

	/** Is agent currently in cover? */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Cover")
	bool bInCover = false;

	/** Nearest available cover position */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Cover")
	FVector NearestCoverLocation = FVector::ZeroVector;

	/** Distance to nearest cover (cm) */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Cover")
	float DistanceToNearestCover = 0.0f;

	//--------------------------------------------------------------------------
	// MOVEMENT (Updated by movement tasks)
	//--------------------------------------------------------------------------

	/** Current movement destination */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Movement")
	FVector MovementDestination = FVector::ZeroVector;

	/** Is agent currently moving? */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Movement")
	bool bIsMoving = false;

	/** Movement speed multiplier */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Movement")
	float MovementSpeedMultiplier = 1.0f;

	//--------------------------------------------------------------------------
	// REINFORCEMENT LEARNING (Updated during tactical execution)
	//--------------------------------------------------------------------------

	/** Accumulated reward this episode */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "RL")
	float AccumulatedReward = 0.0f;

	/** Reward from last action */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "RL")
	float LastReward = 0.0f;

	/** Use RL policy vs rule-based fallback */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "RL")
	bool bUseRLPolicy = true;

	//--------------------------------------------------------------------------
	// STATE FLAGS (Updated by state transitions)
	//--------------------------------------------------------------------------

	/** Is follower alive? */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "State")
	bool bIsAlive = true;

	/** Is follower under fire? */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "State")
	bool bUnderFire = false;

	/** Has line of sight to primary target? */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "State")
	bool bHasLOS = false;

	/** Is weapon ready to fire? */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "State")
	bool bWeaponReady = true;

	//--------------------------------------------------------------------------
	// DEBUG
	//--------------------------------------------------------------------------

	/** Enable debug visualization */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Debug")
	bool bDrawDebugInfo = false;

	/** Enable verbose logging */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Debug")
	bool bEnableDebugLog = false;
};
