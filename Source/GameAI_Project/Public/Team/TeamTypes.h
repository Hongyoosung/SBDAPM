#pragma once

#include "CoreMinimal.h"
#include "TeamTypes.generated.h"

/**
 * Strategic events that trigger MCTS decision-making
 */
UENUM(BlueprintType)
enum class EStrategicEvent : uint8
{
	// Combat events
	EnemyEncounter      UMETA(DisplayName = "Enemy Encountered"),
	AllyKilled          UMETA(DisplayName = "Ally Killed"),
	EnemyEliminated     UMETA(DisplayName = "Enemy Eliminated"),
	AllyRescueSignal    UMETA(DisplayName = "Ally Rescue Signal"),
	AllyUnderFire       UMETA(DisplayName = "Ally Under Heavy Fire"),
	EnemySpotted		UMETA(DisplayName = "Enemy Spotted"),
	UnderFire			UMETA(DisplayName = "Under Heavy Fire"),
	EnemyKilled			UMETA(DisplayName = "Enemy Killed"),

	// Environmental events
	EnteredDangerZone   UMETA(DisplayName = "Entered Suspected Enemy Zone"),
	ObjectiveSpotted    UMETA(DisplayName = "Objective Spotted"),
	AmbushDetected      UMETA(DisplayName = "Ambush Detected"),
	CoverCompromised    UMETA(DisplayName = "Cover Compromised"),

	// Team status events
	LowTeamHealth       UMETA(DisplayName = "Low Team Health"),
	LowTeamAmmo         UMETA(DisplayName = "Low Team Ammunition"),
	FormationBroken     UMETA(DisplayName = "Formation Broken"),
	TeamRegrouped       UMETA(DisplayName = "Team Regrouped"),

	// Mission events
	ObjectiveComplete   UMETA(DisplayName = "Objective Completed"),
	ObjectiveFailed     UMETA(DisplayName = "Objective Failed"),
	ReinforcementsArrived UMETA(DisplayName = "Reinforcements Arrived"),
	TimeRunningOut      UMETA(DisplayName = "Time Running Out"),

	// Custom
	Custom              UMETA(DisplayName = "Custom Event")
};

/**
 * Event priority levels (affects MCTS trigger threshold)
 */
UENUM(BlueprintType)
enum class EEventPriority : uint8
{
	None = 0        UMETA(DisplayName = "None"),
	Low = 1         UMETA(DisplayName = "Low Priority"),
	Medium = 5      UMETA(DisplayName = "Medium Priority"),
	High = 8        UMETA(DisplayName = "High Priority"),
	Critical = 10   UMETA(DisplayName = "Critical Priority")
};


/**
 * Event context information
 */
USTRUCT(BlueprintType)
struct GAMEAI_PROJECT_API FStrategicEventContext
{
	GENERATED_BODY()

	/** Event type */
	UPROPERTY(BlueprintReadWrite, Category = "Event")
	EStrategicEvent EventType = EStrategicEvent::Custom;

	/** Actor that triggered event (if applicable) */
	UPROPERTY(BlueprintReadWrite, Category = "Event")
	AActor* Instigator = nullptr;

	/** Location where event occurred */
	UPROPERTY(BlueprintReadWrite, Category = "Event")
	FVector Location = FVector::ZeroVector;

	/** Event priority (affects MCTS trigger threshold) */
	UPROPERTY(BlueprintReadWrite, Category = "Event")
	int32 Priority = 5;

	/** Event timestamp */
	UPROPERTY(BlueprintReadOnly, Category = "Event")
	float Timestamp = 0.0f;

	/** Additional context data */
	UPROPERTY(BlueprintReadWrite, Category = "Event")
	TMap<FString, FString> ContextData;

	FStrategicEventContext()
	{
		Timestamp = FPlatformTime::Seconds();
	}
};

/**
 * Team performance metrics
 */
USTRUCT(BlueprintType)
struct GAMEAI_PROJECT_API FTeamMetrics
{
	GENERATED_BODY()

public:

	UPROPERTY(BlueprintReadOnly, Category = "Metrics")
	int32 TotalFollowers = 0;

	UPROPERTY(BlueprintReadOnly, Category = "Metrics")
	int32 AliveFollowers = 0;

	UPROPERTY(BlueprintReadOnly, Category = "Metrics")
	float AverageHealth = 100.0f;

	UPROPERTY(BlueprintReadOnly, Category = "Metrics")
	int32 EnemiesEliminated = 0;

	UPROPERTY(BlueprintReadOnly, Category = "Metrics")
	int32 FollowersLost = 0;

	UPROPERTY(BlueprintReadOnly, Category = "Metrics")
	float KillDeathRatio = 0.0f;

	UPROPERTY(BlueprintReadOnly, Category = "Metrics")
	int32 CommandsIssued = 0;

	UPROPERTY(BlueprintReadOnly, Category = "Metrics")
	float MCTSExecutionTime = 0.0f;
};

/**
 * Follower state enum (command-driven FSM states)
 */
UENUM(BlueprintType)
enum class EFollowerState : uint8
{
	Idle        UMETA(DisplayName = "Idle - No orders"),
	Assault     UMETA(DisplayName = "Assault - Offensive actions"),
	Defend      UMETA(DisplayName = "Defend - Defensive actions"),
	Support     UMETA(DisplayName = "Support - Support actions"),
	Move        UMETA(DisplayName = "Move - Movement actions"),
	Retreat     UMETA(DisplayName = "Retreat - Retreat actions"),
	Dead        UMETA(DisplayName = "Dead - Terminal state")
};
