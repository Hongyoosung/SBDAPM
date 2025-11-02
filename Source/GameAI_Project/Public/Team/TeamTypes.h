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
 * Strategic command types issued by team leader
 */
UENUM(BlueprintType)
enum class EStrategicCommandType : uint8
{
	None			UMETA(DisplayName = "None"),
	
	// Offensive commands
	Assault         UMETA(DisplayName = "Assault - Aggressive attack"),
	Flank           UMETA(DisplayName = "Flank - Attack from side"),
	Suppress        UMETA(DisplayName = "Suppress - Suppressive fire"),
	Charge          UMETA(DisplayName = "Charge - Rush enemy"),

	// Defensive commands
	StayAlert       UMETA(DisplayName = "Stay Alert - Defensive posture"),
	HoldPosition    UMETA(DisplayName = "Hold Position - Defend location"),
	TakeCover       UMETA(DisplayName = "Take Cover - Find cover"),
	Fortify         UMETA(DisplayName = "Fortify - Strengthen position"),

	// Support commands
	RescueAlly      UMETA(DisplayName = "Rescue - Save ally"),
	ProvideSupport  UMETA(DisplayName = "Support - Help teammate"),
	Regroup         UMETA(DisplayName = "Regroup - Return to formation"),
	ShareAmmo       UMETA(DisplayName = "Share Ammo - Resupply teammate"),

	// Movement commands
	Advance         UMETA(DisplayName = "Advance - Move forward"),
	Retreat         UMETA(DisplayName = "Retreat - Fall back"),
	Patrol          UMETA(DisplayName = "Patrol - Guard area"),
	MoveTo          UMETA(DisplayName = "Move To - Navigate to position"),
	Follow          UMETA(DisplayName = "Follow - Follow target"),

	// Special commands
	Investigate     UMETA(DisplayName = "Investigate - Check area"),
	Distract        UMETA(DisplayName = "Distract - Draw attention"),
	Stealth         UMETA(DisplayName = "Stealth - Move quietly"),
	Idle            UMETA(DisplayName = "Idle - No orders")
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
 * Strategic command with parameters
 */
USTRUCT(BlueprintType)
struct GAMEAI_PROJECT_API FStrategicCommand
{
	GENERATED_BODY()

	/** Command type */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Command")
	EStrategicCommandType CommandType = EStrategicCommandType::Idle;

	/** Target location (if applicable) */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Command")
	FVector TargetLocation = FVector::ZeroVector;

	/** Target actor (enemy, ally, objective) */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Command")
	AActor* TargetActor = nullptr;

	/** Priority level (0-10, higher = more urgent) */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Command")
	int32 Priority = 5;

	/** Expected duration (0 = indefinite) */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Command")
	float ExpectedDuration = 0.0f;

	/** Formation offset (for coordinated movements) */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Command")
	FVector FormationOffset = FVector::ZeroVector;

	/** Additional parameters (key-value pairs) */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Command")
	TMap<FString, FString> Parameters;

	/** Timestamp when issued */
	UPROPERTY(BlueprintReadOnly, Category = "Command")
	float IssuedTime = 0.0f;

	/** Is this command completed? */
	UPROPERTY(BlueprintReadWrite, Category = "Command")
	bool bCompleted = false;

	/** Command execution progress (0-1) */
	UPROPERTY(BlueprintReadWrite, Category = "Command")
	float Progress = 0.0f;

	// Constructor
	FStrategicCommand()
	{
		IssuedTime = FPlatformTime::Seconds();
	}
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
