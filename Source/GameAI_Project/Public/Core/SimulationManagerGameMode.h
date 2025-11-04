#pragma once

#include "CoreMinimal.h"
#include "GameFramework/GameModeBase.h"
#include "Team/TeamTypes.h"
#include "SimulationManagerGameMode.generated.h"

class UTeamLeaderComponent;

/**
 * Team registration info
 */
USTRUCT(BlueprintType)
struct FTeamInfo
{
	GENERATED_BODY()

	/** Team ID (unique identifier) */
	UPROPERTY(BlueprintReadWrite, Category = "Team")
	int32 TeamID = 0;

	/** Team name */
	UPROPERTY(BlueprintReadWrite, Category = "Team")
	FString TeamName;

	/** Team leader component */
	UPROPERTY(BlueprintReadWrite, Category = "Team")
	UTeamLeaderComponent* TeamLeader = nullptr;

	/** Team color */
	UPROPERTY(BlueprintReadWrite, Category = "Team")
	FLinearColor TeamColor = FLinearColor::White;

	/** All actors belonging to this team */
	UPROPERTY(BlueprintReadWrite, Category = "Team")
	TArray<AActor*> TeamMembers;

	/** IDs of enemy teams */
	UPROPERTY(BlueprintReadWrite, Category = "Team")
	TSet<int32> EnemyTeamIDs;

	/** Is team active in simulation? */
	UPROPERTY(BlueprintReadWrite, Category = "Team")
	bool bIsActive = true;
};

/**
 * Simulation statistics
 */
USTRUCT(BlueprintType)
struct FSimulationStats
{
	GENERATED_BODY()

	UPROPERTY(BlueprintReadOnly, Category = "Stats")
	int32 TotalTeams = 0;

	UPROPERTY(BlueprintReadOnly, Category = "Stats")
	int32 ActiveTeams = 0;

	UPROPERTY(BlueprintReadOnly, Category = "Stats")
	float SimulationTime = 0.0f;

	UPROPERTY(BlueprintReadOnly, Category = "Stats")
	int32 TotalAgents = 0;

	UPROPERTY(BlueprintReadOnly, Category = "Stats")
	int32 AliveAgents = 0;
};

/**
 * Simulation Manager GameMode
 *
 * Manages team-based AI simulation with multi-team support.
 * Handles team registration, enemy team tracking, and simulation state.
 *
 * Key Features:
 * - Register multiple teams with unique IDs
 * - Define enemy relationships between teams
 * - Query team information and enemy teams
 * - Track simulation statistics
 * - Broadcast team events
 *
 * Usage:
 * 1. Set this as your GameMode in World Settings
 * 2. Register teams using RegisterTeam() during BeginPlay
 * 3. Set enemy teams using SetEnemyTeams() or AddEnemyTeam()
 * 4. Query teams using GetTeamInfo(), GetEnemyTeams(), etc.
 */
UCLASS()
class GAMEAI_PROJECT_API ASimulationManagerGameMode : public AGameModeBase
{
	GENERATED_BODY()

public:
	ASimulationManagerGameMode();

	virtual void BeginPlay() override;
	virtual void Tick(float DeltaTime) override;

	//--------------------------------------------------------------------------
	// TEAM REGISTRATION
	//--------------------------------------------------------------------------

	/**
	 * Register a team with the simulation manager
	 * @param TeamID - Unique team identifier
	 * @param TeamLeader - Team leader component
	 * @param TeamName - Display name for the team
	 * @param TeamColor - Team color (for visualization)
	 * @return true if registration succeeded
	 */
	UFUNCTION(BlueprintCallable, Category = "Simulation|Teams")
	bool RegisterTeam(
		int32 TeamID,
		UTeamLeaderComponent* TeamLeader,
		const FString& TeamName = TEXT("Team"),
		FLinearColor TeamColor = FLinearColor::White
	);

	/**
	 * Unregister a team
	 * @param TeamID - Team to unregister
	 */
	UFUNCTION(BlueprintCallable, Category = "Simulation|Teams")
	void UnregisterTeam(int32 TeamID);

	/**
	 * Register an agent as a member of a team
	 * @param TeamID - Team ID
	 * @param Agent - Agent actor to register
	 */
	UFUNCTION(BlueprintCallable, Category = "Simulation|Teams")
	bool RegisterTeamMember(int32 TeamID, AActor* Agent);

	/**
	 * Unregister an agent from a team
	 * @param TeamID - Team ID
	 * @param Agent - Agent to unregister
	 */
	UFUNCTION(BlueprintCallable, Category = "Simulation|Teams")
	void UnregisterTeamMember(int32 TeamID, AActor* Agent);

	/**
	 * Get team ID for an actor
	 * @param Agent - Actor to query
	 * @return Team ID (returns -1 if not found)
	 */
	UFUNCTION(BlueprintPure, Category = "Simulation|Teams")
	int32 GetTeamIDForActor(AActor* Agent) const;

	//--------------------------------------------------------------------------
	// ENEMY TEAM MANAGEMENT
	//--------------------------------------------------------------------------

	/**
	 * Set enemy teams for a team (replaces existing enemies)
	 * @param TeamID - Team to configure
	 * @param EnemyTeamIDs - Array of enemy team IDs
	 */
	UFUNCTION(BlueprintCallable, Category = "Simulation|Enemies")
	void SetEnemyTeams(int32 TeamID, const TArray<int32>& EnemyTeamIDs);

	/**
	 * Add a single enemy team
	 * @param TeamID - Team to configure
	 * @param EnemyTeamID - Enemy team to add
	 */
	UFUNCTION(BlueprintCallable, Category = "Simulation|Enemies")
	void AddEnemyTeam(int32 TeamID, int32 EnemyTeamID);

	/**
	 * Remove an enemy team
	 * @param TeamID - Team to configure
	 * @param EnemyTeamID - Enemy team to remove
	 */
	UFUNCTION(BlueprintCallable, Category = "Simulation|Enemies")
	void RemoveEnemyTeam(int32 TeamID, int32 EnemyTeamID);

	/**
	 * Set bidirectional enemy relationship (both teams become enemies of each other)
	 * @param TeamID1 - First team
	 * @param TeamID2 - Second team
	 */
	UFUNCTION(BlueprintCallable, Category = "Simulation|Enemies")
	void SetMutualEnemies(int32 TeamID1, int32 TeamID2);

	/**
	 * Check if two teams are enemies
	 * @param TeamID1 - First team
	 * @param TeamID2 - Second team
	 * @return true if teams are enemies
	 */
	UFUNCTION(BlueprintPure, Category = "Simulation|Enemies")
	bool AreTeamsEnemies(int32 TeamID1, int32 TeamID2) const;

	/**
	 * Check if two actors are on enemy teams
	 * @param Actor1 - First actor
	 * @param Actor2 - Second actor
	 * @return true if actors are on enemy teams
	 */
	UFUNCTION(BlueprintPure, Category = "Simulation|Enemies")
	bool AreActorsEnemies(AActor* Actor1, AActor* Actor2) const;

	/**
	 * Get all enemy team IDs for a team
	 * @param TeamID - Team to query
	 * @return Array of enemy team IDs
	 */
	UFUNCTION(BlueprintPure, Category = "Simulation|Enemies")
	TArray<int32> GetEnemyTeamIDs(int32 TeamID) const;

	/**
	 * Get all enemy actors for a team
	 * @param TeamID - Team to query
	 * @return Array of enemy actors
	 */
	UFUNCTION(BlueprintCallable, Category = "Simulation|Enemies")
	TArray<AActor*> GetEnemyActors(int32 TeamID) const;

	//--------------------------------------------------------------------------
	// TEAM QUERIES
	//--------------------------------------------------------------------------

	/**
	 * Get team info
	 * @param TeamID - Team to query
	 * @param OutTeamInfo - Output team info
	 * @return true if team exists
	 */
	UFUNCTION(BlueprintPure, Category = "Simulation|Teams")
	bool GetTeamInfo(int32 TeamID, FTeamInfo& OutTeamInfo) const;

	/**
	 * Get team leader component
	 * @param TeamID - Team to query
	 * @return Team leader component (nullptr if not found)
	 */
	UFUNCTION(BlueprintPure, Category = "Simulation|Teams")
	UTeamLeaderComponent* GetTeamLeader(int32 TeamID) const;

	/**
	 * Get all team members
	 * @param TeamID - Team to query
	 * @return Array of team member actors
	 */
	UFUNCTION(BlueprintPure, Category = "Simulation|Teams")
	TArray<AActor*> GetTeamMembers(int32 TeamID) const;

	/**
	 * Get all registered team IDs
	 * @return Array of all team IDs
	 */
	UFUNCTION(BlueprintPure, Category = "Simulation|Teams")
	TArray<int32> GetAllTeamIDs() const;

	/**
	 * Check if team exists
	 * @param TeamID - Team to check
	 * @return true if team is registered
	 */
	UFUNCTION(BlueprintPure, Category = "Simulation|Teams")
	bool IsTeamRegistered(int32 TeamID) const;

	//--------------------------------------------------------------------------
	// SIMULATION CONTROL
	//--------------------------------------------------------------------------

	/**
	 * Start simulation (enables all teams)
	 */
	UFUNCTION(BlueprintCallable, Category = "Simulation|Control")
	void StartSimulation();

	/**
	 * Stop simulation (disables all teams)
	 */
	UFUNCTION(BlueprintCallable, Category = "Simulation|Control")
	void StopSimulation();

	/**
	 * Reset simulation (clears all teams and stats)
	 */
	UFUNCTION(BlueprintCallable, Category = "Simulation|Control")
	void ResetSimulation();

	/**
	 * Is simulation running?
	 */
	UFUNCTION(BlueprintPure, Category = "Simulation|Control")
	bool IsSimulationRunning() const { return bSimulationRunning; }

	/**
	 * Get simulation statistics
	 */
	UFUNCTION(BlueprintPure, Category = "Simulation|Stats")
	FSimulationStats GetSimulationStats() const;

	//--------------------------------------------------------------------------
	// CONFIGURATION
	//--------------------------------------------------------------------------

	/** Auto-start simulation on BeginPlay */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Simulation|Config")
	bool bAutoStartSimulation = true;

	/** Enable debug visualization */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Simulation|Debug")
	bool bDrawDebugInfo = false;

	/** Debug draw update interval */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Simulation|Debug")
	float DebugDrawInterval = 1.0f;

private:
	/** Registered teams */
	UPROPERTY()
	TMap<int32, FTeamInfo> RegisteredTeams;

	/** Actor to team ID mapping (for fast lookup) */
	UPROPERTY()
	TMap<AActor*, int32> ActorToTeamMap;

	/** Is simulation running? */
	bool bSimulationRunning = false;

	/** Simulation start time */
	float SimulationStartTime = 0.0f;

	/** Last debug draw time */
	float LastDebugDrawTime = 0.0f;

	/** Update statistics */
	void UpdateStatistics();

	/** Draw debug information */
	void DrawDebugInformation();
};
