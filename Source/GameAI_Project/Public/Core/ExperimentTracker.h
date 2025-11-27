// ExperimentTracker.h - Experiment metrics collection and CSV logging
// Part of v3.0 Academic Publication Infrastructure

#pragma once

#include "CoreMinimal.h"
#include "Components/ActorComponent.h"
#include "ExperimentTracker.generated.h"

// Forward declarations
class UTeamLeaderComponent;
class UFollowerAgentComponent;

/**
 * Episode metadata
 */
USTRUCT(BlueprintType)
struct FEpisodeMetadata
{
	GENERATED_BODY()

	UPROPERTY(BlueprintReadWrite)
	FString ExperimentName = TEXT("Unnamed");

	UPROPERTY(BlueprintReadWrite)
	FString MapName = TEXT("Unknown");

	UPROPERTY(BlueprintReadWrite)
	FString Timestamp = TEXT("");

	UPROPERTY(BlueprintReadWrite)
	int32 EpisodeNumber = 0;

	UPROPERTY(BlueprintReadWrite)
	FString TeamAlphaAIType = TEXT("SBDAPM_v3.0");

	UPROPERTY(BlueprintReadWrite)
	FString TeamBravoAIType = TEXT("SBDAPM_v3.0");
};

/**
 * Per-team metrics collected during episode
 */
USTRUCT(BlueprintType)
struct FTeamEpisodeMetrics
{
	GENERATED_BODY()

	// Basic stats
	UPROPERTY(BlueprintReadWrite)
	FString TeamName = TEXT("Unknown");

	UPROPERTY(BlueprintReadWrite)
	int32 InitialAgents = 0;

	UPROPERTY(BlueprintReadWrite)
	int32 SurvivingAgents = 0;

	UPROPERTY(BlueprintReadWrite)
	float AverageFinalHealth = 0.0f;

	// Combat stats
	UPROPERTY(BlueprintReadWrite)
	int32 TotalKills = 0;

	UPROPERTY(BlueprintReadWrite)
	int32 TotalDeaths = 0;

	UPROPERTY(BlueprintReadWrite)
	float KillDeathRatio = 0.0f;

	UPROPERTY(BlueprintReadWrite)
	float TotalDamageDealt = 0.0f;

	UPROPERTY(BlueprintReadWrite)
	float TotalDamageTaken = 0.0f;

	// Coordination metrics
	UPROPERTY(BlueprintReadWrite)
	int32 CoordinatedKills = 0; // Kills during strategic commands

	UPROPERTY(BlueprintReadWrite)
	int32 CombinedFireEvents = 0; // Multiple agents targeting same enemy

	UPROPERTY(BlueprintReadWrite)
	float AverageFormationCoherence = 0.0f;

	// Objective metrics
	UPROPERTY(BlueprintReadWrite)
	int32 ObjectivesCompleted = 0;

	UPROPERTY(BlueprintReadWrite)
	int32 ObjectivesFailed = 0;

	UPROPERTY(BlueprintReadWrite)
	float AverageObjectiveCompletionTime = 0.0f;

	// MCTS metrics
	UPROPERTY(BlueprintReadWrite)
	int32 MCTSExecutions = 0;

	UPROPERTY(BlueprintReadWrite)
	float AverageMCTSLatency = 0.0f; // milliseconds

	UPROPERTY(BlueprintReadWrite)
	float AverageValueVariance = 0.0f;

	UPROPERTY(BlueprintReadWrite)
	float AveragePolicyEntropy = 0.0f;

	// Reward accumulation
	UPROPERTY(BlueprintReadWrite)
	float TotalReward = 0.0f;

	UPROPERTY(BlueprintReadWrite)
	float IndividualReward = 0.0f;

	UPROPERTY(BlueprintReadWrite)
	float CoordinationReward = 0.0f;

	UPROPERTY(BlueprintReadWrite)
	float StrategicReward = 0.0f;
};

/**
 * Complete episode results
 */
USTRUCT(BlueprintType)
struct FEpisodeResults
{
	GENERATED_BODY()

	UPROPERTY(BlueprintReadWrite)
	FEpisodeMetadata Metadata;

	UPROPERTY(BlueprintReadWrite)
	FTeamEpisodeMetrics TeamAlpha;

	UPROPERTY(BlueprintReadWrite)
	FTeamEpisodeMetrics TeamBravo;

	// Episode outcome
	UPROPERTY(BlueprintReadWrite)
	FString WinnerTeam = TEXT("Draw"); // "TeamAlpha", "TeamBravo", "Draw"

	UPROPERTY(BlueprintReadWrite)
	FString WinCondition = TEXT("None"); // "Elimination", "Objective", "Timeout"

	UPROPERTY(BlueprintReadWrite)
	float EpisodeDuration = 0.0f; // seconds

	UPROPERTY(BlueprintReadWrite)
	int32 TotalCombatEngagements = 0;

	// Performance metrics
	UPROPERTY(BlueprintReadWrite)
	float AverageFrameRate = 0.0f;

	UPROPERTY(BlueprintReadWrite)
	float PeakMemoryUsageMB = 0.0f;
};

/**
 * Experiment Tracker Component
 *
 * Collects metrics during episodes and exports to CSV for analysis.
 * Attach to GameMode for automatic tracking.
 *
 * Features:
 * - Per-team metric collection
 * - CSV export with headers
 * - Batch experiment support
 * - Real-time statistics
 *
 * Usage:
 * 1. Add to GameMode Blueprint as component
 * 2. Call StartEpisode() when match begins
 * 3. Call RecordEvent() during gameplay
 * 4. Call EndEpisode() when match ends
 * 5. CSV auto-exported to Saved/Experiments/
 */
UCLASS(ClassGroup=(Metrics), meta=(BlueprintSpawnableComponent))
class GAMEAI_PROJECT_API UExperimentTracker : public UActorComponent
{
	GENERATED_BODY()

public:
	UExperimentTracker();

	virtual void BeginPlay() override;
	virtual void TickComponent(float DeltaTime, ELevelTick TickType,
		FActorComponentTickFunction* ThisTickFunction) override;

	//--------------------------------------------------------------------------
	// EPISODE LIFECYCLE
	//--------------------------------------------------------------------------

	/** Start new episode tracking */
	UFUNCTION(BlueprintCallable, Category = "Experiment")
	void StartEpisode(const FEpisodeMetadata& Metadata);

	/** End current episode and export data */
	UFUNCTION(BlueprintCallable, Category = "Experiment")
	void EndEpisode(const FString& WinnerTeam, const FString& WinCondition);

	/** Cancel current episode without logging */
	UFUNCTION(BlueprintCallable, Category = "Experiment")
	void CancelEpisode();

	//--------------------------------------------------------------------------
	// EVENT RECORDING
	//--------------------------------------------------------------------------

	/** Record agent kill event */
	UFUNCTION(BlueprintCallable, Category = "Experiment")
	void RecordKill(const FString& TeamName, AActor* Killer, AActor* Victim, bool bCoordinated);

	/** Record agent death event */
	UFUNCTION(BlueprintCallable, Category = "Experiment")
	void RecordDeath(const FString& TeamName, AActor* Victim);

	/** Record damage event */
	UFUNCTION(BlueprintCallable, Category = "Experiment")
	void RecordDamage(const FString& TeamName, float Damage, bool bDealt);

	/** Record objective event */
	UFUNCTION(BlueprintCallable, Category = "Experiment")
	void RecordObjectiveEvent(const FString& TeamName, bool bCompleted, float CompletionTime);

	/** Record MCTS execution */
	UFUNCTION(BlueprintCallable, Category = "Experiment")
	void RecordMCTSExecution(const FString& TeamName, float Latency, float ValueVariance, float PolicyEntropy);

	/** Record combined fire event (coordination) */
	UFUNCTION(BlueprintCallable, Category = "Experiment")
	void RecordCombinedFire(const FString& TeamName);

	/** Record formation coherence sample */
	UFUNCTION(BlueprintCallable, Category = "Experiment")
	void RecordFormationCoherence(const FString& TeamName, float Coherence);

	/** Record reward feedback */
	UFUNCTION(BlueprintCallable, Category = "Experiment")
	void RecordReward(const FString& TeamName, float Reward, const FString& RewardType);

	//--------------------------------------------------------------------------
	// DATA EXPORT
	//--------------------------------------------------------------------------

	/** Export current episode to CSV */
	UFUNCTION(BlueprintCallable, Category = "Experiment")
	bool ExportToCSV();

	/** Export all episodes to single CSV file */
	UFUNCTION(BlueprintCallable, Category = "Experiment")
	bool ExportAllToCSV(const FString& FilePath);

	/** Export summary statistics (aggregated over all episodes) */
	UFUNCTION(BlueprintCallable, Category = "Experiment")
	bool ExportSummary(const FString& FilePath);

	//--------------------------------------------------------------------------
	// QUERIES
	//--------------------------------------------------------------------------

	/** Get current episode results */
	UFUNCTION(BlueprintPure, Category = "Experiment")
	FEpisodeResults GetCurrentResults() const { return CurrentResults; }

	/** Get episode count */
	UFUNCTION(BlueprintPure, Category = "Experiment")
	int32 GetEpisodeCount() const { return CompletedEpisodes.Num(); }

	/** Get win rate for team */
	UFUNCTION(BlueprintCallable, Category = "Experiment")
	float GetTeamWinRate(const FString& TeamName) const;

	/** Get average K/D ratio for team */
	UFUNCTION(BlueprintCallable, Category = "Experiment")
	float GetAverageKDRatio(const FString& TeamName) const;

	/** Is episode currently running? */
	UFUNCTION(BlueprintPure, Category = "Experiment")
	bool IsEpisodeActive() const { return bEpisodeActive; }

	//--------------------------------------------------------------------------
	// AUTO-DISCOVERY (for easy setup)
	//--------------------------------------------------------------------------

	/** Auto-register team leader for tracking */
	UFUNCTION(BlueprintCallable, Category = "Experiment")
	void RegisterTeamLeader(UTeamLeaderComponent* TeamLeader);

	/** Auto-register follower for tracking */
	UFUNCTION(BlueprintCallable, Category = "Experiment")
	void RegisterFollower(UFollowerAgentComponent* Follower);

	//--------------------------------------------------------------------------
	// CONFIGURATION
	//--------------------------------------------------------------------------

	/** Experiment name (used in CSV filename) */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Experiment|Config")
	FString ExperimentName = TEXT("SelfPlay_Training");

	/** CSV output directory (relative to project Saved/ folder) */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Experiment|Config")
	FString OutputDirectory = TEXT("Experiments");

	/** Auto-export CSV after each episode */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Experiment|Config")
	bool bAutoExportCSV = true;

	/** Include detailed MCTS metrics in CSV */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Experiment|Config")
	bool bTrackMCTSMetrics = true;

	/** Include reward breakdown in CSV */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Experiment|Config")
	bool bTrackRewardBreakdown = true;

	/** Sample formation coherence every N seconds */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Experiment|Config")
	float FormationSampleInterval = 2.0f;

private:
	//--------------------------------------------------------------------------
	// INTERNAL STATE
	//--------------------------------------------------------------------------

	/** Is episode currently active? */
	bool bEpisodeActive = false;

	/** Episode start time */
	float EpisodeStartTime = 0.0f;

	/** Current episode results */
	FEpisodeResults CurrentResults;

	/** All completed episodes */
	TArray<FEpisodeResults> CompletedEpisodes;

	/** Tracked team leaders */
	TMap<FString, UTeamLeaderComponent*> RegisteredLeaders;

	/** Formation coherence samples (for averaging) */
	TMap<FString, TArray<float>> FormationSamples;

	/** Objective completion times (for averaging) */
	TMap<FString, TArray<float>> ObjectiveCompletionTimes;

	/** Frame rate samples */
	TArray<float> FrameRateSamples;

	/** Time since last formation sample */
	float TimeSinceFormationSample = 0.0f;

	//--------------------------------------------------------------------------
	// HELPERS
	//--------------------------------------------------------------------------

	/** Get team metrics by name (creates if not exists) */
	FTeamEpisodeMetrics& GetTeamMetrics(const FString& TeamName);

	/** Generate CSV filename */
	FString GenerateCSVFilename() const;

	/** Generate CSV header row */
	FString GenerateCSVHeader() const;

	/** Convert episode results to CSV row */
	FString EpisodeToCSVRow(const FEpisodeResults& Results) const;

	/** Calculate final metrics (called at episode end) */
	void CalculateFinalMetrics();

	/** Collect metrics from registered team leaders */
	void CollectTeamLeaderMetrics();

	/** Collect metrics from followers */
	void CollectFollowerMetrics();
};
