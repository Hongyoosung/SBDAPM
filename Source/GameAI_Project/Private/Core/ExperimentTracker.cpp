// ExperimentTracker.cpp - Implementation

#include "Core/ExperimentTracker.h"
#include "Team/TeamLeaderComponent.h"
#include "Team/FollowerAgentComponent.h"
#include "RL/RewardCalculator.h"
#include "AI/MCTS/MCTS.h"
#include "Combat/HealthComponent.h"
#include "Misc/FileHelper.h"
#include "Misc/Paths.h"
#include "HAL/PlatformFileManager.h"

UExperimentTracker::UExperimentTracker()
{
	PrimaryComponentTick.bCanEverTick = true;
	PrimaryComponentTick.TickInterval = 0.5f; // Update every 500ms
}

void UExperimentTracker::BeginPlay()
{
	Super::BeginPlay();

	UE_LOG(LogTemp, Log, TEXT("ExperimentTracker: Initialized for experiment '%s'"), *ExperimentName);

	// Ensure output directory exists
	FString OutputPath = FPaths::ProjectSavedDir() / OutputDirectory;
	IPlatformFile& PlatformFile = FPlatformFileManager::Get().GetPlatformFile();
	if (!PlatformFile.DirectoryExists(*OutputPath))
	{
		PlatformFile.CreateDirectory(*OutputPath);
		UE_LOG(LogTemp, Log, TEXT("ExperimentTracker: Created output directory '%s'"), *OutputPath);
	}
}

void UExperimentTracker::TickComponent(float DeltaTime, ELevelTick TickType,
	FActorComponentTickFunction* ThisTickFunction)
{
	Super::TickComponent(DeltaTime, TickType, ThisTickFunction);

	if (!bEpisodeActive) return;

	// Sample frame rate
	float CurrentFPS = 1.0f / DeltaTime;
	FrameRateSamples.Add(CurrentFPS);

	// Sample formation coherence
	if (bTrackMCTSMetrics)
	{
		TimeSinceFormationSample += DeltaTime;
		if (TimeSinceFormationSample >= FormationSampleInterval)
		{
			TimeSinceFormationSample = 0.0f;
			CollectTeamLeaderMetrics();
		}
	}
}

//------------------------------------------------------------------------------
// EPISODE LIFECYCLE
//------------------------------------------------------------------------------

void UExperimentTracker::StartEpisode(const FEpisodeMetadata& Metadata)
{
	if (bEpisodeActive)
	{
		UE_LOG(LogTemp, Warning, TEXT("ExperimentTracker: Episode already active, canceling previous"));
		CancelEpisode();
	}

	// Reset state
	CurrentResults = FEpisodeResults();
	CurrentResults.Metadata = Metadata;
	CurrentResults.Metadata.EpisodeNumber = CompletedEpisodes.Num() + 1;
	CurrentResults.Metadata.Timestamp = FDateTime::Now().ToString(TEXT("%Y-%m-%d_%H-%M-%S"));

	FormationSamples.Empty();
	ObjectiveCompletionTimes.Empty();
	FrameRateSamples.Empty();

	bEpisodeActive = true;
	EpisodeStartTime = GetWorld()->GetTimeSeconds();

	UE_LOG(LogTemp, Warning, TEXT("📊 [EXPERIMENT] Episode #%d STARTED: %s vs %s on %s"),
		CurrentResults.Metadata.EpisodeNumber,
		*CurrentResults.Metadata.TeamAlphaAIType,
		*CurrentResults.Metadata.TeamBravoAIType,
		*CurrentResults.Metadata.MapName);
}

void UExperimentTracker::EndEpisode(const FString& WinnerTeam, const FString& WinCondition)
{
	if (!bEpisodeActive)
	{
		UE_LOG(LogTemp, Warning, TEXT("ExperimentTracker: No active episode to end"));
		return;
	}

	// Calculate episode duration
	float CurrentTime = GetWorld()->GetTimeSeconds();
	CurrentResults.EpisodeDuration = CurrentTime - EpisodeStartTime;

	// Set outcome
	CurrentResults.WinnerTeam = WinnerTeam;
	CurrentResults.WinCondition = WinCondition;

	// Collect final metrics
	CalculateFinalMetrics();

	// Log results
	UE_LOG(LogTemp, Warning, TEXT("📊 [EXPERIMENT] Episode #%d ENDED: Winner=%s (%s), Duration=%.1fs"),
		CurrentResults.Metadata.EpisodeNumber,
		*WinnerTeam,
		*WinCondition,
		CurrentResults.EpisodeDuration);

	UE_LOG(LogTemp, Display, TEXT("   Team Alpha: %d kills, %d deaths, %.2f K/D, %d objectives"),
		CurrentResults.TeamAlpha.TotalKills,
		CurrentResults.TeamAlpha.TotalDeaths,
		CurrentResults.TeamAlpha.KillDeathRatio,
		CurrentResults.TeamAlpha.ObjectivesCompleted);

	UE_LOG(LogTemp, Display, TEXT("   Team Bravo: %d kills, %d deaths, %.2f K/D, %d objectives"),
		CurrentResults.TeamBravo.TotalKills,
		CurrentResults.TeamBravo.TotalDeaths,
		CurrentResults.TeamBravo.KillDeathRatio,
		CurrentResults.TeamBravo.ObjectivesCompleted);

	// Store completed episode
	CompletedEpisodes.Add(CurrentResults);

	// Auto-export if enabled
	if (bAutoExportCSV)
	{
		ExportToCSV();
	}

	bEpisodeActive = false;
}

void UExperimentTracker::CancelEpisode()
{
	if (bEpisodeActive)
	{
		UE_LOG(LogTemp, Warning, TEXT("ExperimentTracker: Episode #%d canceled"),
			CurrentResults.Metadata.EpisodeNumber);
		bEpisodeActive = false;
	}
}

//------------------------------------------------------------------------------
// EVENT RECORDING
//------------------------------------------------------------------------------

void UExperimentTracker::RecordKill(const FString& TeamName, AActor* Killer, AActor* Victim, bool bCoordinated)
{
	if (!bEpisodeActive) return;

	FTeamEpisodeMetrics& Metrics = GetTeamMetrics(TeamName);
	Metrics.TotalKills++;

	if (bCoordinated)
	{
		Metrics.CoordinatedKills++;
	}

	UE_LOG(LogTemp, Verbose, TEXT("ExperimentTracker: %s - Kill recorded (Total: %d, Coordinated: %s)"),
		*TeamName, Metrics.TotalKills, bCoordinated ? TEXT("Yes") : TEXT("No"));
}

void UExperimentTracker::RecordDeath(const FString& TeamName, AActor* Victim)
{
	if (!bEpisodeActive) return;

	FTeamEpisodeMetrics& Metrics = GetTeamMetrics(TeamName);
	Metrics.TotalDeaths++;

	UE_LOG(LogTemp, Verbose, TEXT("ExperimentTracker: %s - Death recorded (Total: %d)"),
		*TeamName, Metrics.TotalDeaths);
}

void UExperimentTracker::RecordDamage(const FString& TeamName, float Damage, bool bDealt)
{
	if (!bEpisodeActive) return;

	FTeamEpisodeMetrics& Metrics = GetTeamMetrics(TeamName);

	if (bDealt)
	{
		Metrics.TotalDamageDealt += Damage;
	}
	else
	{
		Metrics.TotalDamageTaken += Damage;
	}
}

void UExperimentTracker::RecordObjectiveEvent(const FString& TeamName, bool bCompleted, float CompletionTime)
{
	if (!bEpisodeActive) return;

	FTeamEpisodeMetrics& Metrics = GetTeamMetrics(TeamName);

	if (bCompleted)
	{
		Metrics.ObjectivesCompleted++;
		ObjectiveCompletionTimes.FindOrAdd(TeamName).Add(CompletionTime);

		UE_LOG(LogTemp, Log, TEXT("ExperimentTracker: %s - Objective completed in %.1fs"),
			*TeamName, CompletionTime);
	}
	else
	{
		Metrics.ObjectivesFailed++;
	}
}

void UExperimentTracker::RecordMCTSExecution(const FString& TeamName, float Latency, float ValueVariance, float PolicyEntropy)
{
	if (!bEpisodeActive || !bTrackMCTSMetrics) return;

	FTeamEpisodeMetrics& Metrics = GetTeamMetrics(TeamName);
	Metrics.MCTSExecutions++;

	// Update running average
	float N = static_cast<float>(Metrics.MCTSExecutions);
	Metrics.AverageMCTSLatency = ((Metrics.AverageMCTSLatency * (N - 1.0f)) + Latency) / N;
	Metrics.AverageValueVariance = ((Metrics.AverageValueVariance * (N - 1.0f)) + ValueVariance) / N;
	Metrics.AveragePolicyEntropy = ((Metrics.AveragePolicyEntropy * (N - 1.0f)) + PolicyEntropy) / N;
}

void UExperimentTracker::RecordCombinedFire(const FString& TeamName)
{
	if (!bEpisodeActive) return;

	FTeamEpisodeMetrics& Metrics = GetTeamMetrics(TeamName);
	Metrics.CombinedFireEvents++;
}

void UExperimentTracker::RecordFormationCoherence(const FString& TeamName, float Coherence)
{
	if (!bEpisodeActive) return;

	FormationSamples.FindOrAdd(TeamName).Add(Coherence);
}

void UExperimentTracker::RecordReward(const FString& TeamName, float Reward, const FString& RewardType)
{
	if (!bEpisodeActive || !bTrackRewardBreakdown) return;

	FTeamEpisodeMetrics& Metrics = GetTeamMetrics(TeamName);
	Metrics.TotalReward += Reward;

	if (RewardType == TEXT("Individual"))
	{
		Metrics.IndividualReward += Reward;
	}
	else if (RewardType == TEXT("Coordination"))
	{
		Metrics.CoordinationReward += Reward;
	}
	else if (RewardType == TEXT("Strategic"))
	{
		Metrics.StrategicReward += Reward;
	}
}

//------------------------------------------------------------------------------
// DATA EXPORT
//------------------------------------------------------------------------------

bool UExperimentTracker::ExportToCSV()
{
	if (CompletedEpisodes.Num() == 0)
	{
		UE_LOG(LogTemp, Warning, TEXT("ExperimentTracker: No episodes to export"));
		return false;
	}

	FString FilePath = GenerateCSVFilename();
	bool bFileExists = FPaths::FileExists(FilePath);

	FString CSVContent;

	// Add header if file doesn't exist
	if (!bFileExists)
	{
		CSVContent += GenerateCSVHeader() + TEXT("\n");
	}

	// Add latest episode row
	CSVContent += EpisodeToCSVRow(CompletedEpisodes.Last()) + TEXT("\n");

	// Append to file
	if (FFileHelper::SaveStringToFile(CSVContent, *FilePath, FFileHelper::EEncodingOptions::AutoDetect, &IFileManager::Get(), FILEWRITE_Append))
	{
		UE_LOG(LogTemp, Log, TEXT("✅ ExperimentTracker: Exported episode #%d to %s"),
			CompletedEpisodes.Last().Metadata.EpisodeNumber, *FilePath);
		return true;
	}

	UE_LOG(LogTemp, Error, TEXT("❌ ExperimentTracker: Failed to write to %s"), *FilePath);
	return false;
}

bool UExperimentTracker::ExportAllToCSV(const FString& FilePath)
{
	if (CompletedEpisodes.Num() == 0)
	{
		UE_LOG(LogTemp, Warning, TEXT("ExperimentTracker: No episodes to export"));
		return false;
	}

	FString CSVContent = GenerateCSVHeader() + TEXT("\n");

	for (const FEpisodeResults& Episode : CompletedEpisodes)
	{
		CSVContent += EpisodeToCSVRow(Episode) + TEXT("\n");
	}

	if (FFileHelper::SaveStringToFile(CSVContent, *FilePath))
	{
		UE_LOG(LogTemp, Log, TEXT("✅ ExperimentTracker: Exported %d episodes to %s"),
			CompletedEpisodes.Num(), *FilePath);
		return true;
	}

	UE_LOG(LogTemp, Error, TEXT("❌ ExperimentTracker: Failed to write to %s"), *FilePath);
	return false;
}

bool UExperimentTracker::ExportSummary(const FString& FilePath)
{
	if (CompletedEpisodes.Num() == 0)
	{
		UE_LOG(LogTemp, Warning, TEXT("ExperimentTracker: No episodes to summarize"));
		return false;
	}

	FString Summary;
	Summary += FString::Printf(TEXT("Experiment Summary: %s\n"), *ExperimentName);
	Summary += FString::Printf(TEXT("Total Episodes: %d\n"), CompletedEpisodes.Num());
	Summary += TEXT("=================================================\n\n");

	// Win rates
	int32 AlphaWins = 0, BravoWins = 0, Draws = 0;
	for (const FEpisodeResults& Ep : CompletedEpisodes)
	{
		if (Ep.WinnerTeam == TEXT("TeamAlpha")) AlphaWins++;
		else if (Ep.WinnerTeam == TEXT("TeamBravo")) BravoWins++;
		else Draws++;
	}

	float AlphaWinRate = (CompletedEpisodes.Num() > 0) ? (100.0f * AlphaWins / CompletedEpisodes.Num()) : 0.0f;
	float BravoWinRate = (CompletedEpisodes.Num() > 0) ? (100.0f * BravoWins / CompletedEpisodes.Num()) : 0.0f;

	Summary += FString::Printf(TEXT("Win Rates:\n"));
	Summary += FString::Printf(TEXT("  Team Alpha: %.1f%% (%d wins)\n"), AlphaWinRate, AlphaWins);
	Summary += FString::Printf(TEXT("  Team Bravo: %.1f%% (%d wins)\n"), BravoWinRate, BravoWins);
	Summary += FString::Printf(TEXT("  Draws: %d\n\n"), Draws);

	// Average metrics
	float AvgAlphaKD = GetAverageKDRatio(TEXT("TeamAlpha"));
	float AvgBravoKD = GetAverageKDRatio(TEXT("TeamBravo"));

	Summary += FString::Printf(TEXT("Average K/D Ratios:\n"));
	Summary += FString::Printf(TEXT("  Team Alpha: %.2f\n"), AvgAlphaKD);
	Summary += FString::Printf(TEXT("  Team Bravo: %.2f\n\n"), AvgBravoKD);

	// Episode duration
	float TotalDuration = 0.0f;
	for (const FEpisodeResults& Ep : CompletedEpisodes)
	{
		TotalDuration += Ep.EpisodeDuration;
	}
	float AvgDuration = TotalDuration / CompletedEpisodes.Num();

	Summary += FString::Printf(TEXT("Average Episode Duration: %.1f seconds\n\n"), AvgDuration);

	// Save to file
	if (FFileHelper::SaveStringToFile(Summary, *FilePath))
	{
		UE_LOG(LogTemp, Log, TEXT("✅ ExperimentTracker: Exported summary to %s"), *FilePath);
		return true;
	}

	UE_LOG(LogTemp, Error, TEXT("❌ ExperimentTracker: Failed to write summary to %s"), *FilePath);
	return false;
}

//------------------------------------------------------------------------------
// QUERIES
//------------------------------------------------------------------------------

float UExperimentTracker::GetTeamWinRate(const FString& TeamName) const
{
	if (CompletedEpisodes.Num() == 0) return 0.0f;

	int32 Wins = 0;
	for (const FEpisodeResults& Ep : CompletedEpisodes)
	{
		if (Ep.WinnerTeam == TeamName)
		{
			Wins++;
		}
	}

	return 100.0f * Wins / CompletedEpisodes.Num();
}

float UExperimentTracker::GetAverageKDRatio(const FString& TeamName) const
{
	if (CompletedEpisodes.Num() == 0) return 0.0f;

	float TotalKD = 0.0f;
	for (const FEpisodeResults& Ep : CompletedEpisodes)
	{
		if (TeamName == TEXT("TeamAlpha"))
		{
			TotalKD += Ep.TeamAlpha.KillDeathRatio;
		}
		else if (TeamName == TEXT("TeamBravo"))
		{
			TotalKD += Ep.TeamBravo.KillDeathRatio;
		}
	}

	return TotalKD / CompletedEpisodes.Num();
}

//------------------------------------------------------------------------------
// AUTO-DISCOVERY
//------------------------------------------------------------------------------

void UExperimentTracker::RegisterTeamLeader(UTeamLeaderComponent* TeamLeader)
{
	if (!TeamLeader) return;

	RegisteredLeaders.Add(TeamLeader->TeamName, TeamLeader);

	UE_LOG(LogTemp, Log, TEXT("ExperimentTracker: Registered team leader '%s'"),
		*TeamLeader->TeamName);
}

void UExperimentTracker::RegisterFollower(UFollowerAgentComponent* Follower)
{
	if (!Follower || !Follower->GetOwner()) return;

	// Auto-bind to health events for tracking
	UHealthComponent* HealthComp = Follower->GetOwner()->FindComponentByClass<UHealthComponent>();
	if (HealthComp)
	{
		// Note: In production, you'd bind delegates here
		// For now, tracking happens via explicit RecordXXX calls
		UE_LOG(LogTemp, Verbose, TEXT("ExperimentTracker: Registered follower '%s'"),
			*Follower->GetOwner()->GetName());
	}
}

//------------------------------------------------------------------------------
// HELPERS
//------------------------------------------------------------------------------

FTeamEpisodeMetrics& UExperimentTracker::GetTeamMetrics(const FString& TeamName)
{
	if (TeamName.Contains(TEXT("Alpha")))
	{
		CurrentResults.TeamAlpha.TeamName = TEXT("TeamAlpha");
		return CurrentResults.TeamAlpha;
	}
	else
	{
		CurrentResults.TeamBravo.TeamName = TEXT("TeamBravo");
		return CurrentResults.TeamBravo;
	}
}

FString UExperimentTracker::GenerateCSVFilename() const
{
	FString BasePath = FPaths::ProjectSavedDir() / OutputDirectory;
	FString Filename = FString::Printf(TEXT("%s_%s.csv"),
		*ExperimentName,
		*CurrentResults.Metadata.MapName);

	return BasePath / Filename;
}

FString UExperimentTracker::GenerateCSVHeader() const
{
	TArray<FString> Headers;

	// Metadata
	Headers.Add(TEXT("EpisodeNumber"));
	Headers.Add(TEXT("Timestamp"));
	Headers.Add(TEXT("MapName"));
	Headers.Add(TEXT("Winner"));
	Headers.Add(TEXT("WinCondition"));
	Headers.Add(TEXT("Duration"));

	// Team Alpha metrics
	Headers.Add(TEXT("Alpha_Kills"));
	Headers.Add(TEXT("Alpha_Deaths"));
	Headers.Add(TEXT("Alpha_KD"));
	Headers.Add(TEXT("Alpha_DamageDealt"));
	Headers.Add(TEXT("Alpha_DamageTaken"));
	Headers.Add(TEXT("Alpha_CoordKills"));
	Headers.Add(TEXT("Alpha_CombinedFire"));
	Headers.Add(TEXT("Alpha_FormationCoherence"));
	Headers.Add(TEXT("Alpha_ObjectivesComplete"));
	Headers.Add(TEXT("Alpha_ObjectivesFailed"));

	if (bTrackMCTSMetrics)
	{
		Headers.Add(TEXT("Alpha_MCTSExecutions"));
		Headers.Add(TEXT("Alpha_MCTSLatency"));
		Headers.Add(TEXT("Alpha_ValueVariance"));
		Headers.Add(TEXT("Alpha_PolicyEntropy"));
	}

	if (bTrackRewardBreakdown)
	{
		Headers.Add(TEXT("Alpha_TotalReward"));
		Headers.Add(TEXT("Alpha_IndividualReward"));
		Headers.Add(TEXT("Alpha_CoordinationReward"));
		Headers.Add(TEXT("Alpha_StrategicReward"));
	}

	// Team Bravo metrics (same structure)
	Headers.Add(TEXT("Bravo_Kills"));
	Headers.Add(TEXT("Bravo_Deaths"));
	Headers.Add(TEXT("Bravo_KD"));
	Headers.Add(TEXT("Bravo_DamageDealt"));
	Headers.Add(TEXT("Bravo_DamageTaken"));
	Headers.Add(TEXT("Bravo_CoordKills"));
	Headers.Add(TEXT("Bravo_CombinedFire"));
	Headers.Add(TEXT("Bravo_FormationCoherence"));
	Headers.Add(TEXT("Bravo_ObjectivesComplete"));
	Headers.Add(TEXT("Bravo_ObjectivesFailed"));

	if (bTrackMCTSMetrics)
	{
		Headers.Add(TEXT("Bravo_MCTSExecutions"));
		Headers.Add(TEXT("Bravo_MCTSLatency"));
		Headers.Add(TEXT("Bravo_ValueVariance"));
		Headers.Add(TEXT("Bravo_PolicyEntropy"));
	}

	if (bTrackRewardBreakdown)
	{
		Headers.Add(TEXT("Bravo_TotalReward"));
		Headers.Add(TEXT("Bravo_IndividualReward"));
		Headers.Add(TEXT("Bravo_CoordinationReward"));
		Headers.Add(TEXT("Bravo_StrategicReward"));
	}

	// Performance
	Headers.Add(TEXT("AvgFPS"));
	Headers.Add(TEXT("PeakMemoryMB"));

	return FString::Join(Headers, TEXT(","));
}

FString UExperimentTracker::EpisodeToCSVRow(const FEpisodeResults& Results) const
{
	TArray<FString> Values;

	// Metadata
	Values.Add(FString::FromInt(Results.Metadata.EpisodeNumber));
	Values.Add(Results.Metadata.Timestamp);
	Values.Add(Results.Metadata.MapName);
	Values.Add(Results.WinnerTeam);
	Values.Add(Results.WinCondition);
	Values.Add(FString::SanitizeFloat(Results.EpisodeDuration));

	// Team Alpha
	Values.Add(FString::FromInt(Results.TeamAlpha.TotalKills));
	Values.Add(FString::FromInt(Results.TeamAlpha.TotalDeaths));
	Values.Add(FString::SanitizeFloat(Results.TeamAlpha.KillDeathRatio));
	Values.Add(FString::SanitizeFloat(Results.TeamAlpha.TotalDamageDealt));
	Values.Add(FString::SanitizeFloat(Results.TeamAlpha.TotalDamageTaken));
	Values.Add(FString::FromInt(Results.TeamAlpha.CoordinatedKills));
	Values.Add(FString::FromInt(Results.TeamAlpha.CombinedFireEvents));
	Values.Add(FString::SanitizeFloat(Results.TeamAlpha.AverageFormationCoherence));
	Values.Add(FString::FromInt(Results.TeamAlpha.ObjectivesCompleted));
	Values.Add(FString::FromInt(Results.TeamAlpha.ObjectivesFailed));

	if (bTrackMCTSMetrics)
	{
		Values.Add(FString::FromInt(Results.TeamAlpha.MCTSExecutions));
		Values.Add(FString::SanitizeFloat(Results.TeamAlpha.AverageMCTSLatency));
		Values.Add(FString::SanitizeFloat(Results.TeamAlpha.AverageValueVariance));
		Values.Add(FString::SanitizeFloat(Results.TeamAlpha.AveragePolicyEntropy));
	}

	if (bTrackRewardBreakdown)
	{
		Values.Add(FString::SanitizeFloat(Results.TeamAlpha.TotalReward));
		Values.Add(FString::SanitizeFloat(Results.TeamAlpha.IndividualReward));
		Values.Add(FString::SanitizeFloat(Results.TeamAlpha.CoordinationReward));
		Values.Add(FString::SanitizeFloat(Results.TeamAlpha.StrategicReward));
	}

	// Team Bravo
	Values.Add(FString::FromInt(Results.TeamBravo.TotalKills));
	Values.Add(FString::FromInt(Results.TeamBravo.TotalDeaths));
	Values.Add(FString::SanitizeFloat(Results.TeamBravo.KillDeathRatio));
	Values.Add(FString::SanitizeFloat(Results.TeamBravo.TotalDamageDealt));
	Values.Add(FString::SanitizeFloat(Results.TeamBravo.TotalDamageTaken));
	Values.Add(FString::FromInt(Results.TeamBravo.CoordinatedKills));
	Values.Add(FString::FromInt(Results.TeamBravo.CombinedFireEvents));
	Values.Add(FString::SanitizeFloat(Results.TeamBravo.AverageFormationCoherence));
	Values.Add(FString::FromInt(Results.TeamBravo.ObjectivesCompleted));
	Values.Add(FString::FromInt(Results.TeamBravo.ObjectivesFailed));

	if (bTrackMCTSMetrics)
	{
		Values.Add(FString::FromInt(Results.TeamBravo.MCTSExecutions));
		Values.Add(FString::SanitizeFloat(Results.TeamBravo.AverageMCTSLatency));
		Values.Add(FString::SanitizeFloat(Results.TeamBravo.AverageValueVariance));
		Values.Add(FString::SanitizeFloat(Results.TeamBravo.AveragePolicyEntropy));
	}

	if (bTrackRewardBreakdown)
	{
		Values.Add(FString::SanitizeFloat(Results.TeamBravo.TotalReward));
		Values.Add(FString::SanitizeFloat(Results.TeamBravo.IndividualReward));
		Values.Add(FString::SanitizeFloat(Results.TeamBravo.CoordinationReward));
		Values.Add(FString::SanitizeFloat(Results.TeamBravo.StrategicReward));
	}

	// Performance
	Values.Add(FString::SanitizeFloat(Results.AverageFrameRate));
	Values.Add(FString::SanitizeFloat(Results.PeakMemoryUsageMB));

	return FString::Join(Values, TEXT(","));
}

void UExperimentTracker::CalculateFinalMetrics()
{
	// Calculate K/D ratios
	if (CurrentResults.TeamAlpha.TotalDeaths > 0)
	{
		CurrentResults.TeamAlpha.KillDeathRatio =
			static_cast<float>(CurrentResults.TeamAlpha.TotalKills) / CurrentResults.TeamAlpha.TotalDeaths;
	}
	else
	{
		CurrentResults.TeamAlpha.KillDeathRatio = static_cast<float>(CurrentResults.TeamAlpha.TotalKills);
	}

	if (CurrentResults.TeamBravo.TotalDeaths > 0)
	{
		CurrentResults.TeamBravo.KillDeathRatio =
			static_cast<float>(CurrentResults.TeamBravo.TotalKills) / CurrentResults.TeamBravo.TotalDeaths;
	}
	else
	{
		CurrentResults.TeamBravo.KillDeathRatio = static_cast<float>(CurrentResults.TeamBravo.TotalKills);
	}

	// Calculate average formation coherence
	if (FormationSamples.Contains(TEXT("TeamAlpha")))
	{
		TArray<float>& Samples = FormationSamples[TEXT("TeamAlpha")];
		if (Samples.Num() > 0)
		{
			float Sum = 0.0f;
			for (float Sample : Samples) Sum += Sample;
			CurrentResults.TeamAlpha.AverageFormationCoherence = Sum / Samples.Num();
		}
	}

	if (FormationSamples.Contains(TEXT("TeamBravo")))
	{
		TArray<float>& Samples = FormationSamples[TEXT("TeamBravo")];
		if (Samples.Num() > 0)
		{
			float Sum = 0.0f;
			for (float Sample : Samples) Sum += Sample;
			CurrentResults.TeamBravo.AverageFormationCoherence = Sum / Samples.Num();
		}
	}

	// Calculate average objective completion time
	if (ObjectiveCompletionTimes.Contains(TEXT("TeamAlpha")))
	{
		TArray<float>& Times = ObjectiveCompletionTimes[TEXT("TeamAlpha")];
		if (Times.Num() > 0)
		{
			float Sum = 0.0f;
			for (float Time : Times) Sum += Time;
			CurrentResults.TeamAlpha.AverageObjectiveCompletionTime = Sum / Times.Num();
		}
	}

	if (ObjectiveCompletionTimes.Contains(TEXT("TeamBravo")))
	{
		TArray<float>& Times = ObjectiveCompletionTimes[TEXT("TeamBravo")];
		if (Times.Num() > 0)
		{
			float Sum = 0.0f;
			for (float Time : Times) Sum += Time;
			CurrentResults.TeamBravo.AverageObjectiveCompletionTime = Sum / Times.Num();
		}
	}

	// Calculate average FPS
	if (FrameRateSamples.Num() > 0)
	{
		float Sum = 0.0f;
		for (float FPS : FrameRateSamples) Sum += FPS;
		CurrentResults.AverageFrameRate = Sum / FrameRateSamples.Num();
	}

	// Collect metrics from registered team leaders
	CollectTeamLeaderMetrics();
}

void UExperimentTracker::CollectTeamLeaderMetrics()
{
	for (const auto& Pair : RegisteredLeaders)
	{
		UTeamLeaderComponent* Leader = Pair.Value;
		if (!Leader) continue;

		FString TeamName = Leader->TeamName;
		FTeamEpisodeMetrics& Metrics = GetTeamMetrics(TeamName);

		// Get team metrics
		FTeamMetrics TeamMetrics = Leader->GetTeamMetrics();

		// Update counts
		Metrics.InitialAgents = Leader->GetFollowerCount();
		Metrics.SurvivingAgents = Leader->GetAliveFollowers().Num();

		// Sample current formation coherence
		if (Leader->CurrentTeamObservation.AliveFollowers > 0)
		{
			RecordFormationCoherence(TeamName, Leader->CurrentTeamObservation.FormationCoherence);
		}

		// Get MCTS metrics if continuous planning is active
		if (Leader->bContinuousPlanning && Leader->StrategicMCTS)
		{
			float ValueVariance = 0.0f;
			float PolicyEntropy = 0.0f;
			float AverageValue = 0.0f;

			Leader->StrategicMCTS->GetMCTSStatistics(ValueVariance, PolicyEntropy, AverageValue);

			// Note: Latency is tracked per execution via RecordMCTSExecution
			// This just ensures we capture final state
		}
	}
}

void UExperimentTracker::CollectFollowerMetrics()
{
	// Collect metrics from all registered followers
	// Note: This would require tracking all followers separately
	// For now, metrics are collected via event callbacks (RecordKill, etc.)
}
