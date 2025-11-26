// Copyright Epic Games, Inc. All Rights Reserved.

#pragma once

#include "CoreMinimal.h"
#include "UObject/NoExportTypes.h"
#include "RLTypes.h"
#include "Observation/TeamObservation.h"
#include "CurriculumManager.generated.h"

/**
 * Stores MCTS uncertainty metrics for a training scenario
 */
USTRUCT(BlueprintType)
struct FMCTSScenarioMetrics
{
	GENERATED_BODY()

	// Strategic command that was executed
	UPROPERTY()
	int32 CommandType = 0;

	// Team observation when decision was made
	UPROPERTY()
	struct FTeamObservation TeamObservation;

	// MCTS statistics
	UPROPERTY()
	float ValueVariance = 0.0f;  // Std dev of child node values

	UPROPERTY()
	float PolicyEntropy = 0.0f;  // Uncertainty in action selection

	UPROPERTY()
	int32 VisitCount = 0;  // Number of MCTS simulations

	UPROPERTY()
	float AverageValue = 0.0f;  // Average value estimate

	// Actual outcome (for validation)
	UPROPERTY()
	float ActualReward = 0.0f;

	UPROPERTY()
	float Timestamp = 0.0f;

	// Priority score (higher = more valuable for training)
	float CalculatePriority() const
	{
		// High variance + high entropy = uncertain scenario = high training value
		// Normalize by visit count to avoid biasing toward under-explored nodes
		const float UncertaintyScore = ValueVariance + PolicyEntropy * 0.5f;
		const float NormalizedVisits = FMath::Max(1.0f, static_cast<float>(VisitCount));
		return UncertaintyScore / FMath::Sqrt(NormalizedVisits);
	}
};

/**
 * Priority queue entry for scenario sampling
 */
struct FScenarioPriorityEntry
{
	int32 ScenarioIndex;
	float Priority;

	bool operator<(const FScenarioPriorityEntry& Other) const
	{
		return Priority > Other.Priority;  // Max heap
	}
};

/**
 * Manages curriculum learning by prioritizing training scenarios based on MCTS uncertainty.
 * MCTS identifies high-variance scenarios, CurriculumManager samples these for RL training.
 */
UCLASS(BlueprintType)
class GAMEAI_PROJECT_API UCurriculumManager : public UObject
{
	GENERATED_BODY()

public:
	UCurriculumManager();

	// Add a scenario from MCTS search
	UFUNCTION(BlueprintCallable, Category = "RL|Curriculum")
	void AddScenario(const FMCTSScenarioMetrics& Scenario);

	// Update scenario with actual outcome (called after episode completes)
	UFUNCTION(BlueprintCallable, Category = "RL|Curriculum")
	void UpdateScenarioOutcome(int32 ScenarioIndex, float ActualReward);

	// Sample N scenarios for training batch (prioritized by uncertainty)
	UFUNCTION(BlueprintCallable, Category = "RL|Curriculum")
	TArray<FMCTSScenarioMetrics> SampleScenarios(int32 BatchSize, bool bUsePrioritization = true);

	// Get statistics for monitoring
	UFUNCTION(BlueprintCallable, Category = "RL|Curriculum")
	void GetStatistics(float& OutAveragePriority, float& OutAverageVariance, int32& OutTotalScenarios) const;

	// Clear old scenarios (keep only recent N)
	UFUNCTION(BlueprintCallable, Category = "RL|Curriculum")
	void PruneOldScenarios(int32 MaxScenarios = 10000);

	// Export scenarios to file for offline training
	UFUNCTION(BlueprintCallable, Category = "RL|Curriculum")
	bool ExportToFile(const FString& FilePath) const;

	// Configuration
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Config")
	float PrioritizationExponent = 0.6f;  // Alpha in prioritized replay (0=uniform, 1=full priority)

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Config")
	float ImportanceSamplingBeta = 0.4f;  // Beta for importance sampling correction

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Config")
	int32 MaxBufferSize = 50000;  // Maximum scenarios to keep in memory

private:
	// Stored scenarios
	UPROPERTY()
	TArray<FMCTSScenarioMetrics> Scenarios;

	// Build priority distribution for sampling
	void RebuildPriorityDistribution();

	// Cached priority sum for sampling
	float TotalPriority = 0.0f;

	// Random number generator
	FRandomStream RandomStream;
};
