// Copyright Epic Games, Inc. All Rights Reserved.

#pragma once

#include "CoreMinimal.h"
#include "UObject/NoExportTypes.h"
#include "RL/RLTypes.h"
#include "RLReplayBuffer.generated.h"

/**
 * Experience Replay Buffer for Reinforcement Learning
 *
 * Stores experience tuples (S, A, R, S', terminal) for offline training
 * Supports:
 *   - Fixed-size circular buffer with automatic oldest-removal
 *   - Random batch sampling
 *   - Priority-based sampling (for PER - Prioritized Experience Replay)
 *   - Export to JSON for Python training
 *
 * Usage:
 *   1. AddExperience() - Store new experiences during gameplay
 *   2. SampleBatch() - Sample random batch for training
 *   3. ExportToJSON() - Export all experiences for offline training
 */
UCLASS(BlueprintType, Blueprintable)
class GAMEAI_PROJECT_API URLReplayBuffer : public UObject
{
	GENERATED_BODY()

public:
	URLReplayBuffer();

	// ========================================
	// Configuration
	// ========================================

	// Maximum buffer capacity (default: 100,000)
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "RL|Config")
	int32 MaxCapacity;

	// Enable prioritized experience replay
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "RL|Config")
	bool bUsePrioritizedReplay;

	// Priority exponent (alpha) for PER
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "RL|Config")
	float PriorityAlpha;

	// Importance sampling exponent (beta) for PER
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "RL|Config")
	float ImportanceSamplingBeta;

	// Small constant to avoid zero priorities
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "RL|Config")
	float PriorityEpsilon;

	// ========================================
	// Buffer Management
	// ========================================

	/**
	 * Add a new experience to the buffer
	 * If buffer is full, removes oldest experience
	 * @param Experience - Experience tuple to add
	 */
	UFUNCTION(BlueprintCallable, Category = "RL")
	void AddExperience(const FRLExperience& Experience);

	/**
	 * Add experience with explicit priority (for PER)
	 * @param Experience - Experience tuple
	 * @param Priority - Initial priority value
	 */
	UFUNCTION(BlueprintCallable, Category = "RL")
	void AddExperienceWithPriority(const FRLExperience& Experience, float Priority);

	/**
	 * Sample a random batch of experiences
	 * @param BatchSize - Number of experiences to sample
	 * @return Array of sampled experiences
	 */
	UFUNCTION(BlueprintCallable, Category = "RL")
	TArray<FRLExperience> SampleBatch(int32 BatchSize);

	/**
	 * Sample a batch using priority-based sampling (PER)
	 * @param BatchSize - Number of experiences to sample
	 * @param OutImportanceWeights - Importance sampling weights for bias correction
	 * @return Array of sampled experiences
	 */
	UFUNCTION(BlueprintCallable, Category = "RL")
	TArray<FRLExperience> SampleBatchPrioritized(int32 BatchSize, TArray<float>& OutImportanceWeights);

	/**
	 * Update priorities for a batch of experiences (for PER)
	 * Used after training to update priorities based on TD error
	 * @param Indices - Indices of experiences to update
	 * @param Priorities - New priority values
	 */
	UFUNCTION(BlueprintCallable, Category = "RL")
	void UpdatePriorities(const TArray<int32>& Indices, const TArray<float>& Priorities);

	/**
	 * Clear all experiences from buffer
	 */
	UFUNCTION(BlueprintCallable, Category = "RL")
	void Clear();

	/**
	 * Remove oldest N experiences
	 * @param Count - Number of experiences to remove
	 */
	UFUNCTION(BlueprintCallable, Category = "RL")
	void RemoveOldest(int32 Count);

	// ========================================
	// Export / Import
	// ========================================

	/**
	 * Export all experiences to JSON file for Python training
	 * @param FilePath - Output file path
	 * @return True if export succeeded
	 */
	UFUNCTION(BlueprintCallable, Category = "RL")
	bool ExportToJSON(const FString& FilePath);

	/**
	 * Import experiences from JSON file
	 * @param FilePath - Input file path
	 * @return Number of experiences imported
	 */
	UFUNCTION(BlueprintCallable, Category = "RL")
	int32 ImportFromJSON(const FString& FilePath);

	// ========================================
	// Statistics
	// ========================================

	/**
	 * Get current buffer size
	 */
	UFUNCTION(BlueprintCallable, Category = "RL")
	int32 GetSize() const { return Experiences.Num(); }

	/**
	 * Check if buffer is empty
	 */
	UFUNCTION(BlueprintCallable, Category = "RL")
	bool IsEmpty() const { return Experiences.Num() == 0; }

	/**
	 * Check if buffer is full
	 */
	UFUNCTION(BlueprintCallable, Category = "RL")
	bool IsFull() const { return Experiences.Num() >= MaxCapacity; }

	/**
	 * Get buffer capacity
	 */
	UFUNCTION(BlueprintCallable, Category = "RL")
	int32 GetCapacity() const { return MaxCapacity; }

	/**
	 * Get buffer usage percentage
	 */
	UFUNCTION(BlueprintCallable, Category = "RL")
	float GetUsagePercentage() const;

	/**
	 * Get average reward across all experiences
	 */
	UFUNCTION(BlueprintCallable, Category = "RL")
	float GetAverageReward() const;

	/**
	 * Get number of terminal experiences
	 */
	UFUNCTION(BlueprintCallable, Category = "RL")
	int32 GetTerminalCount() const;

	/**
	 * Get distribution of actions in buffer
	 * @return Array of 16 counts (one per action)
	 */
	UFUNCTION(BlueprintCallable, Category = "RL")
	TArray<int32> GetActionDistribution() const;

private:
	// ========================================
	// Internal Storage
	// ========================================

	// Experience buffer (circular)
	UPROPERTY()
	TArray<FRLExperience> Experiences;

	// Priority values for each experience (for PER)
	TArray<float> Priorities;

	// Sum tree for efficient priority sampling (for PER)
	TArray<float> SumTree;

	// Current write position (for circular buffer)
	int32 WritePosition;

	// ========================================
	// Helper Functions
	// ========================================

	/**
	 * Calculate priority for a new experience
	 * Uses max priority to ensure new experiences are sampled
	 */
	float CalculateInitialPriority() const;

	/**
	 * Update sum tree after priority change (for PER)
	 */
	void UpdateSumTree(int32 Index);

	/**
	 * Rebuild entire sum tree (called after batch priority updates)
	 */
	void RebuildSumTree();

	/**
	 * Sample index based on priority distribution (for PER)
	 */
	int32 SampleProportionalIndex() const;

	/**
	 * Calculate importance sampling weight (for PER)
	 */
	float CalculateImportanceWeight(int32 Index, float TotalPriority) const;
};
