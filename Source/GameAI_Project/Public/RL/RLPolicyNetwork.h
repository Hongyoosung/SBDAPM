// Copyright Epic Games, Inc. All Rights Reserved.

#pragma once

#include "CoreMinimal.h"
#include "UObject/NoExportTypes.h"
#include "RL/RLTypes.h"
#include "Observation/ObservationElement.h"
#include "RLPolicyNetwork.generated.h"

/**
 * Neural Network-based RL Policy for Tactical Action Selection
 *
 * Architecture:
 *   Input Layer:  71 features (from FObservationElement)
 *   Hidden Layer 1: 128 neurons (ReLU)
 *   Hidden Layer 2: 128 neurons (ReLU)
 *   Hidden Layer 3: 64 neurons (ReLU)
 *   Output Layer: 16 actions (Softmax)
 *
 * Usage:
 *   1. Load trained policy from ONNX: LoadPolicy("path/to/model.onnx")
 *   2. Query for action: SelectAction(Observation)
 *   3. Collect experiences: StoreExperience(State, Action, Reward, NextState)
 *   4. Export for training: ExportExperiencesToJSON("path/to/data.json")
 */
UCLASS(BlueprintType, Blueprintable)
class GAMEAI_PROJECT_API URLPolicyNetwork : public UObject
{
	GENERATED_BODY()

public:
	URLPolicyNetwork();


	// ========================================
	// Initialization
	// ========================================

	/**
	 * Initialize the policy network
	 * @param InConfig - Policy configuration
	 * @return True if initialization succeeded
	 */
	UFUNCTION(BlueprintCallable, Category = "RL")
	bool Initialize(const FRLPolicyConfig& InConfig);

	/**
	 * Load a trained policy from ONNX file
	 * @param ModelPath - Path to .onnx file
	 * @return True if model loaded successfully
	 */
	UFUNCTION(BlueprintCallable, Category = "RL")
	bool LoadPolicy(const FString& ModelPath);

	/**
	 * Unload the current policy
	 */
	UFUNCTION(BlueprintCallable, Category = "RL")
	void UnloadPolicy();

	// ========================================
	// Inference
	// ========================================

	/**
	 * Select a tactical action based on the current observation
	 * Uses epsilon-greedy exploration if enabled
	 * @param Observation - Current 71-feature observation
	 * @return Selected tactical action
	 */
	UFUNCTION(BlueprintCallable, Category = "RL")
	ETacticalAction SelectAction(const FObservationElement& Observation);

	/**
	 * Get action probabilities for all actions
	 * @param Observation - Current 71-feature observation
	 * @return Array of 16 probabilities (sums to 1.0)
	 */
	UFUNCTION(BlueprintCallable, Category = "RL")
	TArray<float> GetActionProbabilities(const FObservationElement& Observation);

	/**
	 * Get the Q-value for a specific action
	 * @param Observation - Current observation
	 * @param Action - Action to evaluate
	 * @return Estimated Q-value
	 */
	UFUNCTION(BlueprintCallable, Category = "RL")
	float GetActionValue(const FObservationElement& Observation, ETacticalAction Action);

	// ========================================
	// Experience Collection
	// ========================================

	/**
	 * Store an experience tuple for offline training
	 * @param State - Current state
	 * @param Action - Action taken
	 * @param Reward - Immediate reward
	 * @param NextState - Resulting state
	 * @param bTerminal - Is this a terminal state?
	 */
	UFUNCTION(BlueprintCallable, Category = "RL")
	void StoreExperience(const FObservationElement& State, ETacticalAction Action, float Reward, const FObservationElement& NextState, bool bTerminal);

	/**
	 * Export collected experiences to JSON file for Python training
	 * @param FilePath - Output file path
	 * @return True if export succeeded
	 */
	UFUNCTION(BlueprintCallable, Category = "RL")
	bool ExportExperiencesToJSON(const FString& FilePath);

	/**
	 * Clear all collected experiences
	 */
	UFUNCTION(BlueprintCallable, Category = "RL")
	void ClearExperiences();

	/**
	 * Get number of collected experiences
	 */
	UFUNCTION(BlueprintCallable, Category = "RL")
	int32 GetExperienceCount() const { return CollectedExperiences.Num(); }

	// ========================================
	// Statistics
	// ========================================

	/**
	 * Get training statistics
	 */
	UFUNCTION(BlueprintCallable, Category = "RL")
	FRLTrainingStats GetTrainingStats() const { return TrainingStats; }

	/**
	 * Reset training statistics
	 */
	UFUNCTION(BlueprintCallable, Category = "RL")
	void ResetStatistics();

	/**
	 * Update epsilon value (for exploration decay)
	 */
	UFUNCTION(BlueprintCallable, Category = "RL")
	void UpdateEpsilon();

	/**
	 * Get current epsilon value
	 */
	UFUNCTION(BlueprintCallable, Category = "RL")
	float GetEpsilon() const { return Config.Epsilon; }

	// ========================================
	// Utility
	// ========================================

	/**
	 * Check if policy is loaded and ready
	 */
	UFUNCTION(BlueprintCallable, Category = "RL")
	bool IsReady() const { return bIsInitialized; }

	/**
	 * Get human-readable action name
	 */
	UFUNCTION(BlueprintCallable, Category = "RL")
	static FString GetActionName(ETacticalAction Action);


private:
	// ========================================
	// Neural Network Inference (ONNX)
	// ========================================

	/**
	 * Forward pass through the neural network
	 * @param InputFeatures - 71-element input vector
	 * @return 16-element output vector (action probabilities)
	 */
	TArray<float> ForwardPass(const TArray<float>& InputFeatures);

	/**
	 * Softmax activation function
	 */
	static TArray<float> Softmax(const TArray<float>& Logits);

	// ========================================
	// Rule-Based Fallback (for testing without trained model)
	// ========================================

	/**
	 * Rule-based action selection (used when ONNX model not available)
	 * Implements heuristics based on observation features
	 */
	ETacticalAction SelectActionRuleBased(const FObservationElement& Observation);

	/**
	 * Calculate heuristic action probabilities
	 */
	TArray<float> GetRuleBasedProbabilities(const FObservationElement& Observation);

	// ========================================
	// Helper Functions
	// ========================================

	/**
	 * Sample action from probability distribution
	 */
	ETacticalAction SampleAction(const TArray<float>& Probabilities);

	/**
	 * Get action with highest probability
	 */
	ETacticalAction GetGreedyAction(const TArray<float>& Probabilities);

	/**
	 * Get random action (for epsilon-greedy exploration)
	 */
	ETacticalAction GetRandomAction();

	/**
	 * Convert ETacticalAction to integer index
	 */
	static int32 ActionToIndex(ETacticalAction Action);

	/**
	 * Convert integer index to ETacticalAction
	 */
	static ETacticalAction IndexToAction(int32 Index);

public:
	// ========================================
	// Configuration
	// ========================================

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "RL|Config")
	FRLPolicyConfig Config;

	// Enable epsilon-greedy exploration
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "RL|Config")
	bool bEnableExploration;

	// Use ONNX model for inference (if false, uses rule-based fallback)
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "RL|Config")
	bool bUseONNXModel;

	// Enable experience collection for offline training
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "RL|Config")
	bool bCollectExperiences;

	// Maximum experiences to store before export
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "RL|Config")
	int32 MaxExperienceBufferSize;


private:
	// ========================================
	// Internal State
	// ========================================

	// Is the policy initialized?
	bool bIsInitialized;

	// Collected experiences for offline training
	TArray<FRLExperience> CollectedExperiences;

	// Training statistics
	FRLTrainingStats TrainingStats;

	// Current episode reward accumulator
	float CurrentEpisodeReward;

	// Current episode step count
	int32 CurrentEpisodeSteps;
};
