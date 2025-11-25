// Copyright Epic Games, Inc. All Rights Reserved.

#pragma once

#include "CoreMinimal.h"
#include "UObject/NoExportTypes.h"
#include "Observation/TeamObservation.h"
#include "NNE.h"
#include "NNEModelData.h"
#include "NNERuntimeCPU.h"
#include "TeamValueNetwork.generated.h"

/**
 * Team Value Network for MCTS Leaf Node Evaluation
 *
 * Architecture (v3.0):
 *   Input: FTeamObservation (40 team + N×71 individual features)
 *   Embedding Layer: 256 neurons (ReLU)
 *   Shared Trunk: 256→256→128 (ReLU)
 *   Value Head: 128→64→1 (Tanh)
 *   Output: Team state value [-1, 1] (loss → win probability)
 *
 * Purpose:
 *   - Replaces hand-crafted CalculateTeamReward() heuristics in MCTS
 *   - Guides MCTS tree search with learned value estimates
 *   - Trained on MCTS rollout outcomes via TD-learning
 *
 * Usage:
 *   1. Load trained model: LoadModel("Models/team_value_network.onnx")
 *   2. Evaluate state: float Value = EvaluateState(TeamObservation)
 *   3. Use in MCTS: Replace static evaluation with network prediction
 */
UCLASS(BlueprintType, Blueprintable)
class GAMEAI_PROJECT_API UTeamValueNetwork : public UObject
{
	GENERATED_BODY()

public:
	UTeamValueNetwork();

	// ========================================
	// Initialization
	// ========================================

	/**
	 * Initialize the value network
	 * @param MaxAgents - Maximum number of agents per team (for input sizing)
	 * @return True if initialization succeeded
	 */
	UFUNCTION(BlueprintCallable, Category = "RL|ValueNetwork")
	bool Initialize(int32 MaxAgents = 10);

	/**
	 * Load a trained value network from ONNX file
	 * @param ModelPath - Path to .onnx file
	 * @return True if model loaded successfully
	 */
	UFUNCTION(BlueprintCallable, Category = "RL|ValueNetwork")
	bool LoadModel(const FString& ModelPath);

	/**
	 * Unload the current model
	 */
	UFUNCTION(BlueprintCallable, Category = "RL|ValueNetwork")
	void UnloadModel();

	// ========================================
	// Value Estimation
	// ========================================

	/**
	 * Evaluate team state value for MCTS leaf nodes
	 * @param TeamObs - Current team observation
	 * @return State value in [-1, 1] (loss → win probability)
	 */
	UFUNCTION(BlueprintCallable, Category = "RL|ValueNetwork")
	float EvaluateState(const FTeamObservation& TeamObs);

	/**
	 * Batch evaluate multiple states (for parallel MCTS simulations)
	 * @param TeamObservations - Array of team states to evaluate
	 * @return Array of values in [-1, 1]
	 */
	UFUNCTION(BlueprintCallable, Category = "RL|ValueNetwork")
	TArray<float> EvaluateStateBatch(const TArray<FTeamObservation>& TeamObservations);

	// ========================================
	// Utility
	// ========================================

	/**
	 * Check if model is loaded and ready
	 */
	UFUNCTION(BlueprintCallable, Category = "RL|ValueNetwork")
	bool IsReady() const { return bIsInitialized && bModelLoaded; }

	/**
	 * Get input feature size
	 */
	UFUNCTION(BlueprintCallable, Category = "RL|ValueNetwork")
	int32 GetInputSize() const { return InputSize; }

private:
	// ========================================
	// Neural Network Inference
	// ========================================

	/**
	 * Forward pass through the value network
	 * @param InputFeatures - Flattened team observation
	 * @return Single value in [-1, 1]
	 */
	float ForwardPass(const TArray<float>& InputFeatures);

	/**
	 * Convert TeamObservation to flat feature vector
	 * @param TeamObs - Team observation
	 * @return Flattened features (40 team + N×71 individual)
	 */
	TArray<float> TeamObservationToFeatures(const FTeamObservation& TeamObs);

	// ========================================
	// Configuration
	// ========================================

	// Is the network initialized?
	UPROPERTY()
	bool bIsInitialized;

	// Is the ONNX model loaded?
	UPROPERTY()
	bool bModelLoaded;

	// Maximum agents per team (for input sizing)
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "RL|ValueNetwork", meta = (AllowPrivateAccess = "true"))
	int32 MaxAgents;

	// Input feature size (40 team + MaxAgents×71 individual)
	UPROPERTY()
	int32 InputSize;

	// Model path (for debugging)
	UPROPERTY()
	FString ModelPath;

	// ========================================
	// NNE (Neural Network Engine) State
	// ========================================

	// NNE model data (loaded from .onnx file)
	UPROPERTY()
	TObjectPtr<UNNEModelData> ModelData;

	// NNE model instance for inference
	TSharedPtr<UE::NNE::IModelInstanceCPU> ModelInstance;

	// Input/output tensor buffers
	TArray<float> InputBuffer;
	TArray<float> OutputBuffer;
};
