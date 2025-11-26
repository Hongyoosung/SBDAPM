// Copyright Epic Games, Inc. All Rights Reserved.

#pragma once

#include "CoreMinimal.h"
#include "UObject/NoExportTypes.h"
#include "NNE.h"
#include "NNERuntimeCPU.h"
#include "Observation/TeamObservation.h"
#include "Observation/TeamObservationTypes.h"
#include "Simulation/StateTransition.h"
#include "RL/RLTypes.h"
#include "WorldModel.generated.h"

/**
 * Configuration for world model
 */
USTRUCT(BlueprintType)
struct FWorldModelConfig
{
	GENERATED_BODY()

	// ONNX model asset path
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "World Model")
	FString ModelPath = TEXT("/Game/AI/Models/world_model.onnx");

	// Maximum prediction steps (rollout depth)
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "World Model")
	int32 MaxPredictionSteps = 5;

	// Prediction time step (seconds)
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "World Model")
	float TimeStepSeconds = 1.0f;

	// Use stochastic sampling (add noise to predictions)
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "World Model")
	bool bStochasticSampling = true;

	// Sampling temperature (higher = more exploration)
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "World Model")
	float SamplingTemperature = 0.1f;

	// Ensemble size (average N model predictions)
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "World Model")
	int32 EnsembleSize = 1;

	// Blend with heuristic predictions (0.0 = all learned, 1.0 = all heuristic)
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "World Model")
	float HeuristicBlendWeight = 0.3f;

	// Enable logging for debugging
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "World Model")
	bool bEnableLogging = false;

	FWorldModelConfig()
		: MaxPredictionSteps(5)
		, TimeStepSeconds(1.0f)
		, bStochasticSampling(true)
		, SamplingTemperature(0.1f)
		, EnsembleSize(1)
		, HeuristicBlendWeight(0.3f)
		, bEnableLogging(false)
	{}
};

/**
 * World Model - Predicts future states for MCTS simulation
 *
 * Architecture:
 *   Input: CurrentState (TeamObs) + Actions (strategic + tactical)
 *   Output: NextState (predicted state delta)
 *
 * Training:
 *   - Supervised learning on real game transitions
 *   - MSE loss on state prediction
 *
 * Usage:
 *   1. Train model with train_world_model.py
 *   2. Export to ONNX
 *   3. Load in MCTS via InitializeWorldModel()
 *   4. Call PredictNextState() during simulation
 */
UCLASS(BlueprintType)
class GAMEAI_PROJECT_API UWorldModel : public UObject
{
	GENERATED_BODY()

public:
	UWorldModel();

	/**
	 * Initialize world model with ONNX model
	 * @param Config World model configuration
	 * @return True if initialization succeeded
	 */
	UFUNCTION(BlueprintCallable, Category = "World Model")
	bool Initialize(const FWorldModelConfig& Config);

	/**
	 * Check if world model is initialized and ready
	 */
	UFUNCTION(BlueprintCallable, Category = "World Model")
	bool IsReady() const { return bIsInitialized; }

	/**
	 * Get average inference time (ms)
	 */
	UFUNCTION(BlueprintCallable, Category = "World Model")
	float GetAverageInferenceTime() const { return AverageInferenceTimeMs; }

	/**
	 * Get model version
	 */
	UFUNCTION(BlueprintCallable, Category = "World Model")
	int32 GetModelVersion() const { return ModelVersion; }

	/**
	 * Predict next state given current state and actions
	 * @param CurrentState Current team observation
	 * @param TacticalActions Tactical actions being executed
	 * @return Predicted state delta with confidence
	 */
	UFUNCTION(BlueprintCallable, Category = "World Model")
	FWorldModelPrediction PredictNextState(
		const FTeamObservation& CurrentState,
		const TArray<FTacticalAction>& TacticalActions
	);

	/**
	 * Predict rollout of future states
	 * @param InitialState Starting state
	 * @param ActionSequence Sequence of actions per step
	 * @param NumSteps Number of steps to predict
	 * @return Array of predictions
	 */
	UFUNCTION(BlueprintCallable, Category = "World Model")
	TArray<FWorldModelPrediction> PredictRollout(
		const FTeamObservation& InitialState,
		const TArray<FActionSequence>& ActionSequence,
		int32 NumSteps
	);

	/**
	 * Log state transition sample for training
	 * @param Sample Transition sample to log
	 */
	UFUNCTION(BlueprintCallable, Category = "World Model")
	void LogTransitionSample(const FStateTransitionSample& Sample);

	/**
	 * Export logged samples to JSON for training
	 * @param OutputPath File path to export
	 * @return True if export succeeded
	 */
	UFUNCTION(BlueprintCallable, Category = "World Model")
	bool ExportTransitionSamples(const FString& OutputPath);

private:
	// NNE runtime and model
	TSharedPtr<UE::NNE::IModelInstanceCPU> ModelInstance;
	bool bIsInitialized = false;

	// Configuration
	FWorldModelConfig Config;

	// Input/output tensor specs
	int32 InputSize = 0;
	int32 OutputSize = 0;

	// Performance tracking
	float AverageInferenceTimeMs = 0.0f;
	int32 InferenceCount = 0;
	int32 ModelVersion = 1;

	// Training data collection
	TArray<FStateTransitionSample> CollectedSamples;
	bool bCollectingData = false;

	/**
	 * Encode state and actions to model input tensor
	 */
	TArray<float> EncodeInput(
		const FTeamObservation& State,
		const TArray<FTacticalAction>& Actions
	);

	/**
	 * Decode model output tensor to state delta
	 */
	FTeamStateDelta DecodeOutput(const TArray<float>& OutputTensor);

	/**
	 * Apply stochastic sampling to prediction
	 */
	void ApplyStochasticSampling(FTeamStateDelta& Delta);


	/**
	 * Update performance metrics
	 */
	void UpdatePerformanceMetrics(float InferenceTimeMs);

	/**
	 * Validate model output
	 */
	bool ValidateOutput(const FTeamStateDelta& Delta) const;
};
