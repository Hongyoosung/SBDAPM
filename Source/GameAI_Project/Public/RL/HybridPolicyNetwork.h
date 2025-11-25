// Copyright Epic Games, Inc. All Rights Reserved.

#pragma once

#include "CoreMinimal.h"
#include "UObject/NoExportTypes.h"
#include "RL/RLTypes.h"
#include "Observation/ObservationElement.h"
#include "Observation/TeamObservation.h"
#include "NNE.h"
#include "NNEModelData.h"
#include "NNERuntimeCPU.h"
#include "HybridPolicyNetwork.generated.h"

/**
 * Hybrid Policy Network (v3.0 Sprint 4)
 *
 * Dual-head architecture for coupled MCTS + RL:
 * - Policy Head: Softmax probabilities for immediate tactical action selection
 * - Prior Head: Logits for MCTS node initialization (guides tree search)
 *
 * Architecture:
 *   Input Layer:  71 features (individual observation)
 *   OR
 *   Input Layer:  40 features (team observation) for strategic priors
 *
 *   Shared Trunk: 256 → 256 → 128 (ReLU)
 *
 *   ├─→ Policy Head: 128 → 64 → 8 (Tanh/Sigmoid)  → Immediate action
 *   └─→ Prior Head:  128 → 64 → 7 (Softmax)       → MCTS priors
 *
 * Training:
 *   - Policy head trained via PPO (on-policy RL)
 *   - Prior head trained via supervised learning (MCTS visit counts as targets)
 *   - Joint loss: L_policy + λ * L_prior (λ = 0.3)
 *
 * Usage:
 *   1. Load trained model: LoadModel("path/to/hybrid_policy.onnx")
 *   2. Get tactical action: GetAction(Observation) → FTacticalAction
 *   3. Get MCTS priors: GetObjectivePriors(TeamObs) → TArray<float>
 */
UCLASS(BlueprintType, Blueprintable)
class GAMEAI_PROJECT_API UHybridPolicyNetwork : public UObject
{
	GENERATED_BODY()

public:
	UHybridPolicyNetwork();

	// ========================================
	// Initialization
	// ========================================

	/**
	 * Initialize the hybrid policy network
	 * @param InConfig - Policy configuration
	 * @return True if initialization succeeded
	 */
	UFUNCTION(BlueprintCallable, Category = "RL|Hybrid")
	bool Initialize(const FRLPolicyConfig& InConfig);

	/**
	 * Load trained hybrid policy from ONNX file
	 * Model should have 2 outputs: [action_logits, prior_logits]
	 * @param ModelPath - Path to .onnx file
	 * @return True if model loaded successfully
	 */
	UFUNCTION(BlueprintCallable, Category = "RL|Hybrid")
	bool LoadModel(const FString& ModelPath);

	/**
	 * Unload the current model
	 */
	UFUNCTION(BlueprintCallable, Category = "RL|Hybrid")
	void UnloadModel();

	// ========================================
	// Policy Head (Immediate Action Selection)
	// ========================================

	/**
	 * Get tactical action from policy head
	 * @param Observation - Current 71-feature observation
	 * @param CurrentObjective - Current objective context (optional)
	 * @return 8-dimensional atomic action
	 */
	UFUNCTION(BlueprintCallable, Category = "RL|Hybrid")
	FTacticalAction GetAction(const FObservationElement& Observation, class UObjective* CurrentObjective);

	// ========================================
	// Prior Head (MCTS Initialization)
	// ========================================

	/**
	 * Get objective priors from prior head for MCTS tree search
	 * @param TeamObs - Team-level observation
	 * @return Array of 7 prior probabilities (one per objective type)
	 */
	UFUNCTION(BlueprintCallable, Category = "RL|Hybrid")
	TArray<float> GetObjectivePriors(const FTeamObservation& TeamObs);

	/**
	 * Get both action and priors in a single forward pass (more efficient)
	 * @param Observation - Individual observation
	 * @param TeamObs - Team observation
	 * @param OutAction - Output tactical action
	 * @param OutPriors - Output MCTS priors
	 */
	UFUNCTION(BlueprintCallable, Category = "RL|Hybrid")
	void GetActionAndPriors(
		const FObservationElement& Observation,
		const FTeamObservation& TeamObs,
		FTacticalAction& OutAction,
		TArray<float>& OutPriors
	);

	// ========================================
	// Utility
	// ========================================

	/**
	 * Check if model is loaded and ready
	 */
	UFUNCTION(BlueprintCallable, Category = "RL|Hybrid")
	bool IsReady() const { return bIsInitialized && bModelLoaded; }

private:
	// ========================================
	// Neural Network Inference
	// ========================================

	/**
	 * Forward pass through dual-head network
	 * @param InputFeatures - Input observation
	 * @param OutActionLogits - Policy head output (8 dims)
	 * @param OutPriorLogits - Prior head output (7 dims)
	 */
	void ForwardPass(
		const TArray<float>& InputFeatures,
		TArray<float>& OutActionLogits,
		TArray<float>& OutPriorLogits
	);

	/**
	 * Convert action logits to tactical action
	 */
	FTacticalAction ActionLogitsToTacticalAction(const TArray<float>& ActionLogits);

	/**
	 * Convert prior logits to normalized probabilities
	 */
	TArray<float> PriorLogitsToProbabilities(const TArray<float>& PriorLogits);

public:
	// ========================================
	// Configuration
	// ========================================

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "RL|Config")
	FRLPolicyConfig Config;

	// Path to trained hybrid model
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "RL|Config")
	FString ModelPath;

private:
	// ========================================
	// Internal State
	// ========================================

	bool bIsInitialized;
	bool bModelLoaded;

	// ========================================
	// NNE (Neural Network Engine) State
	// ========================================

	UPROPERTY()
	TObjectPtr<UNNEModelData> ModelData;

	TSharedPtr<UE::NNE::IModelInstanceCPU> ModelInstance;

	// Input/output buffers
	TArray<float> InputBuffer;
	TArray<float> ActionLogitsBuffer;   // Policy head output
	TArray<float> PriorLogitsBuffer;    // Prior head output
};
