// ScholaAgentComponent.h - Schola agent component for follower pawns

#pragma once

#include "CoreMinimal.h"
#include "Inference/InferenceComponent.h"
#include "ScholaAgentComponent.generated.h"

class UFollowerAgentComponent;
class UTacticalObserver;
class UTacticalRewardProvider;

/**
 * Schola Agent Component - Integrates RL training via Schola/RLlib
 *
 * This component attaches to follower pawns and:
 * - Exposes 71-feature tactical observations (TacticalObserver)
 * - Provides combat rewards (TacticalRewardProvider)
 * - Bridges between FollowerAgentComponent and Schola's InferenceComponent
 *
 * Architecture:
 * Training: UE5.6 + Schola ←→ gRPC ←→ OpenAI Gym ←→ RLlib
 * Inference: UE5.6 + NNE + ONNX Runtime (no Python)
 *
 * Usage:
 * 1. Add InferenceComponent (from Schola plugin) to follower pawn
 * 2. Add ScholaAgentComponent to the same pawn
 * 3. Ensure FollowerAgentComponent is on the same pawn
 * 4. Component auto-configures observers/rewards
 * 5. Start UE with Schola server enabled
 * 6. Run Python training script (train_rllib.py)
 */
UCLASS(ClassGroup = (AI), meta = (BlueprintSpawnableComponent))
class GAMEAI_PROJECT_API UScholaAgentComponent : public UActorComponent
{
	GENERATED_BODY()

public:
	UScholaAgentComponent();

	virtual void BeginPlay() override;
	virtual void EndPlay(const EEndPlayReason::Type EndPlayReason) override;
	virtual void TickComponent(float DeltaTime, ELevelTick TickType, FActorComponentTickFunction* ThisTickFunction) override;

	//--------------------------------------------------------------------------
	// CONFIGURATION
	//--------------------------------------------------------------------------

	/** Auto-find and configure FollowerAgentComponent */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Schola|Config")
	bool bAutoConfigureFollower = true;

	/** Enable Schola training (gRPC connection to Python) */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Schola|Config")
	bool bEnableScholaTraining = false;

	/** gRPC server port for Schola communication */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Schola|Config")
	int32 ScholaServerPort = 50051;

	//--------------------------------------------------------------------------
	// COMPONENTS
	//--------------------------------------------------------------------------

	/** Tactical observer (71 features) */
	UPROPERTY(EditAnywhere, Instanced, Category = "Schola|Components")
	UTacticalObserver* TacticalObserver = nullptr;

	/** Tactical reward provider (combat events) */
	UPROPERTY(EditAnywhere, Instanced, Category = "Schola|Components")
	UTacticalRewardProvider* RewardProvider = nullptr;

	/** Reference to follower agent component (auto-found) */
	UPROPERTY(BlueprintReadOnly, Category = "Schola|State")
	UFollowerAgentComponent* FollowerAgent = nullptr;

	/** Reference to Schola's InferenceComponent (auto-found) */
	UPROPERTY(BlueprintReadOnly, Category = "Schola|State")
	class UInferenceComponent* InferenceComponent = nullptr;

	//--------------------------------------------------------------------------
	// UTILITY
	//--------------------------------------------------------------------------

	/** Initialize Schola components (observers, actuators, rewards) */
	UFUNCTION(BlueprintCallable, Category = "Schola")
	void InitializeScholaComponents();

	/** Get reward from combat events */
	UFUNCTION(BlueprintPure, Category = "Schola")
	float GetCurrentReward() const;

	/** Check if episode has terminated */
	UFUNCTION(BlueprintPure, Category = "Schola")
	bool IsEpisodeTerminated() const;

	/** Reset episode for new training round */
	UFUNCTION(BlueprintCallable, Category = "Schola")
	void ResetEpisode();

private:
	/** Find inference component on owner */
	class UInferenceComponent* FindInferenceComponent() const;
	/** Find follower agent component on owner */
	UFollowerAgentComponent* FindFollowerAgent() const;

	/** Configure observers (TacticalObserver) */
	void ConfigureObservers();

	/** Configure reward provider */
	void ConfigureRewardProvider();
};
