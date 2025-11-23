// Copyright (c) 2023-2025 Advanced Micro Devices, Inc. All Rights Reserved.

#pragma once
#include "CoreMinimal.h"

#include "Subsystem/SubsystemSettings/TrainingSettings.h"
#include "Subsystem/SubsystemSettings/StableBaselines/SB3LoggingSettings.h"
#include "Subsystem/SubsystemSettings/StableBaselines/SB3CheckpointSettings.h"
#include "Subsystem/SubsystemSettings/StableBaselines/SB3ResumeSettings.h"
#include "Subsystem/SubsystemSettings/StableBaselines/SB3NetworkArchitectureSettings.h"
#include "Subsystem/SubsystemSettings/StableBaselines/Algorithms/SB3PPOSettings.h"
#include "Subsystem/SubsystemSettings/StableBaselines/Algorithms/SB3SACSettings.h"
#include "Subsystem/SubsystemSettings/ArgBuilder.h"

#include "SB3TrainingSettings.generated.h"

/**
 * @brief An enumeration of training algorithms supported by SB3
 */
UENUM()
enum class ESB3TrainingAlgorithm
{
	PPO,
	SAC
};

/**
 * @brief A struct to hold all the settings for an SB3 training script
 * @note This is a partial implementation of the SB3 settings, and is not exhaustive
 */
USTRUCT(BlueprintType)
struct SCHOLA_API FSB3TrainingSettings : public FTrainingSettings
{
	GENERATED_BODY()

public:
	/** The number of timesteps to train for */
	UPROPERTY(EditAnywhere, BlueprintReadOnly, meta = (ClampMin = 0), Category = "Training Settings")
	int Timesteps = 8000;

	/* Logging related arguments */
	UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "Training Settings")
	FSB3LoggingSettings LoggingSettings;

	/** Checkpoint related arguments */
	UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "Training Settings")
	FSB3CheckpointSettings CheckpointSettings;

	/** Resume related arguments */
	UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "Training Settings")
	FSB3ResumeSettings ResumeSettings;

	/** Network architecture related arguments */
	UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "Training Settings")
	FSB3NetworkArchSettings NetworkArchitectureSettings;

	/** Display a progress bar during training */
	UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "Training Settings")
	bool bDisplayProgressBar = true;

	/** The algorithm to use during training (e.g. SAC, PPO) */
	UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "Training Settings")
	ESB3TrainingAlgorithm Algorithm = ESB3TrainingAlgorithm::PPO;

	/** PPO specific settings */
	UPROPERTY(EditAnywhere, BlueprintReadOnly, meta = (EditCondition = "Algorithm==ESB3TrainingAlgorithm::PPO", EditConditionHides, DisplayName = "PPO Algorithm Settings"), Category="Training Settings")
	FSB3PPOSettings PPOSettings;

	/** SAC specific settings */
	UPROPERTY(EditAnywhere, BlueprintReadOnly, meta = (EditCondition = "Algorithm==ESB3TrainingAlgorithm::SAC", EditConditionHides, DisplayName = "SAC Algorithm Settings"), Category="Training Settings")
	FSB3SACSettings SACSettings;

	void GenerateTrainingArgs(int Port, FScriptArgBuilder& ArgBuilder) const;

	virtual ~FSB3TrainingSettings();
};