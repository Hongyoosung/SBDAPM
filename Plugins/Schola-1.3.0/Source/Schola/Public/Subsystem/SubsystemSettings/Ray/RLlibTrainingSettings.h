// Copyright (c) 2024-2025 Advanced Micro Devices, Inc. All Rights Reserved.

#pragma once
#include "CoreMinimal.h"

#include "Subsystem/SubsystemSettings/ArgBuilder.h"
#include "Subsystem/SubsystemSettings/TrainingSettings.h"
#include "Subsystem/SubsystemSettings/Ray/RLlibLoggingSettings.h"
#include "Subsystem/SubsystemSettings/Ray/RLlibCheckpointSettings.h"
#include "Subsystem/SubsystemSettings/Ray/RLlibResumeSettings.h"
#include "Subsystem/SubsystemSettings/Ray/RLlibNetworkArchitectureSettings.h"
#include "Subsystem/SubsystemSettings/Ray/RLlibResourceSettings.h"
#include "Subsystem/SubsystemSettings/Ray/Algorithms/RLlibPPOSettings.h"
#include "Subsystem/SubsystemSettings/Ray/Algorithms/RLlibAPPOSettings.h"
#include "Subsystem/SubsystemSettings/Ray/Algorithms/RLlibIMPALASettings.h"

#include "RLlibTrainingSettings.generated.h"


UENUM()
enum class ERLlibTrainingAlgorithm
{
	PPO,
	APPO,
	IMPALA
};


/**
 * @brief A struct to hold all the settings for an RLlib training script
 */
USTRUCT(BlueprintType)
struct SCHOLA_API FRLlibTrainingSettings : public FTrainingSettings
{
	GENERATED_BODY()

public:
	/** The number of timesteps to train for */
	UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "Training Settings")
	int Timesteps = 8000;

	UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "Training Settings")
	float LearningRate = 0.0003;

	UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "Training Settings")
	int MinibatchSize = 128;

	UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "Training Settings")
	int TrainBatchSizePerLearner = 256;

	UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "Training Settings")
	int NumSGDIter = 5;

	UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "Training Settings")
	float Gamma = 0.99;

	/** The logging settings for the training script */
	UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "Logging Settings")
	FRLlibLoggingSettings LoggingSettings;

	/** The checkpoint settings for the training script */
	UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "Checkpoint Settings")
	FRLlibCheckpointSettings CheckpointSettings;

	/** The resume settings for the training script */
	UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "Resume Settings")
	FRLlibResumeSettings ResumeSettings;

	/** The network architecture settings for the training script */
	UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "Network Architecture Settings")
	FRLlibNetworkArchSettings NetworkArchitectureSettings;

	/** The resource settings for the training script */
	UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "Resource Settings")
	FRLlibResourceSettings ResourceSettings;

	/** The algorithm to use during training (e.g. SAC, PPO) */
	UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "Algorithm Settings")
	ERLlibTrainingAlgorithm Algorithm = ERLlibTrainingAlgorithm::PPO;

	/** PPO specific settings */
	UPROPERTY(EditAnywhere, BlueprintReadOnly, meta = (EditCondition = "Algorithm==ERLlibTrainingAlgorithm::PPO", EditConditionHides, DisplayName = "PPO Algorithm Settings"), Category = "Algorithm Settings")
	FRLlibPPOSettings PPOSettings;

	/** APPO specific settings */
	UPROPERTY(EditAnywhere, BlueprintReadOnly, meta = (EditCondition = "Algorithm==ERLlibTrainingAlgorithm::APPO", EditConditionHides, DisplayName = "APPO Algorithm Settings"), Category = "Algorithm Settings")
	FRLlibAPPOSettings APPOSettings;

	/** IMPALA specific settings */
	UPROPERTY(EditAnywhere, BlueprintReadOnly, meta = (EditCondition = "Algorithm==ERLlibTrainingAlgorithm::IMPALA", EditConditionHides, DisplayName = "IMPALA Algorithm Settings"), Category = "Algorithm Settings")
	FRLlibIMPALASettings IMPALASettings;

	void GenerateTrainingArgs(int Port, FScriptArgBuilder& ArgBuilder) const;

	virtual ~FRLlibTrainingSettings();
};
