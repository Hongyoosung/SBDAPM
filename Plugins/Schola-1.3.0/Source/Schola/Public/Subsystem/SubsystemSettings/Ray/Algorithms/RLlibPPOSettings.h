// Copyright (c) 2024-2025 Advanced Micro Devices, Inc. All Rights Reserved.

#pragma once
#include "CoreMinimal.h"

#include "Subsystem/SubsystemSettings/ArgBuilder.h"
#include "Subsystem/SubsystemSettings/TrainingSettings.h"
#include "RLlibPPOSettings.generated.h"

/**
 * @brief A struct to hold Proximal Policy Optimization(PPO) settings for an RLLib training script
 * @note This is a partial implementation of the PPO settings, and is not exhaustive
 */
USTRUCT(BlueprintType)
struct SCHOLA_API FRLlibPPOSettings : public FTrainingSettings
{
	GENERATED_BODY()

public:
	UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "PPO Settings")
    float GAELambda = 0.95;

    UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "PPO Settings")
    float ClipParam = 0.2;

    UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "PPO Settings")
    bool bUseGAE = true;

	void GenerateTrainingArgs(int Port, FScriptArgBuilder& ArgBuilder) const;

	virtual ~FRLlibPPOSettings();
};
