// Copyright (c) 2024-2025 Advanced Micro Devices, Inc. All Rights Reserved.

#pragma once
#include "CoreMinimal.h"
#include "Subsystem/SubsystemSettings/ArgBuilder.h"
#include "TrainingSettings.generated.h"

/**
 * @brief Abstract class for any training settings
 */
USTRUCT(BlueprintType)
struct SCHOLA_API FTrainingSettings
{
	GENERATED_BODY()

public:
	/**
	 * @brief Generate the training arguments for the script using the ArgBuilder
	 * @param[in] Port The port to use for the script
	 * @param[in] ArgBuilder The builder to use to generate the arguments
	 * @note port is supplied as it is a common argument to pass to scripts, and is set at a high level but might be needed by specific subsettings
	 */
	virtual void GenerateTrainingArgs(int Port, FScriptArgBuilder& ArgBuilder) const;

	virtual ~FTrainingSettings() {};
};

