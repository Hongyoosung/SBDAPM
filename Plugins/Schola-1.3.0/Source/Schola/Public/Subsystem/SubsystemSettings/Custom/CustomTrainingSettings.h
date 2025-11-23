// Copyright (c) 2024-2025 Advanced Micro Devices, Inc. All Rights Reserved.
#pragma once

#include "CoreMinimal.h"
#include "Subsystem/SubsystemSettings/ArgBuilder.h"
#include "Subsystem/SubsystemSettings/TrainingSettings.h"
#include "CustomTrainingSettings.generated.h"

/**
 * @brief A struct to hold settings for a custom training script
 */
USTRUCT(BlueprintType)
struct SCHOLA_API FCustomTrainingSettings : public FTrainingSettings
{
	GENERATED_BODY()

public:
	/** The path to the script to launch */
	UPROPERTY(Config, EditAnywhere, Category = "Script Settings|Custom Script")
	FFilePath LaunchScript;

	/** The arguments to pass to the script */
	UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "Script Settings|Custom Script")
	TMap<FString, FString> Args;

	/** The flags to pass to the script */
	UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "Script Settings|Custom Script")
	TArray<FString> Flags;

	void GenerateTrainingArgs(int Port, FScriptArgBuilder& ArgBuilder) const;

	virtual ~FCustomTrainingSettings();
};