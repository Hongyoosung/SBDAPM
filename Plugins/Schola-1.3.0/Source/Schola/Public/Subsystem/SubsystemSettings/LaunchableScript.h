// Copyright (c) 2024-2025 Advanced Micro Devices, Inc. All Rights Reserved.

#pragma once
#include "CoreMinimal.h"

#include "LaunchableScript.generated.h"

/**
 * @brief A struct to hold the configuration of a launchable script.
 */
USTRUCT(BlueprintType)
struct SCHOLA_API FLaunchableScript
{
	GENERATED_BODY()

public:
	/**
	 * @brief Construct a launchable script with no arguments or URL. Will fail to launch unless ScriptURL is set
	 */
	FLaunchableScript();

	/**
	 * @brief Construct a launchable script with a file URL
	 * @param[in] ScriptURL The URL of the script to launch
	 */
	FLaunchableScript(FString ScriptURL);

	/**
	 * @brief Construct a launchable script with a file URL and arguments
	 * @param[in] ScriptURL The URL of the script to launch
	 * @param[in] Args The arguments to pass to the script
	 */
	FLaunchableScript(FString ScriptURL, FString Args);

	/** A path to the script to be launched */
	UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "Networking URL")
	FString ScriptURL;

	/** The arguments to be passed to the script */
	UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "Networking URL")
	FString Args;

	/** 
	 * @brief Append additional arguments to the script 
	 * @param[in] AdditionalArgs The arguments to append to the script
	 */
	void AppendArgs(FString& AdditionalArgs);

	/**
	 * @brief Launch the script
	 * @return The process handle of the launched script
	 */
	FProcHandle LaunchScript() const;
};