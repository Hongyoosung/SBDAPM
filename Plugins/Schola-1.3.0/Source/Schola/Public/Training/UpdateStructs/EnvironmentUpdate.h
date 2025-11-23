// Copyright (c) 2024-2025 Advanced Micro Devices, Inc. All Rights Reserved.

#pragma once

#include "CoreMinimal.h"
#include "Containers/SortedMap.h"
#include "Agent/AgentAction.h"
#include "EnvironmentUpdate.generated.h"

/**
 * @brief A Struct representing an update to an environment in the form of a step.
 */
USTRUCT(BlueprintType)
struct SCHOLA_API FEnvStep
{
	GENERATED_BODY()
	/** Map from Agent Id, to Action */
	TSortedMap<int, FAction> Actions;

};

/**
 * @brief A Struct representing an update to an environment in the form of a reset.
 */
USTRUCT(BlueprintType)
struct SCHOLA_API FEnvReset
{
	GENERATED_BODY()

	/** Options Mapping passed from Gym */
	UPROPERTY(EditAnywhere, Category = "Environment Reset Options")
	TMap<FString, FString> Options;

	/** Seed for the environment */
	UPROPERTY(EditAnywhere, Category = "Environment Reset Options")
	int					   Seed = 0;

	/** Whether a seed was supplied as part of the reset */
	UPROPERTY(EditAnywhere, Category = "Environment Reset Options")
	bool bHasSeed = false;

};

/**
 * @brief A Struct representing an update to an environment in the form of either a reset or step. 
 */
USTRUCT(BlueprintType)
struct SCHOLA_API FEnvUpdate
{
	GENERATED_BODY()
	/** A variant containing either a reset or a step */
	TVariant<FEnvReset, FEnvStep> Update;

	/**
	 * @brief Construct a new default FEnvUpdate object
	 */
	FEnvUpdate()
	{

	}

	/**
	 * @brief Construct a new step FEnvUpdate object
	 * @param[in] EnvStep The step to create the object from
	 */
	FEnvUpdate(FEnvStep& EnvStep)
		: Update(TInPlaceType<FEnvStep>(), EnvStep)
	{

	}

	/**
	 * @brief Construct a new reset FEnvUpdate object
	 * @param[in] EnvReset The reset to create the object from
	 */
	FEnvUpdate(FEnvReset& EnvReset) : Update(TInPlaceType<FEnvReset>(), EnvReset)
	{

	}

	/**
	 * @brief Check if the update is a reset
	 * @return true iff the update is a reset
	 */
	bool IsReset() const
	{
		return Update.IsType<FEnvReset>();
	}

	/**
	 * @brief Check if the update is a step
	 * @return true iff the update is a step
	 */
	bool IsStep() const
	{
		return Update.IsType<FEnvStep>();
	}

	/**
	 * @brief Get the step contained in the update
	 * @return The step contained in the update
	 */
	const FEnvStep& GetStep() const
	{
		return Update.Get<FEnvStep>();
	}

	/**
	 * @brief Get the reset contained in the update
	 * @return The reset contained in the update
	 */
	const FEnvReset& GetReset() const
	{
		return Update.Get<FEnvReset>();
	}

};

