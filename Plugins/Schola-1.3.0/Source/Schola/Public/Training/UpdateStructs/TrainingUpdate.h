// Copyright (c) 2024-2025 Advanced Micro Devices, Inc. All Rights Reserved.

#pragma once

#include "CoreMinimal.h"
#include "Containers/SortedMap.h"
#include "Training/UpdateStructs/EnvironmentUpdate.h"
#include "TrainingUpdate.generated.h"

/**
 * @brief An enumeration representing the type of action taken by an agent.
 */
UENUM(BlueprintType)
enum class EConnectorStatusUpdate : uint8
{
	NONE = 0 UMETA(DisplayName = "No New Status"),
	ERRORED = 1 UMETA(DisplayName = "Error"),
	CLOSED = 2 UMETA(DisplayName = "Closed"),
};

/**
 * @brief A Struct representing an update to the overall training state.
 */
USTRUCT(BlueprintType)
struct SCHOLA_API FTrainingStateUpdate
{
	GENERATED_BODY()

	/** Map from Environment Id, to Environment Update */
	TSortedMap<int, FEnvUpdate> EnvUpdates;

	/** The status of the connector */
	UPROPERTY()
	EConnectorStatusUpdate Status = EConnectorStatusUpdate::NONE;

	/**
	 * @brief Construct a new default FTrainingStateUpdate object
	 */
	FTrainingStateUpdate()
	{

	}

	/**
	 * @brief Is the update an error
	 * @return true iff source of the update experienced an error
	 */
	bool IsError() const
	{
		return Status == EConnectorStatusUpdate::ERRORED;
	}

	/**
	 * @brief Is the update a close
	 * @return true iff the source of the update has closed
	 */
	bool IsClose() const
	{
		return Status == EConnectorStatusUpdate::CLOSED;
	}
};
