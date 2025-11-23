// Copyright (c) 2024-2025 Advanced Micro Devices, Inc. All Rights Reserved.
#pragma once

#include "CoreMinimal.h"
#include "Common/LogSchola.h"
#include "Training/StateStructs/SharedEnvironmentState.h"
THIRD_PARTY_INCLUDES_START
#include "../Generated/GymConnector.pb.h"
THIRD_PARTY_INCLUDES_END
#include "Communicator/ProtobufSerializer.h"
#include "TrainingState.generated.h"

/**
 * @brief A Struct representing the state of the training session given by a set of EnvironmentStates
 */
USTRUCT(BlueprintType)
struct FTrainingState
{
	GENERATED_BODY()

	/** Map from EnvironmentId to EnvironmentState */
	TArray<FSharedEnvironmentState> EnvironmentStates;

	FTrainingState(){};

	/**
	 * @brief Convert this object to a protobuf message (Schola::TrainingState)
	 * @return A new protobuf message containing the contents of this object
	 */
	Schola::TrainingState* ToProto() const
	{
		Schola::TrainingState* TrainingStateMessage = new Schola::TrainingState();
		for (const FSharedEnvironmentState& EnvState : EnvironmentStates)
		{
			// Fill the mappings
			EnvState.ToProto((*TrainingStateMessage->add_environment_states()));
		}

		return TrainingStateMessage;
	}

	/**
	 * @brief Convert this object to a protobuf message (Schola::InitialTrainingState) representing the initial state of a subset of environments after a reset.
	 * @param[in] TargetEnvironments The list of environment ids to include in the message
	 * @return A new protobuf message containing the initial state of the specified environments
	 */
	Schola::InitialTrainingState* ToResetProto(const TArray<int>& TargetEnvironments) const
	{
		Schola::InitialTrainingState* TrainingStateMessage = new Schola::InitialTrainingState();
		for (int EnvId : TargetEnvironments)
		{
			const FSharedEnvironmentState& EnvState = EnvironmentStates[EnvId];
			// Fill the mappings
			EnvState.ToResetProto((*TrainingStateMessage->mutable_environment_states())[EnvId]);
		}

		return TrainingStateMessage;
	}
};