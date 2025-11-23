// Copyright (c) 2024-2025 Advanced Micro Devices, Inc. All Rights Reserved.
#pragma once

#include "CoreMinimal.h"
#include "Common/Spaces.h"
#include "Common/Points.h"
#include "Common/LogSchola.h"
#include "Containers/SortedMap.h"
#include "Training/StateStructs/TrainerState.h"
#include "Training/AbstractTrainer.h"
THIRD_PARTY_INCLUDES_START
#include "../Generated/GymConnector.pb.h"
THIRD_PARTY_INCLUDES_END
#include "Communicator/ProtobufSerializer.h"
#include "SharedEnvironmentState.generated.h"

/**
 * @brief A struct representing the state of an environment given by a set of AgentStates.
 */
USTRUCT(BlueprintType)
struct FSharedEnvironmentState
{

	GENERATED_BODY()

	/** Map from AgentId to AgentState */
	TSortedMap<int, FTrainerState*> AgentStates;

	/** 
	 * @brief Default constructor for FSharedEnvironmentState.
	 */
	FSharedEnvironmentState() = default;

	/**
	 * @brief Add a shared agent state to the shared environment state
	 * @param[in] AgentId The key to add the agent state under
	 * @param[in,out] Trainer The trainer to get the State from
	 */
	void AddSharedAgentState(int AgentId, AAbstractTrainer* Trainer)
	{
		this->AgentStates.Add(AgentId,&Trainer->State);
	}

	/**
	 * @brief Fill a protobuf message (Schola::EnvironmentState) with the contents of this object
	 * @param[out] OutMsg A reference to the protobuf message to fill
	 */
	void ToProto(Schola::EnvironmentState& OutMsg) const
	{
		for (const TPair<int, FTrainerState*>& IdToSharedState : AgentStates)
		{
			FTrainerState* State = IdToSharedState.Value;
			//Send the message if we are running, or if we just completed and are sending the last state
			if (State->LastStatus == EAgentTrainingStatus::Running || State->TrainingStatus == EAgentTrainingStatus::Running)
			{
				Schola::AgentState& AgentStateMsg = (*OutMsg.mutable_agent_states())[IdToSharedState.Key];
				IdToSharedState.Value->ToProto(AgentStateMsg);
			}
		}
	}

	/**
	 * @brief Fill a protobuf message (Schola::EnvironmentState) with the contents of this object
	 * @param[out] OutMsg A pointer to the protobuf message to fill
	 */
	void ToProto(Schola::EnvironmentState* OutMsg) const
	{
		return this->ToProto(*OutMsg);
	}

	/**
	 * @brief Convert this object to a protobuf message (Schola::EnvironmentState)
	 * @return A new protobuf message containing the contents of this object
	 */
	Schola::EnvironmentState* ToProto() const
	{
		Schola::EnvironmentState* EnvironmentStateMessage = new Schola::EnvironmentState();
		this->ToProto(EnvironmentStateMessage);
		return EnvironmentStateMessage;
	}

	/**
	 * @brief Fill a protobuf message (Schola::InitialEnvironmentState) with the initial state of the environment after a reset.
	 * @param[out] OutTrainingStateMessage The protobuf message reference to fill
	 */
	void ToResetProto(Schola::InitialEnvironmentState& OutTrainingStateMessage) const
	{
		for (const TPair<int, FTrainerState*>& AgentIdToState : AgentStates)
		{
			if(AgentIdToState.Value->TrainingStatus == EAgentTrainingStatus::Running && AgentIdToState.Value->bExists)
			{
				AgentIdToState.Value->ToResetProto((*OutTrainingStateMessage.mutable_agent_states())[AgentIdToState.Key]);
			}
		}
	}
};