// Copyright (c) 2024-2025 Advanced Micro Devices, Inc. All Rights Reserved.

#pragma once

#include "CoreMinimal.h"
#include "Training/DefinitionStructs/TrainerDefinition.h"
THIRD_PARTY_INCLUDES_START
#include "../Generated/Definitions.pb.h"
THIRD_PARTY_INCLUDES_END
#include "EnvironmentDefinition.generated.h"


/**
 * @brief Struct containing the properties that define an environment. Shared between the GymConnector and the Environment Objects.
 */
USTRUCT(BlueprintType)
struct SCHOLA_API FEnvironmentDefinition
{
	GENERATED_BODY()

	/** Map from Environment Name,Agent Name to Agent Definitions */
	TSortedMap<int, FTrainerDefinition*> AgentDefinitions;

	/**
	 * @brief Fill a protobuf message (Schola::EnvironmentDefinition) with the contents of this object
	 * @param[out] Msg The protobuf message to fill
	 */
	void ToProtobuf(Schola::EnvironmentDefinition* Msg) const
	{
		for (const TTuple<int, FTrainerDefinition*>& IdToAgentDefn : AgentDefinitions)
		{
			Schola::AgentDefinition AgentDefnMessage;
			IdToAgentDefn.Value->ToProtobuf(&AgentDefnMessage);
			(*Msg->mutable_agent_definitions())[IdToAgentDefn.Key] = AgentDefnMessage;
		}
	}

	/**
	 * @brief Add a shared agent definition to the shared environment definition
	 * @param[in] Key The key to add the agent definition under
	 * @param[in] SharedDefnPointer The shared agent definition to add
	 */
	void AddSharedAgentDefn(int Key, FTrainerDefinition* SharedDefnPointer)
	{
		this->AgentDefinitions.Add(Key, SharedDefnPointer);
	}

};