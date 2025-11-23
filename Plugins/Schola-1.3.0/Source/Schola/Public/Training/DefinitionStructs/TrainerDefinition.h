// Copyright (c) 2024-2025 Advanced Micro Devices, Inc. All Rights Reserved.

#pragma once

#include "CoreMinimal.h"
#include "Common/InteractionDefinition.h"
#include "Training/DefinitionStructs/AgentId.h"
THIRD_PARTY_INCLUDES_START
#include "../Generated/Definitions.pb.h"
THIRD_PARTY_INCLUDES_END
#include "TrainerDefinition.generated.h"


/**
 * @brief Struct containing the properties that define an agent.
*/
USTRUCT(BlueprintType)
struct SCHOLA_API FTrainerDefinition
{

	GENERATED_BODY()

	/* The Id of the agent. */
	UPROPERTY(EditAnywhere, Category = "Reinforcement Learning")
	FAgentId Id;

	/* The name of this agent. Used for logging etc. Is not a unique identifier. */
	UPROPERTY(EditAnywhere, Category = "Reinforcement Learning")
	FString Name;

	/* The policy definition, stored and linked in the interaction manager */
	FInteractionDefinition* PolicyDefinition;

	/**
	 * @brief Construct a new FTrainerDefinition object
	 * @note This is required for the Unreal Engine reflection system to avoid C4239 and C2512 errors
	 */
	FTrainerDefinition()
	{
		// Just make a definition with default values
	}
	/**
	 * @brief Copy construct a new FTrainerDefinition object
	 * @param[in] Other An existing FTrainerDefinition object to copy
	 */
	FTrainerDefinition(const FTrainerDefinition& Other)
	{
		Name = Other.Name;
		Id = Other.Id;
		PolicyDefinition = Other.PolicyDefinition;
	}

	UPROPERTY(EditAnywhere, Category = "Reinforcement Learning")
	TSubclassOf<APawn> AgentClass;

	/**
	 * @brief Fill a protobuf message (Schola::AgentDefinition) with the contents of this object
	 * @param[out] Msg The protobuf message to fill
	 */
	void ToProtobuf(Schola::AgentDefinition* Msg) const
	{
		this->PolicyDefinition->ActionSpaceDefn.FillProtobuf(Msg->mutable_action_space());
		this->PolicyDefinition->ObsSpaceDefn.FillProtobuf(Msg->mutable_obs_space());

		//TODO return remove this from the msg format
		Msg->set_normalize_obs(false);

		// Placeholder for potential action normalization later.
		Msg->set_normalize_actions(false);

		Msg->set_name(std::string(TCHAR_TO_UTF8(*Name)));
	}
	/**
	 * @brief Convert this object to a protobuf message (Schola::AgentDefinition)
	 * @return A new protobuf message containing the contents of this object
	 */
	Schola::AgentDefinition* ToProtobuf() const
	{
		Schola::AgentDefinition* Msg = new Schola::AgentDefinition();
		this->ToProtobuf(Msg);
		return Msg;
	}
};