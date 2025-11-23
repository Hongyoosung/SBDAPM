// Copyright (c) 2024-2025 Advanced Micro Devices, Inc. All Rights Reserved.

#pragma once

#include "CoreMinimal.h"
#include "Training/DefinitionStructs/EnvironmentDefinition.h"
THIRD_PARTY_INCLUDES_START
#include "../Generated/Definitions.pb.h"
THIRD_PARTY_INCLUDES_END
#include "TrainingDefinition.generated.h"


/**
 * @brief Struct containing the properties that define a training session.
 */
USTRUCT(BlueprintType)
struct SCHOLA_API FTrainingDefinition
{
	GENERATED_BODY()
	/** Map from EnvironmentID to Environment Definition, which is itself a Map from AgentId to AgentDefinition */
	UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "Reinforcement Learning")
	TArray<FEnvironmentDefinition> EnvironmentDefinitions;

	/**
	 * @brief Fill a protobuf message (Schola::TrainingDefinition) with the contents of this object
	 * @return A new protobuf message containing the contents of this object
	 */
	Schola::TrainingDefinition* ToProtobuf() const
	{
		Schola::TrainingDefinition* Msg = new Schola::TrainingDefinition();
		for (const FEnvironmentDefinition& EnvDefn : EnvironmentDefinitions)
		{
			Schola::EnvironmentDefinition* EnvDefnMessage = Msg->add_environment_definitions();
			EnvDefn.ToProtobuf(EnvDefnMessage);
		}

		return Msg;
	}
};