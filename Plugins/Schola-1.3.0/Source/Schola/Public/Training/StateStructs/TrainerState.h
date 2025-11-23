// Copyright (c) 2024-2025 Advanced Micro Devices, Inc. All Rights Reserved.
#pragma once

#include "CoreMinimal.h"
#include "Common/Spaces.h"
#include "Common/Points.h"
#include "Common/LogSchola.h"
THIRD_PARTY_INCLUDES_START
#include "../Generated/GymConnector.pb.h"
THIRD_PARTY_INCLUDES_END
#include "Communicator/ProtobufSerializer.h"
#include "TrainerState.generated.h"

/**
 * @brief An enumeration representing the various states common across gym and gym-like environments. Typically use Running, Truncated and Completed
 */
UENUM(BlueprintType)
enum class EAgentTrainingStatus : uint8
{
	Running	  UMETA(DisplayName = "Running"),
	Truncated UMETA(DisplayName = "Truncated"),
	Completed UMETA(DisplayName = "Completed"),
};
/**
 * @brief An enumeration representing the status of the training message
 */
UENUM(BlueprintType)
enum class ETrainingMsgStatus : uint8
{
	NoStatus	   UMETA(DisplayName = "None"),
	LastMsgPending UMETA(DisplayName = "Pending"),
	LastMsgSent	   UMETA(DisplayName = "Sent"),
};

/**
 * @brief A Struct representing the state of the agent given by a Reward, a vector observation and a status
 */
USTRUCT(BlueprintType)
struct FTrainerState
{
	GENERATED_BODY()

	/** The reward received by the agent in the last step */
	UPROPERTY(BlueprintReadOnly, Category = "Trainer State")
	float Reward = 0.0;

	/** Whether we have sent out our last message after completing an episode */
	UPROPERTY(BlueprintReadOnly, Category = "Trainer State")
	EAgentTrainingStatus LastStatus = EAgentTrainingStatus::Running;

	/** The vector observation of the agent in the last step. Not a UProperty because FDictPoint is not blueprintable */
	FDictPoint* Observations;

	/** A map of key-value pairs containing additional information about the agent from the last step*/
	UPROPERTY(EditAnywhere, Category = "Trainer State")
	TMap<FString,FString> Info;

	/** The status of the agent in the last step */
	UPROPERTY(BlueprintReadOnly, Category = "Trainer State")
	EAgentTrainingStatus TrainingStatus = EAgentTrainingStatus::Running;
	
	/** Does the trainer associated with this state exist. */
	UPROPERTY(BlueprintReadOnly, Category = "Trainer State")
	bool bExists = false;

	/** The current step of the agent */
	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "Trainer State")
	int Step = 0;

	/**
	 * @brief Fill a protobuf message (Schola::AgentState) with the agent's state
	 * @param[out] OutState The protobuf message reference to fill
	 */
	void ToProto(Schola::AgentState& OutState) const
	{
		ProtobufSerializer Serializer = ProtobufSerializer(OutState.mutable_observations());

		Observations->Accept(Serializer);
		
		for (auto& InfoEntry : this->Info)
		{
			(*OutState.mutable_info())[TCHAR_TO_UTF8(*InfoEntry.Key)] = TCHAR_TO_UTF8(*InfoEntry.Value);
		}

		OutState.set_reward(this->Reward);

		// convert from Unreal visible enum to gRPC enum
		switch (this->TrainingStatus)
		{
			case EAgentTrainingStatus::Running:
				OutState.set_status(Schola::RUNNING);
				break;
			case EAgentTrainingStatus::Completed:
				OutState.set_status(Schola::COMPLETED);
				break;
			case EAgentTrainingStatus::Truncated:
				OutState.set_status(Schola::TRUNCATED);
				break;
		}
	}

	/** 
	* @brief Reset the Trainer State at the end of an episode
	*/
	void Reset()
	{
		this->Observations->Reset();
		this->Info.Reset();
		this->Step = 0;
	}

	/**
	 * @brief Fill a protobuf message (Schola::AgentState) with the agent's state
	 * @param[out] OutState The protobuf message ptr to fill
	 */
	void ToProto(Schola::AgentState* OutState) const
	{
		return this->ToProto(*OutState);
	}

	/**
	 * @brief Convert this object to a protobuf message (Schola::AgentState)
	 * @return A new protobuf message containing the contents of this object
	 */
	Schola::AgentState* ToProto() const
	{
		Schola::AgentState* AgentStateMsg = new Schola::AgentState();
		this->ToProto(AgentStateMsg);
		return AgentStateMsg;
	}

	/**
	 * @brief Is this agent done the current episode of training.
	 * @return true iff the agent is done it's current episode.
	 */
	bool IsDone() const
	{
		return TrainingStatus == EAgentTrainingStatus::Completed || TrainingStatus == EAgentTrainingStatus::Truncated;
	}

	/**
	 * @brief Create a protobuf message (Schola::InitialAgentState) corresponding to the initial state of the agent after a reset.
	 * @param[out] OutState The protobuf message reference to fill
	 */
	void ToResetProto(Schola::InitialAgentState& OutState) const
	{
		ProtobufSerializer Visitor = ProtobufSerializer(OutState.mutable_observations());
		Observations->Accept(Visitor);
		
		for (auto& InfoEntry : this->Info)
		{
			(*OutState.mutable_info())[TCHAR_TO_UTF8(*InfoEntry.Key)] = TCHAR_TO_UTF8(*InfoEntry.Value);
		}
		
	}

};