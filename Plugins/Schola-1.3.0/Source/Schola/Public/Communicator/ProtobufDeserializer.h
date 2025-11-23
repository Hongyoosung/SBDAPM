// Copyright (c) 2023 Advanced Micro Devices, Inc. All Rights Reserved.

#pragma once

#include "../Common/Points.h"
#include "../Training/UpdateStructs/TrainingUpdate.h"
#include "../Training/UpdateStructs/EnvironmentUpdate.h"
THIRD_PARTY_INCLUDES_START
#include "../Generated/GymConnector.pb.h"
#include "../Generated/Spaces.pb.h"
THIRD_PARTY_INCLUDES_END


/**
 * @brief A namespace containing functions to deserialize protobuf messages into Unreal Engine types
 */
namespace ProtobufDeserializer
{
	/**
	 * @brief Deserialize a protobuf message (Schola::FundamentalPoint) into a new object of type TPoint
	 * @param[in] ProtoMsg The protobuf message to deserialize
	 * @param[out] OutPoint The type of the Unreal object to deserialize into
	 */
	void Deserialize(const Schola::FundamentalPoint& ProtoMsg, TPoint& OutPoint);
	
	/**
	 * @brief Deserialize a protobuf message (Schola::DictPoint) into a new object of type FDictPoint
	 * @param[in] ProtoMsg The protobuf message to deserialize
	 * @param[out] OutPoint The type of the Unreal object to deserialize into
	 */
	void Deserialize(const Schola::DictPoint& ProtoMsg, FDictPoint& OutPoint);

	/**
	 * @brief Deserialize a protobuf message (Schola::EnvironmentStep) into a new object of type FBinaryPoint
	 * @param[in] ProtoMsg The protobuf message to deserialize
	 * @param[out] OutEnvStep The type of the Unreal object to deserialize into
	 */
	void Deserialize(const Schola::EnvironmentStep& ProtoMsg, FEnvStep& OutEnvStep);

	/**
	 * @brief Deserialize a protobuf message (Schola::EnvironmentReset) into a new object of type FEnvReset
	 * @param[in] ProtoMsg The protobuf message to deserialize
	 * @param[out] OutEnvReset The type of the Unreal object to deserialize into
	 */
	void Deserialize(const Schola::EnvironmentReset& ProtoMsg, FEnvReset& OutEnvReset);

	/**
	 * @brief Deserialize a protobuf message (Schola::EnvironmentStateUpdate) into a new object of type FEnvUpdate
	 * @param[in] ProtoMsg The protobuf message to deserialize
	 * @param[out] OutEnvUpdate The type of the Unreal object to deserialize into
	 */
	void Deserialize(const Schola::EnvironmentStateUpdate& ProtoMsg, FEnvUpdate& OutEnvUpdate);

	/**
	 * @brief Deserialize a protobuf message (Schola::TrainingStateUpdate) into a new object of type FTrainingStateUpdate
	 * @param[in] ProtoMsg The protobuf message to deserialize
	 * @param[out] OutTrainingStateUpdate The type of the Unreal object to deserialize into
	 */
	void Deserialize(const Schola::TrainingStateUpdate& ProtoMsg, FTrainingStateUpdate& OutTrainingStateUpdate);

	/**
	 * @brief Deserialize a protobuf message (Schola::AgentStateUpdate) into a new object of type FAction
	 * @param[in] ProtoMsg The protobuf message to deserialize
	 * @param[out] OutAction The type of the Unreal object to deserialize into
	 */
	void Deserialize(const Schola::AgentStateUpdate& ProtoMsg, FAction& OutAction);

	/**
	 * @brief Deserialize a protobuf message into a new object of type UnrealType
	 * @tparam ProtoType The type of the protobuf message
	 * @tparam UnrealType The type of the Unreal object to deserialize into
	 * @param ProtoMsg The protobuf message to deserialize
	 * @return A new object of type UnrealType
	 */
	template<typename ProtoType,typename UnrealType>
	UnrealType* Deserialize(const ProtoType& ProtoMsg){
		UnrealType* Object = new UnrealType();
		Deserialize(ProtoMsg, *Object);
		return Object;
	};


};


