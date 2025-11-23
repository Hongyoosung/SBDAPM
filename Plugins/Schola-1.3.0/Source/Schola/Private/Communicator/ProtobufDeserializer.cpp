// Copyright (c) 2023 Advanced Micro Devices, Inc. All Rights Reserved.

#include "Communicator/ProtobufDeserializer.h"

using namespace ProtobufDeserializer;

void ProtobufDeserializer::Deserialize(const Schola::FundamentalPoint& ProtoMsg, TPoint& OutPoint)
{
	switch (ProtoMsg.point_case())
	{
		//If we make local variables here then it complains about local variables inside switch statement
		case Schola::FundamentalPoint::kBinaryPoint:
			OutPoint.Emplace<FBinaryPoint>(ProtoMsg.binary_point().values().data(), ProtoMsg.binary_point().values_size());
			break;
		case Schola::FundamentalPoint::kBoxPoint:
			OutPoint.Emplace<FBoxPoint>(ProtoMsg.box_point().values().data(), ProtoMsg.box_point().values_size());
			break;
		case Schola::FundamentalPoint::kDiscretePoint:
			OutPoint.Emplace<FDiscretePoint>(ProtoMsg.discrete_point().values().data(), ProtoMsg.discrete_point().values_size());
			break;
	}
}

void ProtobufDeserializer::Deserialize(const Schola::DictPoint& ProtoMsg, FDictPoint& OutPoint)
{
	for (const Schola::FundamentalPoint& FundPoint : ProtoMsg.values())
	{
		switch (FundPoint.point_case())
		{
			case Schola::FundamentalPoint::kBinaryPoint:
				(void)OutPoint.Points.Emplace_GetRef(TInPlaceType<FBinaryPoint>(), FundPoint.binary_point().values().data(), FundPoint.binary_point().values_size());
				break;
			case Schola::FundamentalPoint::kBoxPoint:
				(void)OutPoint.Points.Emplace_GetRef(TInPlaceType<FBoxPoint>(), FundPoint.box_point().values().data(), FundPoint.box_point().values_size());
				break;
			case Schola::FundamentalPoint::kDiscretePoint:
				(void)OutPoint.Points.Emplace_GetRef(TInPlaceType<FDiscretePoint>(), FundPoint.discrete_point().values().data(), FundPoint.discrete_point().values_size());
				break;
		}

		
	}
}

void ProtobufDeserializer::Deserialize(const Schola::EnvironmentStep& ProtoMsg, FEnvStep& OutEnvStep)
{
	for (auto& AgentUpdateTuple : ProtoMsg.updates())
	{
		FAction& Action = OutEnvStep.Actions.Add(AgentUpdateTuple.first);
		Deserialize(AgentUpdateTuple.second, Action);

	}
}

void ProtobufDeserializer::Deserialize(const Schola::EnvironmentReset& ProtoMsg, FEnvReset& OutEnvReset)
{

	for (auto& Item : ProtoMsg.options())
	{
		OutEnvReset.Options.Add(UTF8_TO_TCHAR(Item.first.c_str()), UTF8_TO_TCHAR(Item.second.c_str()));
	}
	
	if (ProtoMsg.has_seed())
	{
		OutEnvReset.Seed = ProtoMsg.seed();
		OutEnvReset.bHasSeed = true;
	}
}

void ProtobufDeserializer::Deserialize(const Schola::EnvironmentStateUpdate& ProtoMsg, FEnvUpdate& OutEnvUpdate)
{
	switch (ProtoMsg.update_msg_case())
	{
		//Can't make local variables here because of switch statement
		case (Schola::EnvironmentStateUpdate::kReset):
			OutEnvUpdate.Update.Emplace<FEnvReset>();
			Deserialize(ProtoMsg.reset(), OutEnvUpdate.Update.Get<FEnvReset>());
			break;
		case (Schola::EnvironmentStateUpdate::kStep):
			OutEnvUpdate.Update.Emplace<FEnvStep>();
			Deserialize(ProtoMsg.step(), OutEnvUpdate.Update.Get<FEnvStep>());
			break;
		default:
			break;
	}
}

void ProtobufDeserializer::Deserialize(const Schola::TrainingStateUpdate& ProtoMsg, FTrainingStateUpdate& OutTrainingStateUpdate)
{
	OutTrainingStateUpdate.Status = static_cast<EConnectorStatusUpdate>(ProtoMsg.status());
	for (auto& EnvUpdateMsg : ProtoMsg.updates())
	{
		Deserialize(EnvUpdateMsg.second, OutTrainingStateUpdate.EnvUpdates.Add(EnvUpdateMsg.first));
	}
}

void ProtobufDeserializer::Deserialize(const Schola::AgentStateUpdate& ProtoMsg, FAction& OutAction)
{
	Deserialize(ProtoMsg.actions(), OutAction.Values);
}
