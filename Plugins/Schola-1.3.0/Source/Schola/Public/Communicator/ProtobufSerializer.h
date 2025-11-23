// Copyright (c) 2024 Advanced Micro Devices, Inc. All Rights Reserved.
#pragma once
#include "Common/Points.h"
#include "Common/Points/PointVisitor.h"	
THIRD_PARTY_INCLUDES_START
#include "../Generated/Spaces.pb.h"
THIRD_PARTY_INCLUDES_END

using Schola::DictPoint;
using Schola::BinaryPoint;
using Schola::DiscretePoint;
using Schola::BoxPoint;

/**
 * @brief A class that serializes Points into the corresponding protobuf messages
 */
class ProtobufSerializer : public ConstPointVisitor
{

	DictPoint* PointContainer;

public:
	// Takes as argument the protobuf object that should be filled when traversing the datastructure
	ProtobufSerializer(DictPoint* InitialPoint)
		: PointContainer(InitialPoint){};

	void Visit(const FBinaryPoint& Point) override
	{
		BinaryPoint* PointMsg = PointContainer->add_values()->mutable_binary_point();
		for (auto& PointValue : Point.Values)
		{
			PointMsg->add_values(PointValue);
		}
	};

	void Visit(const FDiscretePoint& Point) override
	{
		DiscretePoint* PointMsg = PointContainer->add_values()->mutable_discrete_point();
		// PointMsg->mutable_values()->Add(Point.Values.begin(), Point.Values.end()); leads to a compile error here
		for (auto& PointValue : Point.Values)
		{
			PointMsg->add_values(PointValue);
		}
	};

	void Visit(const FBoxPoint& Point) override
	{
		BoxPoint* PointMsg = PointContainer->add_values()->mutable_box_point();
		for (auto& PointValue : Point.Values)
		{
			PointMsg->add_values(PointValue);
		}
	};

	DictPoint* GetDictPoint()
	{
		return PointContainer;
	}
};