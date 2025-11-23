// Copyright (c) 2023-2025 Advanced Micro Devices, Inc. All Rights Reserved.

#include "Common/Spaces/BinarySpace.h"

FBinarySpace::FBinarySpace()
{
}

FBinarySpace::FBinarySpace(int Shape)
{
	this->Shape = Shape;
}

void FBinarySpace::Merge(const FBinarySpace& Other)
{
	this->Shape += Other.Shape;
}

void FBinarySpace::Copy(const FBinarySpace& Other)
{
	this->Shape = Other.Shape;
}

void FBinarySpace::FillProtobuf(FundamentalSpace* Msg) const
{

	this->FillProtobuf(Msg->mutable_binary_space());
}

int FBinarySpace::GetNumDimensions() const
{
	return Shape;
}

ESpaceValidationResult FBinarySpace::Validate(TPoint& Observation) const
{

	// Is the shape right?
	if (!Observation.IsType<FBinaryPoint>())
	{
		return ESpaceValidationResult::WrongDataType;
	}
	FBinaryPoint& TypedObservation = Observation.Get<FBinaryPoint>();


	if (Shape != TypedObservation.Values.Num())
	{
		return ESpaceValidationResult::WrongDimensions;
	}
	else
	{
		return ESpaceValidationResult::Success;
	}
}


void FBinarySpace::FillProtobuf(BinarySpace* Msg) const
{
	this->FillProtobuf(*Msg);
}

void FBinarySpace::FillProtobuf(BinarySpace& Msg) const
{
	Msg.set_shape(Shape);
}

TPoint FBinarySpace::UnflattenAction(const TArray<float>& Data, int Offset) const
{
	TArray<bool> BoolArray;
	for (int i = 0; i < GetNumDimensions(); i++)
	{
		BoolArray.Add(static_cast<bool>(Data[i + Offset]));
	}
	return TPoint(TInPlaceType<FBinaryPoint>(), BoolArray);
}

void FBinarySpace::FlattenPoint(TArrayView<float> Buffer, const TPoint& Point) const
{
	assert(Buffer.Num() == this->GetFlattenedSize());
	TArray<bool> Arr = Point.Get<FBinaryPoint>().Values;
	for (int i = 0; i < Arr.Num(); i++)
	{
		Buffer[i] = Arr[i];
	}
}

int FBinarySpace::GetFlattenedSize() const
{
	return this->Shape;
}

bool FBinarySpace::IsEmpty() const
{
	return this->Shape == 0;
}

TPoint FBinarySpace::MakeTPoint() const
{
	return TPoint(TInPlaceType<FBinaryPoint>());
}