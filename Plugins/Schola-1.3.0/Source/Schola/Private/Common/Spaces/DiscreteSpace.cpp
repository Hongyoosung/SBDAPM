// Copyright (c) 2023-2025 Advanced Micro Devices, Inc. All Rights Reserved.

#include "Common/Spaces/DiscreteSpace.h"


FDiscreteSpace::FDiscreteSpace()
{
}

FDiscreteSpace::FDiscreteSpace(TArray<int>& High)
	: High(High)
{
}

void FDiscreteSpace::Copy(const FDiscreteSpace& Other)
{
	this->High = TArray<int>(Other.High);
}

void FDiscreteSpace::Merge(const FDiscreteSpace& Other)
{
	this->High.Append(Other.High);
}

void FDiscreteSpace::Add(int DimSize)
{
	this->High.Add(DimSize);
}

ESpaceValidationResult FDiscreteSpace::Validate(TPoint& Observation) const
{
	// Is the shape right?
	if (!Observation.IsType<FDiscretePoint>())
	{
		return ESpaceValidationResult::WrongDataType;
	}
	FDiscretePoint& TypedObservation = Observation.Get<FDiscretePoint>();

	if (High.Num() != TypedObservation.Values.Num())
	{
		return ESpaceValidationResult::WrongDimensions;
	}

	// Are all our values in our bounding box?
	for (int i = 0; i < High.Num(); i++)
	{
		//Note, we are exclusive of our high values here
		if (TypedObservation.Values[i] >= High[i] || TypedObservation.Values[i] < 0)
		{
			return ESpaceValidationResult::OutOfBounds;
		}
	}

	return ESpaceValidationResult::Success;
}

void FDiscreteSpace::FillProtobuf(FundamentalSpace* Msg) const
{
	this->FillProtobuf(Msg->mutable_discrete_space());
}

int FDiscreteSpace::GetNumDimensions() const
{
	return High.Num();
}

FDiscreteSpace::~FDiscreteSpace()
{
	this->High.Empty();
}

void FDiscreteSpace::FillProtobuf(DiscreteSpace* Msg) const
{
	this->FillProtobuf(*Msg);
}

void FDiscreteSpace::FillProtobuf(DiscreteSpace& Msg) const
{
	for (const int& HighValue : High)
	{
		Msg.add_high(HighValue);
	}
}

int FDiscreteSpace::GetMaxValue(const TArray<float>& Vector) const
{
	float CurrMax = Vector[0];
	int	  Index = 0;
	int	  CurrIndex = 0;
	for (const float& Value : Vector)
	{
		if (Value > CurrMax)
		{
			CurrMax = Value;
			Index = CurrIndex;
		}
		CurrIndex += 1;
	}
	return Index;
}

TPoint FDiscreteSpace::UnflattenAction(const TArray<float>& Data, int Offset) const
{
	
	TPoint Point = this->MakeTPoint();
	FDiscretePoint& TypedPoint = Point.Get<FDiscretePoint>();
	int			   OutputValue = 0;
	int			   CurrIndex = 0;
	int			   dim = GetNumDimensions();
	//TODO remove the copy from this function
	for (int i = 0; i < dim; i++)
	{
		TArray<float> BranchArray = {};
		int			  BranchHigh = High[i];
		for (int j = CurrIndex; j < BranchHigh + CurrIndex; j++)
		{
			BranchArray.Add(Data[j + Offset]);
		}
		OutputValue = GetMaxValue(BranchArray);
		TypedPoint.Values.Add(OutputValue);
		CurrIndex += High[i];
	}

	return Point;
}

void FDiscreteSpace::FlattenPoint(TArrayView<float> Buffer, const TPoint& Point) const
{
	//WE assume that the buffer is zeroed out
	assert(Buffer.Num() == this->GetFlattenedSize());
	
	const TArray<int>& Arr = Point.Get<FDiscretePoint>().Values;
	int			BranchStart = 0;
	for (int i = 0; i < this->High.Num(); i++)
	{
		Buffer[Arr[i] + BranchStart] = 1;
		BranchStart += this->High[i];
	}
}

int FDiscreteSpace::GetFlattenedSize() const
{
	int Size = 0;
	for (const int& HighValue : High)
	{
		Size += HighValue;
	}
	return Size;
}

bool FDiscreteSpace::IsEmpty() const
{
	return this->GetNumDimensions() == 0;
}

TPoint FDiscreteSpace::MakeTPoint() const
{
	return TPoint(TInPlaceType<FDiscretePoint>());
}