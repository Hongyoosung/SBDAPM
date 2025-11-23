// Copyright (c) 2023-2025 Advanced Micro Devices, Inc. All Rights Reserved.

#include "Common/Spaces/BoxSpace.h"

int GetTotalSize(const TArray<int>& Shape)
{
	if(Shape.Num() == 0)
	{
		return 0;
	}
	else
	{
		int TotalSize = 1;
		for (const int Dim : Shape)
		{
			TotalSize *= Dim;
		}
		return TotalSize;
	}
}

int GetTotalSize(std::initializer_list<int> Shape)
{
	if (Shape.size() == 0)
	{
		return 0;
	}
	else
	{
		int TotalSize = 1;
		for (const int Dim : Shape)
		{
			TotalSize *= Dim;
		}
		return TotalSize;
	}
}

FBoxSpace::FBoxSpace()
{
}

FBoxSpace::FBoxSpace(TArray<float>& Low, TArray<float>& High, const TArray<int>& Shape)
{
	verifyf(Low.Num() == High.Num(), TEXT("High and Low must have the same shape"));
	
	int TotalShapeSize = GetTotalSize(Shape);
	if (TotalShapeSize == 0)
	{
		this->Shape = TArray<int>({ High.Num() });
	}
	else
	{
		verifyf(TotalShapeSize == High.Num(), TEXT("High and Low must match the shape of the BoxSpace"));
		this->Shape = Shape;
	}
	//Can use just low here since both high and low are the same size
	for (int i = 0; i < Low.Num(); i++)
	{
		this->Dimensions.Add(FBoxSpaceDimension(Low[i], High[i]));
	}
	this->Shape = Shape;
}

FBoxSpace::FBoxSpace(TArray<FBoxSpaceDimension>& Dimensions, const TArray<int>& Shape)
{
	int TotalShapeSize = GetTotalSize(Shape);
	if (TotalShapeSize == 0)
	{
		this->Shape = TArray<int>({ Dimensions.Num() });
	}
	else
	{
		verifyf(TotalShapeSize == Dimensions.Num(), TEXT("Dimensions must match the shape of the BoxSpace"));
		this->Shape = Shape;
	}
	for (FBoxSpaceDimension& Dimension : Dimensions)
	{
		this->Dimensions.Add(FBoxSpaceDimension(Dimension));
	}
}

FBoxSpace::FBoxSpace(std::initializer_list<float> Low, std::initializer_list<float> High, std::initializer_list<int> Shape)
{
	auto LowIt = Low.begin();
	auto HighIt = High.begin();
	assert(sizeof(Low) == sizeof(High));
	int TotalShapeSize = GetTotalSize(Shape);
	if (TotalShapeSize == 0)
	{
		this->Shape = TArray<int>({sizeof(Low)});
	}
	else
	{
		verifyf(TotalShapeSize == sizeof(Low), TEXT("Dimensions must match the shape of the BoxSpace"));
		this->Shape = Shape;
	}
	
	while (LowIt != Low.end() && HighIt != High.end())
	{
		this->Dimensions.Add(FBoxSpaceDimension(*LowIt, *HighIt));
		LowIt++;
		HighIt++;
	}
}

FBoxSpace::FBoxSpace(const TArray<int>& Shape)
{
	this->Dimensions.SetNumUninitialized(GetTotalSize(Shape));
	this->Shape = Shape;
}

void FBoxSpace::Copy(const FBoxSpace& Other)
{
	this->Dimensions = TArray<FBoxSpaceDimension>(Other.Dimensions);
	this->Shape = Other.Shape;
}

ESpaceValidationResult FBoxSpace::Validate(TPoint& Observation) const
{
	if (!Observation.IsType<FBoxPoint>())
	{
		return ESpaceValidationResult::WrongDataType;
	}
	FBoxPoint& TypedObservation = Observation.Get<FBoxPoint>();

	// Is the shape right?
	if (Dimensions.Num() != TypedObservation.Values.Num())
	{
		return ESpaceValidationResult::WrongDimensions;
	}
	

	// Are all our values in our bounding box?
	for (int i = 0; i < Dimensions.Num(); i++)
	{
		if (TypedObservation.Values[i] > Dimensions[i].High || TypedObservation.Values[i] < Dimensions[i].Low)
		{
			return ESpaceValidationResult::OutOfBounds;
		}
	}

	return ESpaceValidationResult::Success;
}

FBoxPoint FBoxSpace::NormalizeObservation(const FBoxPoint& Observation) const
{
	FBoxPoint OutObservation = FBoxPoint(this->GetFlattenedSize());
	// Use bounding box to normalize the observations, we can safely do so because it has already been validated
	for (int i = 0; i < Dimensions.Num(); i++)
	{
		OutObservation.Add(Dimensions[i].NormalizeValue(Observation.Values[i]));
	}
	return OutObservation;
}

FBoxSpace FBoxSpace::GetNormalizedObservationSpace() const
{
	FBoxSpace OutBoxSpace;

	// Set extent to be between 0 and 1 for normalized observations.
	for (int i = 0; i < Dimensions.Num(); i++)
	{
		OutBoxSpace.Add(FBoxSpaceDimension::ZeroOneUnitDimension());
	}
	return OutBoxSpace;
}

FBoxSpace::~FBoxSpace()
{
	this->Dimensions.Empty();
	this->Shape.Empty();
}

int FBoxSpace::GetNumDimensions() const
{
	return Dimensions.Num();
}

void FBoxSpace::FillProtobuf(BoxSpace* Msg) const
{
	this->FillProtobuf(*Msg);
}

void FBoxSpace::FillProtobuf(BoxSpace& Msg) const
{
	for (const FBoxSpaceDimension& Dimension : this->Dimensions)
	{
		Dimension.FillProtobuf(Msg.add_dimensions());
	}
	for (int ShapeDim : this->Shape)
	{
		Msg.add_shape_dimensions(ShapeDim);
	}
}

void FBoxSpace::FillProtobuf(FundamentalSpace* Msg) const
{
	this->FillProtobuf(Msg->mutable_box_space());
}


TPoint FBoxSpace::UnflattenAction(const TArray<float>& Data, int Offset) const
{
	TPoint OutPoint = TPoint(TInPlaceType<FBoxPoint>(), (const float*)Data.GetData() + Offset, Dimensions.Num());
	return OutPoint;
}

void FBoxSpace::FlattenPoint(TArrayView<float> Buffer, const TPoint& Point) const
{
	assert(Buffer.Num() == this->GetFlattenedSize());
	TArray<float> Arr = Point.Get<FBoxPoint>().Values;
	for (int i = 0; i < Arr.Num();i++)
	{
		Buffer[i] = Arr[i];
	}
}

int FBoxSpace::GetFlattenedSize() const
{
	return Dimensions.Num();
}

void FBoxSpace::Add(float Low, float High)
{
	this->Dimensions.Add(FBoxSpaceDimension(Low, High));
}

void FBoxSpace::Add(const FBoxSpaceDimension& Dimension)
{
	this->Dimensions.Add(Dimension);
}

bool FBoxSpace::IsEmpty() const
{
	return this->GetNumDimensions() == 0;
}

TPoint FBoxSpace::MakeTPoint() const
{
	return TPoint(TInPlaceType<FBoxPoint>());
}