// Copyright (c) 2023-2025 Advanced Micro Devices, Inc. All Rights Reserved.

#include "Common/Spaces/DictSpace.h"

FDictSpace::FDictSpace()
{
}


int FDictSpace::Num()
{
	return 0;
}


int FDictSpace::GetFlattenedSize() const
{	
	int Size = 0;
	for (const TSpace& Space : this->Spaces)
	{
		Size += Visit([](auto _Space) { return _Space.GetFlattenedSize(); }, Space);
	}

	return Size;
}

ESpaceValidationResult FDictSpace::Validate(FDictPoint& PointMap) const
{
	ESpaceValidationResult Result = ESpaceValidationResult::NoResults;

	for (int i =0; i < this->Spaces.Num(); i++)
	{
		const TSpace& Space = this->Spaces[i];
		TPoint& Point = PointMap[i];
		ESpaceValidationResult CurrentResult = Visit([&Point](auto& TypedSpace) { return TypedSpace.Validate(Point); }, Space);
		switch (CurrentResult)
		{
			case ESpaceValidationResult::NoResults:
				break;
			case ESpaceValidationResult::Success:
				Result = CurrentResult;
				break;
			default:
				Result = CurrentResult;
				return Result;
		}

	}

	return Result;
}

void FDictSpace::Reset()
{
	this->Labels.Empty();
	this->Spaces.Empty();
}

DictSpace* FDictSpace::ToProtobuf() const
{
	DictSpace* RetVal = new DictSpace();
	FillProtobuf(RetVal);
	return RetVal;
}

void FDictSpace::FillProtobuf(DictSpace* Msg) const
{
	for (int i =0; i < this->Spaces.Num();i++)
	{
		Msg->add_labels(TCHAR_TO_UTF8(*this->Labels[i]));
		FundamentalSpace* SpaceMsg = Msg->add_values();
		Visit([SpaceMsg](auto& TypedSpace) { TypedSpace.FillProtobuf(SpaceMsg); }, this->Spaces[i]);
	}
}

void FDictSpace::InitializeEmptyDictPoint(FDictPoint& EmptyPoint)
{	
	EmptyPoint.Points.Empty();
	for (TSpace& Space : this->Spaces)
	{
		EmptyPoint.Points.Add((Visit([](auto& TypedSpace) { return TypedSpace.MakeTPoint(); }, Space)));
	}
}

FDictPoint FDictSpace::UnflattenPoint(TArray<float>& FlattenedPoint)
{
	int StartIndex = 0;
	FDictPoint Output = FDictPoint();
	for (const TSpace& Space : this->Spaces)
	{
		int Size = Visit([](auto& TypedSpace) { return TypedSpace.GetFlattenedSize(); }, Space);
		TPoint Point = Visit([&FlattenedPoint, StartIndex](auto& TypedSpace) { return TypedSpace.UnflattenAction(FlattenedPoint, StartIndex); }, Space);
		Output.Points.Add(Point);
		StartIndex += Size;
	}
	return Output;
}


TSpace& FDictSpace::Add(const FString& Label)
{
	this->Labels.Add(Label);
	return this->Spaces.Emplace_GetRef();
}