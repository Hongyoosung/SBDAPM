// Copyright (c) 2024 Advanced Micro Devices, Inc. All Rights Reserved.

#include "Observers/DebugObservers.h"

FBoxSpace UDebugBoxObserver::GetObservationSpace() const
{
	return this->ObservationSpace;
}

void UDebugBoxObserver::CollectObservations(FBoxPoint& OutObservations)
{
	for (const FBoxSpaceDimension& BoxSpaceDim : this->ObservationSpace.Dimensions)
	{
		OutObservations.Add(FMath::FRandRange(BoxSpaceDim.Low,BoxSpaceDim.High));
	}
}

FString UDebugBoxObserver::GenerateId() const
{
	FString Output = FString("DebugBox");
	for (const FBoxSpaceDimension& BoxSpaceDim : this->ObservationSpace.Dimensions)
	{
		Output.Appendf(TEXT("_%f_%f"), BoxSpaceDim.Low, BoxSpaceDim.High);
	}
	return Output;
}

FBinarySpace UDebugBinaryObserver::GetObservationSpace() const
{
	return this->ObservationSpace;
}

void UDebugBinaryObserver::CollectObservations(FBinaryPoint& OutObservations)
{
	for (int i = 0; i < this->ObservationSpace.Shape; i++)
	{
		OutObservations.Add(FMath::RandBool());
	}
}

FString UDebugBinaryObserver::GenerateId() const
{
	FString Output = FString("DebugBinary");
	Output.Appendf(TEXT("_%d"), this->ObservationSpace.Shape);
	return Output;
}

FDiscreteSpace UDebugDiscreteObserver::GetObservationSpace() const
{
	return this->ObservationSpace;
}

void UDebugDiscreteObserver::CollectObservations(FDiscretePoint& OutObservations)
{
	for (auto& DimUpperBound : this->ObservationSpace.High)
	{
		OutObservations.Add(FMath::RandHelper(DimUpperBound));
	}
}

FString UDebugDiscreteObserver::GenerateId() const
{
	FString Output = FString("DebugDiscrete");
	for (auto& DimUpperBound : this->ObservationSpace.High)
	{
		Output.Appendf(TEXT("_%d"), DimUpperBound);
	}
	return Output;
}
