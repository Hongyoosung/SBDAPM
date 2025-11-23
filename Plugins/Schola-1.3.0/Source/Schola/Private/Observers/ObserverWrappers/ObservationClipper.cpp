// Copyright (c) 2024 Advanced Micro Devices, Inc. All Rights Reserved.

#include "Observers/ObserverWrappers/ObservationClipper.h"

FBoxSpace UObservationClipper::WrapBoxObservationSpace(const FBoxSpace& Space)
{
    this->OriginalSpace = Space;
	return Space;
}

FBoxPoint UObservationClipper::WrapBoxObservation(const FBoxPoint& Point)
{
	int		  NumDims = Point.Values.Num();
	FBoxPoint OutputPoint = FBoxPoint(NumDims);
    for(int i = 0; i < NumDims; i++)
    {
        OutputPoint.Add(FMath::Clamp(Point[i], this->OriginalSpace.Dimensions[i].Low, this->OriginalSpace.Dimensions[i].High));
    }
    
    return OutputPoint;
}

FString UObservationClipper::GenerateId() const
{
	return FString("Clip");
}
