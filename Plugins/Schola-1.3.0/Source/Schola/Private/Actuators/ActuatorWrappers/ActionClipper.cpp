// Copyright (c) 2024 Advanced Micro Devices, Inc. All Rights Reserved.

#include "Actuators/ActuatorWrappers/ActionClipper.h"

FBoxSpace UActionClipper::WrapBoxActionSpace(const FBoxSpace& Space)
{
    this->OriginalSpace = Space;
    return Space;
}

FString UActionClipper::GenerateId() const
{
	return FString("Clip");
}


FBoxPoint UActionClipper::WrapBoxAction(const FBoxPoint& Point)
{
	int		  NumDims = Point.Values.Num();
	FBoxPoint OutputPoint = FBoxPoint(NumDims);
  for(int i=0;i < NumDims;i++)
  {
      OutputPoint.Add(FMath::Clamp(Point[i], this->OriginalSpace.Dimensions[i].Low, this->OriginalSpace.Dimensions[i].High));
  }
  return OutputPoint;
}