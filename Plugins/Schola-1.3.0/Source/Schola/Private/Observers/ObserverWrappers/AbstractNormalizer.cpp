// Copyright (c) 2024 Advanced Micro Devices, Inc. All Rights Reserved.

#include "Observers/ObserverWrappers/AbstractNormalizer.h"

FBoxSpace UAbstractNormalizer::WrapBoxObservationSpace(const FBoxSpace& Space)
{
	this->OriginalSpace = Space;
	return Space.GetNormalizedObservationSpace();
}

FString UAbstractNormalizer::GenerateId() const
{
	return FString("Norm");
}
