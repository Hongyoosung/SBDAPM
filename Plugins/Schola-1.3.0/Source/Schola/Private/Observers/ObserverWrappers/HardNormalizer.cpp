// Copyright (c) 2024 Advanced Micro Devices, Inc. All Rights Reserved.

#include "Observers/ObserverWrappers/HardNormalizer.h"

FBoxPoint UHardNormalizer::WrapBoxObservation(const FBoxPoint& Point)
{
	return this->OriginalSpace.NormalizeObservation(Point);
}

FString UHardNormalizer::GenerateId() const
{
	return FString("HardNorm");
}
