// Copyright (c) 2024 Advanced Micro Devices, Inc. All Rights Reserved.

#include "Observers/ObserverWrappers/ObserverWrapperInterfaces.h"

FBoxSpace IBoxObserverWrapper::WrapBoxObservationSpace(const FBoxSpace& Space)
{
	return Space;
}


FDiscreteSpace IDiscreteObserverWrapper::WrapDiscreteObservationSpace(const FDiscreteSpace& Space)
{
	return Space;
}

FBinarySpace IBinaryObserverWrapper::WrapBinaryObservationSpace(const FBinarySpace& Space)
{
	return Space;
}
