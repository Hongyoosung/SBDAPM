// Copyright (c) 2024 Advanced Micro Devices, Inc. All Rights Reserved.

#include "Actuators/ActuatorWrappers/ActuatorWrapperInterfaces.h"

FBoxSpace IBoxActuatorWrapper::WrapBoxActionSpace(const FBoxSpace& Space)
{
	return Space;
}


FDiscreteSpace IDiscreteActuatorWrapper::WrapDiscreteActionSpace(const FDiscreteSpace& Space)
{
	return Space;
}

FBinarySpace IBinaryActuatorWrapper::WrapBinaryActionSpace(const FBinarySpace& Space)
{
	return Space;
}
