// Copyright (c) 2024 Advanced Micro Devices, Inc. All Rights Reserved.

#include "Actuators/DebugActuators.h"

FBoxSpace UDebugBoxActuator::GetActionSpace()
{
	return this->ActionSpace;
}

FDiscreteSpace UDebugDiscreteActuator::GetActionSpace()
{
	return this->ActionSpace;
}

FBinarySpace UDebugBinaryActuator::GetActionSpace()
{
	return this->ActionSpace;
}
