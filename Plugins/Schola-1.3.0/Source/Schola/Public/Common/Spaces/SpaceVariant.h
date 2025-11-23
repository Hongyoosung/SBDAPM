// Copyright (c) 2023-2024 Advanced Micro Devices, Inc. All Rights Reserved.

#pragma once

#include "Common/Spaces/BinarySpace.h"
#include "Common/Spaces/BoxSpace.h"
#include "Common/Spaces/DiscreteSpace.h"

/** A variant over all Concrete Space types */
typedef TVariant<FBoxSpace, FDiscreteSpace, FBinarySpace> TSpace;

/**
 * @brief A struct representing the type of a space (e.g. Box, Discrete, Binary)
 */
UENUM()
enum class ESpaceType
{
	Box,
	Discrete,
	Binary
};
