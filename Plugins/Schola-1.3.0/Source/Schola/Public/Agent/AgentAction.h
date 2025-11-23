// Copyright (c) 2023 Advanced Micro Devices, Inc. All Rights Reserved.

#pragma once

#include "CoreMinimal.h"
#include "Common/Points.h"
#include "AgentAction.generated.h"

/**
 * @brief Struct representing an action taken by an agent. Currently just a wrapper around a DictPoint but designed to leave room for expansion.
 */
USTRUCT()
struct SCHOLA_API FAction
{
	GENERATED_BODY()
	/** The Body of the action */
	FDictPoint Values;

	FAction(){};

};

