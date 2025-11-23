// Copyright (c) 2024-2025 Advanced Micro Devices, Inc. All Rights Reserved.

#pragma once
#include "Subsystem/SubsystemSettings/Ray/RLlibLoggingSettings.h"

void FRLlibLoggingSettings::GenerateTrainingArgs(int Port, FScriptArgBuilder& ArgBuilder) const
{
	ArgBuilder.AddIntArg(TEXT("schola-verbosity"), EnvLoggingVerbosity);
	ArgBuilder.AddIntArg(TEXT("rllib-verbosity"), TrainerLoggingVerbosity);
}

FRLlibLoggingSettings::~FRLlibLoggingSettings()
{
    
}