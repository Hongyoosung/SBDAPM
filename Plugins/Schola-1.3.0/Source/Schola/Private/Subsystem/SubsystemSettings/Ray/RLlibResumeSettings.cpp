// Copyright (c) 2024-2025 Advanced Micro Devices, Inc. All Rights Reserved.

#pragma once
#include "Subsystem/SubsystemSettings/Ray/RLlibResumeSettings.h"

void FRLlibResumeSettings::GenerateTrainingArgs(int Port, FScriptArgBuilder& ArgBuilder) const
{
	ArgBuilder.AddConditionalStringArg(TEXT("resume-from"), this->ModelPath.FilePath, bLoadModel);
}

FRLlibResumeSettings::~FRLlibResumeSettings()
{
}