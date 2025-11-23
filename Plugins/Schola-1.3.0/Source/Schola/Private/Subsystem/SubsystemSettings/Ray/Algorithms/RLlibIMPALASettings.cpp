// Copyright (c) 2025 Advanced Micro Devices, Inc. All Rights Reserved.

#include "Subsystem/SubsystemSettings/Ray/Algorithms/RLlibIMPALASettings.h"

void FRLlibIMPALASettings::GenerateTrainingArgs(int Port, FScriptArgBuilder& ArgBuilder) const
{
    ArgBuilder.AddFlag(TEXT("disable-vtrace"), !this->bVTrace);
    ArgBuilder.AddFloatArg(TEXT("vtrace-clip-rho-threshold"), this->VTraceClipRhoThreshold);
    ArgBuilder.AddFloatArg(TEXT("vtrace-clip-pg-rho-threshold"), this->VTraceClipPGRhoThreshold);
}

FRLlibIMPALASettings::~FRLlibIMPALASettings()
{
}