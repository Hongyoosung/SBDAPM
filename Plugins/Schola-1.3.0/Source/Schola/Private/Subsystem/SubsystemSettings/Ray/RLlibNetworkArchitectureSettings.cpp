// Copyright (c) 2024-2025 Advanced Micro Devices, Inc. All Rights Reserved.

#pragma once
#include "Subsystem/SubsystemSettings/Ray/RLlibNetworkArchitectureSettings.h"

void FRLlibNetworkArchSettings::GenerateTrainingArgs(int Port, FScriptArgBuilder& ArgBuilder) const
{
	FString ActivationString;
	switch (ActivationFunction)
	{
		case (ERLlibActivationFunctionEnum::TanH):
			ActivationString = TEXT("tanh");
			break;

		case (ERLlibActivationFunctionEnum::ReLU):
			ActivationString = TEXT("relu");
			break;

		default:
			ActivationString = TEXT("sigmoid");
			break;
	}
	ArgBuilder.AddStringArg(TEXT("activation"),ActivationString);
	ArgBuilder.AddIntArrayArg(TEXT("fcnet-hiddens"), this->FCNetHiddens);
	if(this->bUseAttention){
		ArgBuilder.AddFlag(TEXT("use-attention"));
		ArgBuilder.AddIntArg(TEXT("attention-dims"),this->AttentionDims);
	}
}

FRLlibNetworkArchSettings::~FRLlibNetworkArchSettings()
{
}