// Copyright (c) 2024-2025 Advanced Micro Devices, Inc. All Rights Reserved.

#pragma once
#include "Subsystem/SubsystemSettings/StableBaselines/SB3TrainingSettings.h"

void FSB3TrainingSettings::GenerateTrainingArgs(int Port, FScriptArgBuilder& ArgBuilder) const
{
	this->CheckpointSettings.GenerateTrainingArgs(Port, ArgBuilder);
	this->LoggingSettings.GenerateTrainingArgs(Port, ArgBuilder);
	this->ResumeSettings.GenerateTrainingArgs(Port, ArgBuilder);
	this->NetworkArchitectureSettings.GenerateTrainingArgs(Port, ArgBuilder);
	//Note that if the NetworkArch Args go right before the Algorithm, the Algorithm gets eaten by the Variable length argument defining network arch.

	ArgBuilder.AddIntArg(TEXT("port"), Port);
	ArgBuilder.AddIntArg(TEXT("timesteps"), Timesteps);
	ArgBuilder.AddFlag(TEXT("pbar"), bDisplayProgressBar);

	switch (this->Algorithm)
	{
		case (ESB3TrainingAlgorithm::SAC):
			ArgBuilder.AddPositionalArgument(TEXT("SAC"));
			this->SACSettings.GenerateTrainingArgs(Port, ArgBuilder);
			break;
		default:
			ArgBuilder.AddPositionalArgument(TEXT("PPO"));
			this->PPOSettings.GenerateTrainingArgs(Port, ArgBuilder);
			break;
	}
}

FSB3TrainingSettings::~FSB3TrainingSettings()
{
}