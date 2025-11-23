// Copyright (c) 2024-2025 Advanced Micro Devices, Inc. All Rights Reserved.

#pragma once
#include "Subsystem/SubsystemSettings/Ray/RLlibTrainingSettings.h"

void FRLlibTrainingSettings::GenerateTrainingArgs(int Port, FScriptArgBuilder& ArgBuilder) const
{
	ArgBuilder.AddIntArg(TEXT("port"), Port);
	ArgBuilder.AddIntArg(TEXT("timesteps"), this->Timesteps);
	ArgBuilder.AddFloatArg(TEXT("learning-rate"), this->LearningRate);
	ArgBuilder.AddIntArg(TEXT("minibatch-size"), this->MinibatchSize);
	ArgBuilder.AddIntArg(TEXT("train-batch-size-per-learner"), this->TrainBatchSizePerLearner);
	ArgBuilder.AddIntArg(TEXT("num-sgd-iter"), this->NumSGDIter);
	ArgBuilder.AddFloatArg(TEXT("gamma"), this->Gamma);

	this->CheckpointSettings.GenerateTrainingArgs(Port, ArgBuilder);
	this->LoggingSettings.GenerateTrainingArgs(Port, ArgBuilder);
	this->ResumeSettings.GenerateTrainingArgs(Port, ArgBuilder);
	this->NetworkArchitectureSettings.GenerateTrainingArgs(Port, ArgBuilder);
	this->ResourceSettings.GenerateTrainingArgs(Port, ArgBuilder);

	switch (this->Algorithm)
	{
		case (ERLlibTrainingAlgorithm::PPO):
			ArgBuilder.AddPositionalArgument(TEXT("PPO"));
			this->PPOSettings.GenerateTrainingArgs(Port, ArgBuilder);
			break;
		case (ERLlibTrainingAlgorithm::IMPALA):
			ArgBuilder.AddPositionalArgument(TEXT("IMPALA"));
			this->IMPALASettings.GenerateTrainingArgs(Port, ArgBuilder);
			break;
		default:
			ArgBuilder.AddPositionalArgument(TEXT("APPO"));
			this->APPOSettings.GenerateTrainingArgs(Port, ArgBuilder);
			break;
	}
}

FRLlibTrainingSettings::~FRLlibTrainingSettings()
{
}