// Copyright (c) 2023 Advanced Micro Devices, Inc. All Rights Reserved.

#include "Subsystem/SubsystemSettings/SubsystemSettings.h"



UScholaManagerSubsystemSettings::UScholaManagerSubsystemSettings()
{
	
}

FLaunchableScript UScholaManagerSubsystemSettings::GetScript() const
{
	FLaunchableScript Script = this->ScriptSettings.GetLaunchableScript();
	FString			  TrainingArgs = this->ScriptSettings.GetTrainingArgs(this->CommunicatorSettings.Port);
	Script.AppendArgs(TrainingArgs);
	return Script;
}
