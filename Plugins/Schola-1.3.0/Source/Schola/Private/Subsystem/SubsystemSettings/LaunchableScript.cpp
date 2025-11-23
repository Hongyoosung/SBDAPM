// Copyright (c) 2024-2025 Advanced Micro Devices, Inc. All Rights Reserved.

#pragma once
#include "Subsystem/SubsystemSettings/LaunchableScript.h"


FLaunchableScript::FLaunchableScript()
{
	this->ScriptURL = FString("");
	this->Args = FString("");
}

FLaunchableScript::FLaunchableScript(FString ScriptURL)
{
	this->ScriptURL = ScriptURL;
	this->Args = FString("");
}

FLaunchableScript::FLaunchableScript(FString ScriptURL, FString Args)
{
	this->ScriptURL = ScriptURL;
	this->Args = Args;
}

void FLaunchableScript::AppendArgs(FString& AdditionalArgs)
{
	this->Args += (FString(" ") + AdditionalArgs);
}

FProcHandle FLaunchableScript::LaunchScript() const
{
	return FPlatformProcess::CreateProc(*this->ScriptURL, *this->Args, false, false, false, nullptr, 0, nullptr, nullptr);
}
