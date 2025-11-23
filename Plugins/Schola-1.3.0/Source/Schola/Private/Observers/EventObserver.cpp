// Copyright (c) 2023 Advanced Micro Devices, Inc. All Rights Reserved.

#include "Observers/EventObserver.h"

void UEventObserver::TriggerEvent()
{
	bEventTriggered = true;
}

void UEventObserver::ClearEvent()
{
	bEventTriggered = false;
}

FBinarySpace UEventObserver::GetObservationSpace() const
{
	return FBinarySpace(1);
}

void UEventObserver::CollectObservations(FBinaryPoint& OutObservations)
{
	OutObservations.Add(bEventTriggered);
	if (bAutoClearEventFlag)
	{
		bEventTriggered = false;
	}
}

FString UEventObserver::GenerateId() const
{
	FString Output = FString("Event");
	Output.Append(bAutoClearEventFlag ? "_AutoClear" : "_NoAutoClear");
	return Output;
}
