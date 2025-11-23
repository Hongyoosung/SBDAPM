// Copyright (c) 2023 Advanced Micro Devices, Inc. All Rights Reserved.

#include "Agent/AgentComponents/SensorComponent.h"

#if WITH_EDITOR
void USensor::TestSensorValidity()
{
	if (this->Observer)
	{
		this->Observer->TestObserverValidity();
	}
	
}
#endif
