// Copyright (c) 2023 Advanced Micro Devices, Inc. All Rights Reserved.

#pragma once

#include "CoreMinimal.h"
#include "Components/ActorComponent.h"
#include "Containers/Array.h"
#include "Common/Points.h"
#include "Common/Spaces.h"
#include "Observers/AbstractObservers.h"
#include "Agent/AgentComponents/InteractionComponent.h"
#include "SensorComponent.generated.h"

/**
* @brief An ActorComponent for holding onto an Observer and providing some related utility functions
*/
UCLASS(ClassGroup = Schola, meta = (BlueprintSpawnableComponent))
class SCHOLA_API USensor : public UInteractionComponent
{
	GENERATED_BODY()

public:
	/** The Observer Object inside this Actuator Component */
	UPROPERTY(EditAnywhere, NoClear, Instanced, meta = (ShowInnerProperties), Category = "Sensor")
	UAbstractObserver* Observer;

#if WITH_EDITOR
	/** Test the sensor validity by checking, if the number of dimensions is correct and if values are out of bounds. */
	UFUNCTION(CallInEditor, Category="Sensor", Meta = (Tooltip = "Test the sensor validity by checking if the number of dimensions is correct and if values are out of bounds."))
	void TestSensorValidity();

	/** Generate a DebugId for this Observer, to check what will be used for Naming */
	UFUNCTION(CallInEditor, Category = "Sensor", Meta = (Tooltip = "Display the Id for the Observer contained in this sensor."))
	void GenerateDebugId()
	{
		if (Observer)
		{
			Observer->DisplayId = Observer->GetId();
		}
	}
	
#endif

};