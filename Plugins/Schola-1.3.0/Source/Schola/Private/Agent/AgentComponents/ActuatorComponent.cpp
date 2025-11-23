// Copyright (c) 2023 Advanced Micro Devices, Inc. All Rights Reserved.

#include "Agent/AgentComponents/ActuatorComponent.h"

// Sets default values for this component's properties
UActuatorComponent::UActuatorComponent()
{
	// Set this component to be initialized when the game starts, and to be ticked every frame.  You can turn these features
	// off to improve performance if you don't need them.
	PrimaryComponentTick.bCanEverTick = true;

	// ...
}

// Called when the game starts
void UActuatorComponent::BeginPlay()
{
	Super::BeginPlay();

	// ...
}

// Called every frame
void UActuatorComponent::TickComponent(float DeltaTime, ELevelTick TickType, FActorComponentTickFunction* ThisTickFunction)
{
	Super::TickComponent(DeltaTime, TickType, ThisTickFunction);

	// ...
}
