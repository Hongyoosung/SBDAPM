// Copyright (c) 2023 Advanced Micro Devices, Inc. All Rights Reserved.

#pragma once

#include "CoreMinimal.h"
#include "Components/ActorComponent.h"
#include "InteractionComponent.generated.h"

/**
* @brief An abstract base class for ActorComponents in Schola containing interaction logic with the environment
*/
UCLASS(ClassGroup = Schola, Abstract)
class SCHOLA_API UInteractionComponent : public UActorComponent
{
	GENERATED_BODY()
public:
	
	/* Home for any shared logic between ActuatorComponents and Sensors */
};