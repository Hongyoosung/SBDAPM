// Copyright (c) 2024 Advanced Micro Devices, Inc. All Rights Reserved.

#pragma once

#include "CoreMinimal.h"
#include "Actuators/ActuatorWrappers/ActuatorWrapperInterfaces.h"
#include "Common/Spaces.h"
#include "Common/Points.h"
#include "ActionClipper.generated.h"


UCLASS(Blueprintable, EditInlineNew)
class SCHOLA_API UActionClipper : public UObject, public IBoxActuatorWrapper
{
	GENERATED_BODY()

private:

	UPROPERTY()
	FBoxSpace OriginalSpace;

public:

	/**
	 * @brief Clips the input action to the specified space
	 * @param[in] Point The point to clip
	 * @return The clipped point
	 */
	UFUNCTION()
	 FBoxPoint WrapBoxAction(const FBoxPoint& Point);

	/**
	 * @brief Just caches a copy of the space for clipping the actions later
	 * @param[in] Space the space being wrapped.
	 * @return The input space.
	 */
	UFUNCTION()
	FBoxSpace WrapBoxActionSpace(const FBoxSpace& Space);
	
	FString GenerateId() const;
};