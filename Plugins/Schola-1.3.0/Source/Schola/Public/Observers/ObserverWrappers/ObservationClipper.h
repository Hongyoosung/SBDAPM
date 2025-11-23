// Copyright (c) 2024 Advanced Micro Devices, Inc. All Rights Reserved.

#pragma once

#include "CoreMinimal.h"
#include "Common/Spaces.h"
#include "Common/Points.h"
#include "Observers/ObserverWrappers/ObserverWrapperInterfaces.h"
#include "ObservationClipper.generated.h"


UCLASS(Blueprintable, EditInlineNew)
class SCHOLA_API UObservationClipper : public UObject, public IBoxObserverWrapper
{
	GENERATED_BODY()

private:
    UPROPERTY()
	FBoxSpace OriginalSpace;

public:

    /**
	 * @brief Cache a copy of the wrapped observation space
	 * @param[in] Space The space being wrapped
	 * @return The Space
	 */
	UFUNCTION()
	virtual FBoxSpace WrapBoxObservationSpace(const FBoxSpace& Space) override;


	UFUNCTION()
	virtual FBoxPoint WrapBoxObservation(const FBoxPoint& Point) override;

	UFUNCTION()
	FString GenerateId() const override;

};