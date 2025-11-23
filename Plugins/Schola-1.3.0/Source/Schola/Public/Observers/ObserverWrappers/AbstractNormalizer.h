// Copyright (c) 2024 Advanced Micro Devices, Inc. All Rights Reserved.

#pragma once

#include "CoreMinimal.h"
#include "Observers/ObserverWrappers/ObserverWrapperInterfaces.h"
#include "Common/Spaces.h"
#include "Common/Points.h"
#include "AbstractNormalizer.generated.h"


UCLASS(Abstract, Blueprintable, EditInlineNew)
class SCHOLA_API UAbstractNormalizer : public UObject, public IBoxObserverWrapper
{
	GENERATED_BODY()

protected:

	UPROPERTY(BlueprintReadOnly, VisibleAnywhere, Category = "Wrapper Properties")
	FBoxSpace OriginalSpace;

public:

	UFUNCTION()
    virtual FBoxSpace WrapBoxObservationSpace(const FBoxSpace& Space) override;

	UFUNCTION()
	virtual FBoxPoint WrapBoxObservation(const FBoxPoint& Point) PURE_VIRTUAL(UAbstractNormalizer::WrapBoxObservation, return FBoxPoint(););

	UFUNCTION()
	virtual FString GenerateId() const;

};