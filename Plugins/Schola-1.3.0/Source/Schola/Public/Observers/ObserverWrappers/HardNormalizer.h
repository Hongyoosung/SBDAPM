// Copyright (c) 2024 Advanced Micro Devices, Inc. All Rights Reserved.

#pragma once

#include "CoreMinimal.h"
#include "Common/Spaces.h"
#include "Common/Points.h"
#include "Observers/ObserverWrappers/AbstractNormalizer.h"
#include "HardNormalizer.generated.h"


UCLASS(Blueprintable, EditInlineNew)
class SCHOLA_API UHardNormalizer : public UAbstractNormalizer
{
	GENERATED_BODY()

protected:
	
	virtual FString GenerateId() const override;

public:

	virtual FBoxPoint WrapBoxObservation(const FBoxPoint& Point) override;

};