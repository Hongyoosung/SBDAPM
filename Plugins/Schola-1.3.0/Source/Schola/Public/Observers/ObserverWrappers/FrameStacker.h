// Copyright (c) 2024 Advanced Micro Devices, Inc. All Rights Reserved.

#pragma once

#include "CoreMinimal.h"
#include "Observers/ObserverWrappers/ObserverWrapperInterfaces.h"
#include "Common/Spaces.h"
#include "Common/Points.h"
#include "FrameStacker.generated.h"

//TODO figure out reset logic
UCLASS(Blueprintable, EditInlineNew)
class SCHOLA_API UFrameStacker : public UObject, public IBoxObserverWrapper
{
	GENERATED_BODY()

public:
	/** Set the number of frames to stack */	
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Wrapper Properties")
	int MemorySize=4;

	/** The size of the space that is wrapped, set in WrapBoxObservationSpace */
	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "Wrapper Properties")
	int IndividualSpaceSize = 0;

	/** The buffer that stores the frames */
	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "Wrapper Properties")
	TArray<float> FrameBuffer;

	/** The default value to fill the buffer with */		
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Wrapper Properties")
	float FillValue = 0;

	/**
	 * @brief Wraps the input observation space to include multiple stacked frames
	 * @param[in] Space The space to stack	
	 * @return The stacked space
	 */
	UFUNCTION()
	virtual FBoxSpace WrapBoxObservationSpace(const FBoxSpace& Space) override;
	
	/**
	 * @brief Wraps the input observation to include multiple previous frames.
	 * @param[in] Point The point to stack
	 * @return The stacked point
	 */
	UFUNCTION()
	virtual FBoxPoint WrapBoxObservation(const FBoxPoint& Point);

	/**
	 * @brief Resets the observer to the fill value
	*/
	UFUNCTION()
	void Reset() override;

	UFUNCTION()
	int GetBufferSize();

	UFUNCTION()
	FString GenerateId() const override;
};