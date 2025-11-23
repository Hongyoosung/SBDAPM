// Copyright (c) 2023 Advanced Micro Devices, Inc. All Rights Reserved.

#pragma once

#include "CoreMinimal.h"
#include "Observers/AbstractObservers.h"
#include "EventObserver.generated.h"

/**
 * @brief A Binary observer that can be triggered by an event.
 */
UCLASS()
class SCHOLA_API UEventObserver : public UBinaryObserver
{
	GENERATED_BODY()
public:

	/** Was an event triggered during this step */
	UPROPERTY(VisibleAnywhere, Category = "Sensor Properties")
	bool bEventTriggered = false;

	/** Should the event flag be cleared automatically after each step */
	UPROPERTY(EditAnywhere, Category = "Sensor Settings")
	bool bAutoClearEventFlag = true;

	/**
	 * @brief Trigger the event.
	 */
	UFUNCTION(BlueprintCallable, Category = "Sensor")
	void TriggerEvent();

	/**
	 * @brief Clear the event.
	 */
	UFUNCTION(BlueprintCallable, Category = "Sensor")
	void ClearEvent();

	/**
	 * @brief Get the observation space of this sensor.
	 * @return A BinarySpace of size 1
	 */
	FBinarySpace GetObservationSpace() const;

	/**
	 * @brief Collect observations about the environment state. Returns 1 if Event was Triggered. 0 Otherwise.
	 * @param[out] OutObservations A BinaryPoint that will be updated with the outputs of this sensor.
	 */
	virtual void CollectObservations(FBinaryPoint& OutObservations);

	virtual FString GenerateId() const override;
};
