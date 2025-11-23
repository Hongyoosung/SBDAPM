// Copyright (c) 2023 Advanced Micro Devices, Inc. All Rights Reserved.

#pragma once

#include "Actuators/AbstractActuators.h"
#include "Common/LogSchola.h"
#include "CoreMinimal.h"
#include "TeleportActuator.generated.h"

UENUM()
enum class ETeleportDirection : int8
{
	Nothing = 0,
	Forward = 1,
	Backward = 2,
};

DECLARE_DYNAMIC_MULTICAST_DELEGATE_OneParam(FOnTeleportSignature, const FVector&, TeleportOffset);

UCLASS(BlueprintType, Blueprintable)
class SCHOLA_API UTeleportActuator : public UDiscreteActuator
{
	GENERATED_BODY()
public:
	/** The distance this agent teleports in the X dimension */
	UPROPERTY(EditAnywhere, meta = (EditCondition = "bHasXDimension"), Category = "Actuator Settings")
	float XDimensionSpeed;

	/** The distance this agent teleports in the Y dimension */
	UPROPERTY(EditAnywhere, meta = (EditCondition = "bHasYDimension"), Category = "Actuator Settings")
	float YDimensionSpeed;

	/** The distance this agent teleports in the Z dimension */
	UPROPERTY(EditAnywhere, meta = (EditCondition = "bHasZDimension"), Category = "Actuator Settings")
	float ZDimensionSpeed;

	/** Toggle for whether this actuator teleports the Agent along the X dimension */
	UPROPERTY(EditAnywhere, BlueprintReadOnly, Meta = (InlineEditConditionToggle), Category = "Actuator Settings")
	bool bHasXDimension = true;

	/** Toggle for whether this actuator teleports the Agent along the Z dimension */
	UPROPERTY(EditAnywhere, BlueprintReadOnly, Meta = (InlineEditConditionToggle), Category = "Actuator Settings")
	bool bHasZDimension = true;

	/** Toggle for whether this actuator teleports the Agent along the Y dimension */
	UPROPERTY(EditAnywhere, BlueprintReadOnly, Meta = (InlineEditConditionToggle), Category = "Actuator Settings")
	bool bHasYDimension = true;

	/** A delegate invoked when this actuator receives input from a brain */
	UPROPERTY(BlueprintAssignable)
	FOnTeleportSignature OnTeleportDelegate;

	/** Type of teleportation to use. See SetActorLocation documentation for more details. */
	UPROPERTY(EditAnywhere, Category = "Actuator Settings")
	ETeleportType TeleportType = ETeleportType::None;

	/** Toggle for whether to sweep while teleporting the actor. See SetActorLocation documentation for more details*/
	UPROPERTY(EditAnywhere, Category = "Actuator Settings")
	bool bSweep;

	FDiscreteSpace GetActionSpace() override;

	/**
	* @brief Convert a Discrete Point with 3 values to an FVector
	* @param[in] Action DiscretePoint that will be converted
	* @return FVector containing the converted DiscretePoint 
	*/
	FVector ConvertActionToFVector(const FDiscretePoint& Action);

	/**
	* @brief Get the magnitude of movement based on the value of a singular 
	* @param[in] Speed The speed of the teleportaiton Actuator
	* @param[in] DiscretePointValue The value of the DiscretePoint, to check for forward, backward, or no movement
	* @return The magnitude of the movement, multiplied by the direction
	*/
	float GetVectorDimension(int Speed, int DiscretePointValue);

	void TakeAction(const FDiscretePoint& Action) override;

	FString GenerateId() const override;
};
