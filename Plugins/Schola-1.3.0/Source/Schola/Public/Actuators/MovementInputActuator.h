// Copyright (c) 2023 Advanced Micro Devices, Inc. All Rights Reserved.

#pragma once

#include "Actuators/AbstractActuators.h"
#include "Common/Points.h"
#include "Common/Spaces.h"
#include "CoreMinimal.h"
#include "Common/LogSchola.h"
#include "Kismet/KismetMathLibrary.h"
#include "Common/PositionalEnums.h"
#include "MovementInputActuator.generated.h"

DECLARE_DYNAMIC_MULTICAST_DELEGATE_OneParam(FOnMovementInputSignature, const FVector&, MovementInput);

UCLASS(BlueprintType, Blueprintable)
class UMovementInputActuator : public UBoxActuator
{
	GENERATED_BODY()

public:
	/** Toggle for whether this actuator moves the Agent along the X dimension */
	UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "Actuator Settings")
	bool bHasXDimension = true;

	/** Toggle for whether this actuator moves the Agent along the Z dimension */
	UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "Actuator Settings")
	bool bHasZDimension = true;

	/** Toggle for whether this actuator moves the Agent along the Y dimension */
	UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "Actuator Settings")
	bool bHasYDimension = true;

	/** The minimum speed at which the agent can move. Set to negative to allow for backwards movement along each axis. */
	UPROPERTY(EditAnywhere, Category = "Actuator Settings")
	float MinSpeed = 0.0;

	/** The maximum speed at which the agent can move */
	UPROPERTY(EditAnywhere, Category = "Actuator Settings")
	float MaxSpeed = 1.0;

	/** A delegate invoked when this actuator receives input from a brain. Useful for debugging and logging */
	UPROPERTY(BlueprintAssignable)
	FOnMovementInputSignature OnMovementDelegate;

	/** Scale to apply input. See AddMovementInput for more details.*/
	UPROPERTY(EditAnywhere, Category = "Actuator Settings")
	float ScaleValue = 1;

	/** Force the Pawn to Move. See AddMovementInput for more details.*/
	UPROPERTY(EditAnywhere, Category = "Actuator Settings")
	bool bForce = false;

	FBoxSpace GetActionSpace() override;

	/**
	* @brief Convert a Box Point with 3 values to an FVector
	* @param[in] Action BoxPoint that will be converted 
	* @return FVector containing the converted BoxPoint
	*/
	FVector ConvertActionToFVector(const FBoxPoint& Action);

	void TakeAction(const FBoxPoint& Action) override;

	FString GenerateId() const override;
};