// Copyright (c) 2023 Advanced Micro Devices, Inc. All Rights Reserved.

#pragma once
#include "Actuators/AbstractActuators.h"
#include "Common/Points.h"
#include "Common/Spaces.h"
#include "CoreMinimal.h"
#include "Common/LogSchola.h"
#include "Common/PositionalEnums.h"
#include "RotationActuator.generated.h"

DECLARE_DYNAMIC_MULTICAST_DELEGATE_OneParam(FOnRotationInputSignature, const FRotator&, RotationInput);

UCLASS(BlueprintType, Blueprintable)
class URotationActuator : public UBoxActuator
{
	GENERATED_BODY()

public:

	/** The Min/Max value for the Pitch of the tracked rotation */
	UPROPERTY(EditAnywhere, meta = (EditCondition = "bHasPitch"), Category = "Actuator Settings")
	FBoxSpaceDimension PitchBounds = FBoxSpaceDimension(-180, 180);

	/** The Min/Max value for the Roll of the tracked rotation */
	UPROPERTY(EditAnywhere, meta = (EditCondition = "bHasRoll"), Category = "Actuator Settings")
	FBoxSpaceDimension RollBounds = FBoxSpaceDimension(-180, 180);

	/** The Min/Max value for the Yaw of the tracked rotation */
	UPROPERTY(EditAnywhere, meta = (EditCondition = "bHasYaw"), Category = "Actuator Settings")
	FBoxSpaceDimension YawBounds = FBoxSpaceDimension(-180, 180);

	/** Toggle for whether this actuator rotates the Agent along the Pitch dimension */
	UPROPERTY(EditAnywhere, BlueprintReadOnly, Meta = (InlineEditConditionToggle), Category = "Actuator Settings")
	bool bHasPitch = true;

	/** Toggle for whether this actuator rotates the Agent along the Roll dimension */
	UPROPERTY(EditAnywhere, BlueprintReadOnly, Meta = (InlineEditConditionToggle), Category = "Actuator Settings")
	bool bHasRoll = true;

	/** Toggle for whether this actuator rotates the Agent along the Yaw dimension */
	UPROPERTY(EditAnywhere, BlueprintReadOnly, Meta = (InlineEditConditionToggle), Category = "Actuator Settings")
	bool bHasYaw = true;

	/** Type of teleportation to use. See SetActorLocation documentation for more details. */
	UPROPERTY(EditAnywhere, Category = "Actuator Settings")
	ETeleportType TeleportType = ETeleportType::None;

	/** Toggle for whether to sweep while teleporting the actor. See SetActorLocation documentation for more details*/
	UPROPERTY(EditAnywhere, Category = "Actuator Settings")
	bool bSweep;

	/** Toggle for whether to use a [0,1] scale that is then rescaled onto the whole range for each rotator. Otherwise, uses the raw output as the delta rotation*/
	UPROPERTY(EditAnywhere, Category = "Actuator Settings")
	bool bNormalizeAndRescale = false;

	/** A delegate invoked when this actuator receives input from a brain */
	UPROPERTY(BlueprintAssignable)
	FOnRotationInputSignature OnRotationDelegate;

	
	FBoxSpace GetActionSpace() override;

	/**
	* @brief Convert a Box Point with 3 values to an FRotator
	* @param[in] Action BoxPoint that will be converted
	* @returns FRotator equivalent to the converted BoxPoin
	*/
	FRotator ConvertActionToFRotator(const FBoxPoint& Action);

	void TakeAction(const FBoxPoint& Action) override;

	FString GenerateId() const override;
};