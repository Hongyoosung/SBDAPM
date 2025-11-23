// Copyright (c) 2023 Advanced Micro Devices, Inc. All Rights Reserved.

#pragma once

#include "CoreMinimal.h"
#include "Observers/AbstractObservers.h"
#include "RotationObserver.generated.h"

/**
 * @brief An observer that tracks the rotation of an actor.
 */
UCLASS(Blueprintable)
class SCHOLA_API URotationObserver : public UBoxObserver
{
	GENERATED_BODY()

public:
	/** The Min/Max value for the pitch of the tracked rotation. */
	UPROPERTY(VisibleAnywhere, meta = (EditCondition = "bHasPitch"), Category = "Sensor Properties")
	FBoxSpaceDimension PitchBounds = FBoxSpaceDimension(-180, 180);

	/** The Min/Max value for the roll of the tracked rotation. */
	UPROPERTY(VisibleAnywhere, meta = (EditCondition = "bHasRoll"), Category = "Sensor Properties")
	FBoxSpaceDimension RollBounds = FBoxSpaceDimension(-180, 180);

	/** The Min/Max value for the yaw of the tracked rotation. */
	UPROPERTY(VisibleAnywhere, meta = (EditCondition = "bHasYaw"), Category = "Sensor Properties")
	FBoxSpaceDimension YawBounds = FBoxSpaceDimension(-180, 180);

	/** Should the observer track the pitch of the rotation. */
	UPROPERTY(EditAnywhere, BlueprintReadOnly, Meta = (InlineEditConditionToggle), Category = "Sensor Properties")
	bool bHasPitch = true;

	/** Should the observer track the roll of the rotation. */
	UPROPERTY(EditAnywhere, BlueprintReadOnly, Meta = (InlineEditConditionToggle), Category = "Sensor Properties")
	bool bHasRoll = true;

	/** Should the observer track the yaw of the rotation. */
	UPROPERTY(EditAnywhere, BlueprintReadOnly, Meta = (InlineEditConditionToggle), Category = "Sensor Properties")
	bool bHasYaw = true;

	/** The actor to track the rotation of. If None, defaults to the owner of the observer. */
	UPROPERTY(EditAnywhere, meta = (EditCondition = "bTrackNonOwner", DisplayName = "Track Other Actor"), Category = "Sensor Properties")
	AActor* TrackedActor;

	/** Should the observer track the rotation of an actor other than the owner. */
	UPROPERTY(EditAnywhere, BlueprintReadOnly, Meta = (InlineEditConditionToggle), Category = "Sensor Properties")
	bool bTrackNonOwner = false;

	FBoxSpace GetObservationSpace() const;

	virtual void CollectObservations(FBoxPoint& OutObservations) override;

	virtual FString GenerateId() const override;
};
