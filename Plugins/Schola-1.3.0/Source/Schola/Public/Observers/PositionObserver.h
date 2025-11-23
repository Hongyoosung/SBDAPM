// Copyright (c) 2023 Advanced Micro Devices, Inc. All Rights Reserved.

#pragma once

#include "CoreMinimal.h"
#include "Observers/AbstractObservers.h"
#include "Common/PositionalEnums.h"
#include "PositionObserver.generated.h"

/**
 * @brief An observer that tracks the position of an actor.
 */
UCLASS(Blueprintable)
class SCHOLA_API UPositionObserver : public UBoxObserver
{
	GENERATED_BODY()
protected:
	FString GenerateId() const;

public:
	/** The Min/Max value for the X dimension of the tracked position. */
	UPROPERTY(EditAnywhere, meta = (EditCondition = "bHasXDimensions"), Category = "Sensor Properties")
	FBoxSpaceDimension XDimensionBounds;

	/** The Min/Max value for the Y dimension of the tracked position. */
	UPROPERTY(EditAnywhere, meta = (EditCondition = "bHasYDimensions"), Category = "Sensor Properties")
	FBoxSpaceDimension YDimensionBounds;

	/** The Min/Max value for the Z dimension of the tracked position. */
	UPROPERTY(EditAnywhere, meta = (EditCondition = "bHasZDimensions"), Category = "Sensor Properties")
	FBoxSpaceDimension ZDimensionBounds;

	/** Should the observer track the X dimension of the position. */
	UPROPERTY(EditAnywhere, BlueprintReadOnly, Meta = (InlineEditConditionToggle), Category = "Sensor Properties")
	bool bHasXDimensions = true;

	/** Should the observer track the Z dimension of the position. */
	UPROPERTY(EditAnywhere, BlueprintReadOnly, Meta = (InlineEditConditionToggle), Category = "Sensor Properties")
	bool bHasZDimensions = true;

	/** Should the observer track the Y dimension of the position. */
	UPROPERTY(EditAnywhere, BlueprintReadOnly, Meta = (InlineEditConditionToggle), Category = "Sensor Properties")
	bool bHasYDimensions = true;

	/** The actor to track the position of. If None, defaults to the owner of the observer. */
	UPROPERTY(EditAnywhere, meta = (EditCondition = "bTrackNonOwner", DisplayName = "Track Other Actor"), Category = "Sensor Properties")
	AActor* TrackedActor;

	/** Should the observer track the position of an actor other than the owner. */
	UPROPERTY(EditAnywhere, BlueprintReadOnly, Meta = (InlineEditConditionToggle), Category = "Sensor Properties")
	bool bTrackNonOwner = false;

	/** What frame of reference to use for this sensor: egocentric to the owning actor, relative to the owning actor, or world coordinates. */
	UPROPERTY(EditAnywhere, Category = "Sensor Properties", meta = (EditCondition = "bTrackNonOwner", HideEditConditionToggle))
	TEnumAsByte<EFrameOfReference> PositionAdjustment = EFrameOfReference::World;

	FBoxSpace GetObservationSpace() const;

	virtual void CollectObservations(FBoxPoint& OutObservations) override;

};
