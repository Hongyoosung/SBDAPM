// Copyright (c) 2023 Advanced Micro Devices, Inc. All Rights Reserved.

#include "Observers/PositionObserver.h"

# 

void UPositionObserver::CollectObservations(FBoxPoint& OutObservations)
{
	AActor* LocalTrackedActor = TrackedActor;
	// We are tracking the owner
	if (!bTrackNonOwner)
	{
		LocalTrackedActor = this->TryGetOwner();
	}

	if (LocalTrackedActor)
	{
		FVector ActorLocation = LocalTrackedActor->GetActorLocation();

		if (bTrackNonOwner && PositionAdjustment == EFrameOfReference::Egocentric)
		{
			// Make it so that the vector is relative to this actor's forward vector

			ActorLocation = this->TryGetOwner()->GetActorTransform().InverseTransformPositionNoScale(ActorLocation);
		}
		else if (bTrackNonOwner && PositionAdjustment == EFrameOfReference::Relative)
		{
			// Make the position relative from the owner
			ActorLocation -= this->TryGetOwner()->GetActorLocation();
		}

		if (bHasXDimensions)
		{
			OutObservations.Values.Add(ActorLocation.X);
		}

		if (bHasYDimensions)
		{
			OutObservations.Values.Add(ActorLocation.Y);
		}

		if (bHasZDimensions)
		{
			OutObservations.Values.Add(ActorLocation.Z);
		}
	}
}

FString UPositionObserver::GenerateId() const
{
	FString Output = FString("Position");
	//Add X Dimension
	if (bHasXDimensions)
	{
		//_X followed by upper and lower bound
		Output.Appendf(TEXT("_X_%.2f_%.2f"), XDimensionBounds.Low, XDimensionBounds.High);
	}
	//Add Y Dimension
	if (bHasYDimensions)
	{
		//_Y followed by upper and lower bound
		Output.Appendf(TEXT("_Y_%.2f_%.2f"), YDimensionBounds.Low, YDimensionBounds.High);
	}

	//Add Z Dimension
	if (bHasZDimensions)
	{
		//_Z followed by upper and lower bound
		Output.Appendf(TEXT("_Z_%.2f_%.2f"), ZDimensionBounds.Low, ZDimensionBounds.High);
	}

	if (bTrackNonOwner)
	{
		Output.Append("_Other");
		Output.Append("_").Append(UEnum::GetValueAsString<EFrameOfReference>(PositionAdjustment));
	}
	
	return Output;
}

FBoxSpace UPositionObserver::GetObservationSpace() const
{
	FBoxSpace SpaceDefinition;

	if (bHasXDimensions)
	{
		SpaceDefinition.Dimensions.Add(XDimensionBounds);
	}

	if (bHasYDimensions)
	{
		SpaceDefinition.Dimensions.Add(YDimensionBounds);
	}

	if (bHasZDimensions)
	{
		SpaceDefinition.Dimensions.Add(ZDimensionBounds);
	}

	return SpaceDefinition;
}