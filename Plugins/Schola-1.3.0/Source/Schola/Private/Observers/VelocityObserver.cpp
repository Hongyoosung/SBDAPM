// Copyright (c) 2023 Advanced Micro Devices, Inc. All Rights Reserved.

#include "Observers/VelocityObserver.h"

void UVelocityObserver::CollectObservations(FBoxPoint& OutObservations)
{
	AActor* LocalTrackedActor = TrackedActor;
	// We are tracking the owner
	if (!bTrackNonOwner)
	{
		LocalTrackedActor = this->TryGetOwner();
	}

	if (LocalTrackedActor)
	{
		FVector ActorVelocity = LocalTrackedActor->GetVelocity();

		if (bHasXDimensions)
		{
			OutObservations.Values.Add(ActorVelocity.X);
		}

		if (bHasYDimensions)
		{
			OutObservations.Values.Add(ActorVelocity.Y);
		}

		if (bHasZDimensions)
		{
			OutObservations.Values.Add(ActorVelocity.Z);
		}
	}
}

FString UVelocityObserver::GenerateId() const
{
	FString Output = FString("Velocity");
	// Add X Dimension
	if (bHasXDimensions)
	{
		// _X followed by upper and lower bound
		Output.Appendf(TEXT("_X_%.2f_%.2f"), XDimensionBounds.Low, XDimensionBounds.High);
	}
	// Add Y Dimension
	if (bHasYDimensions)
	{
		// _Y followed by upper and lower bound
		Output.Appendf(TEXT("_Y_%.2f_%.2f"), YDimensionBounds.Low, YDimensionBounds.High);
	}
	// Add Z Dimension
	if (bHasZDimensions)
	{
		// _Z followed by upper and lower bound
		Output.Appendf(TEXT("_Z_%.2f_%.2f"), ZDimensionBounds.Low, ZDimensionBounds.High);
	}
	if (bTrackNonOwner)
	{
		Output.Append("_Other");
	}
	return Output;
}

FBoxSpace UVelocityObserver::GetObservationSpace() const
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