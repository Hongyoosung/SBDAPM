// Copyright (c) 2023 Advanced Micro Devices, Inc. All Rights Reserved.

#include "Observers/RotationObserver.h"

void URotationObserver::CollectObservations(FBoxPoint& OutObservations)
{
	AActor* LocalTrackedActor = TrackedActor;
	// We are tracking the owner
	if (!bTrackNonOwner)
	{
		LocalTrackedActor = this->TryGetOwner();
	}

	if (LocalTrackedActor)
	{
		FRotator ActorRotation = LocalTrackedActor->GetActorRotation().GetNormalized();

		if (bHasPitch)
		{
			OutObservations.Values.Add(ActorRotation.Pitch);
		}

		if (bHasYaw)
		{
			OutObservations.Values.Add(ActorRotation.Yaw);
		}

		if (bHasRoll)
		{
			OutObservations.Values.Add(ActorRotation.Roll);
		}
	}
}

FString URotationObserver::GenerateId() const
{
	FString Output = FString("Rotation");
	// Add Pitch Dimension
	if (bHasPitch)
	{
		// _Pitch followed by upper and lower bound
		Output.Appendf(TEXT("_Pitch_%.2f_%.2f"), PitchBounds.Low, PitchBounds.High);
	}
	// Add Yaw Dimension
	if (bHasYaw)
	{
		// _Yaw followed by upper and lower bound
		Output.Appendf(TEXT("_Yaw_%.2f_%.2f"), YawBounds.Low, YawBounds.High);
	}
	// Add Roll Dimension
	if (bHasRoll)
	{
		// _Roll followed by upper and lower bound
		Output.Appendf(TEXT("_Roll_%.2f_%.2f"), RollBounds.Low, RollBounds.High);
	}
	if (bTrackNonOwner)
	{
		Output.Append("_Other");
	}

	return Output;
}

FBoxSpace URotationObserver::GetObservationSpace() const
{
	FBoxSpace SpaceDefinition;

	if (bHasPitch)
	{
		SpaceDefinition.Dimensions.Add(PitchBounds);
	}

	if (bHasYaw)
	{
		SpaceDefinition.Dimensions.Add(YawBounds);
	}

	if (bHasRoll)
	{
		SpaceDefinition.Dimensions.Add(RollBounds);
	}

	return SpaceDefinition;
}