// Copyright (c) 2023 Advanced Micro Devices, Inc. All Rights Reserved.

#include "Actuators/RotationActuator.h"


FBoxSpace URotationActuator::GetActionSpace()
{
	FBoxSpace SpaceDefinition;

	if (bHasPitch)
	{
		if (bNormalizeAndRescale)
		{
			SpaceDefinition.Dimensions.Add(FBoxSpaceDimension::ZeroOneUnitDimension());
		}
		else
		{
			SpaceDefinition.Dimensions.Add(PitchBounds);
		}
	}

	if (bHasYaw)
	{
		if (bNormalizeAndRescale)
		{
			SpaceDefinition.Dimensions.Add(FBoxSpaceDimension::ZeroOneUnitDimension());
		}
		else
		{
			SpaceDefinition.Dimensions.Add(YawBounds);
		}
		
	}

	if (bHasRoll)
	{
		if (bNormalizeAndRescale)
		{
			SpaceDefinition.Dimensions.Add(FBoxSpaceDimension::ZeroOneUnitDimension());
		}
		else
		{
			SpaceDefinition.Dimensions.Add(RollBounds);
		}
	}

	return SpaceDefinition;
}

FRotator URotationActuator::ConvertActionToFRotator(const FBoxPoint& Action)
{
	float Pitch = 0;
	float Yaw = 0;
	float Roll = 0;

	int Offset = 0;
	if (bHasPitch)
	{
		Pitch = Action[Offset++];
		if (bNormalizeAndRescale)
		{
			Pitch = PitchBounds.RescaleValue(Pitch);
		}
	}

	if (bHasRoll)
	{	
		Roll = Action[Offset++];
		if (bNormalizeAndRescale)
		{
			Roll = RollBounds.RescaleValue(Roll);
		}
	}

	if (bHasYaw)
	{
		Yaw = Action[Offset++];
		if (bNormalizeAndRescale)
		{
			Yaw = YawBounds.RescaleValue(Yaw);
		}
	}

	return FRotator(Pitch, Yaw, Roll);
}

void URotationActuator::TakeAction(const FBoxPoint& Action)
{

	AActor* LocalTarget = this->TryGetOwner();
	
	if (LocalTarget != nullptr)
	{
		const FRotator& Rotation = ConvertActionToFRotator(Action);
		this->OnRotationDelegate.Broadcast(Rotation);
		LocalTarget->AddActorLocalRotation(Rotation, bSweep, nullptr, TeleportType);
	}
	else
	{
		UE_LOG(LogSchola, Warning, TEXT("RotationActuator %s: No Pawn found to apply rotation to."), *this->GetName());
	}
}

FString URotationActuator::GenerateId() const
{
	FString Output = FString("Rotation");

	if (bHasPitch)
	{
		Output.Appendf(TEXT("_Pitch_%f_%f"), PitchBounds.Low, PitchBounds.High);
	}
	if (bHasYaw)
	{
		Output.Appendf(TEXT("_Yaw_%f_%f"), YawBounds.Low, YawBounds.High);
	}
	if (bHasRoll)
	{
		Output.Appendf(TEXT("_Roll_%f_%f"), RollBounds.Low, RollBounds.High);
	}
	// Add if we are rescaling and Normalizing
	if (bNormalizeAndRescale)
	{
		Output.Append("_Rescaled");
	}

	return Output;
}
