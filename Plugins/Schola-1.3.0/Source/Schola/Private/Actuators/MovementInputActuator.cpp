// Copyright (c) 2023 Advanced Micro Devices, Inc. All Rights Reserved.

#include "Actuators/MovementInputActuator.h"

FBoxSpace UMovementInputActuator::GetActionSpace()
{
	FBoxSpace OutSpace;
	if (bHasXDimension)
	{
		OutSpace.Add(MinSpeed, MaxSpeed);
	}

	if (bHasYDimension)
	{
		OutSpace.Add(MinSpeed, MaxSpeed);
	}

	if (bHasZDimension)
	{
		OutSpace.Add(MinSpeed, MaxSpeed);
	}
	return OutSpace;
}

FVector UMovementInputActuator::ConvertActionToFVector(const FBoxPoint& Action)
{
	FVector OutVector;
	int		Offset = 0;
	if (bHasXDimension)
	{
		OutVector.X = Action[Offset];
		Offset++;
	}

	if (bHasYDimension)
	{
		OutVector.Y = Action[Offset];
		Offset++;
	}

	if (bHasZDimension)
	{
		OutVector.Z = Action[Offset];
		Offset++;
	}

	return OutVector;
}

void UMovementInputActuator::TakeAction(const FBoxPoint& Action)
{
	int Offset = 0;

	APawn* LocalTarget = Cast<APawn>(TryGetOwner());

	if (LocalTarget != nullptr)
	{
		const FVector& ActionVector = ConvertActionToFVector(Action);
		
		this->OnMovementDelegate.Broadcast(ActionVector);
		LocalTarget->AddMovementInput(LocalTarget->GetActorRotation().RotateVector(ActionVector), ScaleValue, bForce);
	}
	else
	{
		UE_LOG(LogSchola, Warning, TEXT("MovementInputActuator %s: No Pawn found to apply movement input to."), *this->GetName());
	}
}

FString UMovementInputActuator::GenerateId() const
{
	FString Output = FString("MovementInput");

	// Add Dimensions one at a time
	Output.Append("_");

	Output.Append(bHasXDimension ? "X" : "");

	Output.Append(bHasYDimension ? "Y" : "");

	Output.Append(bHasZDimension ? "Z" : "");

	// Add Min and Max Speed
	Output.Appendf(TEXT("_%.2f_%.2f"), MinSpeed, MaxSpeed);
	
	return Output;
}
