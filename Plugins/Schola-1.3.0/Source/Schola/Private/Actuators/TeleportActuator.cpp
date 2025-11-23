// Copyright (c) 2023 Advanced Micro Devices, Inc. All Rights Reserved.

#include "Actuators/TeleportActuator.h"

FDiscreteSpace UTeleportActuator::GetActionSpace()
{
	FDiscreteSpace SpaceDefinition;
	int			   NumDirections = (int32)StaticEnum<ETeleportDirection>()->GetMaxEnumValue();
	
	if (bHasXDimension)
	{
		SpaceDefinition.Add(NumDirections);
	}
	
	if (bHasYDimension)
	{
		SpaceDefinition.Add(NumDirections);
	}

	if (bHasZDimension)
	{
		SpaceDefinition.Add(NumDirections);
	}

	return SpaceDefinition;
}

FVector UTeleportActuator::ConvertActionToFVector(const FDiscretePoint& Action)
{
	FVector OutVector = FVector(0);
	int		Offset = 0;

	if (bHasXDimension)
	{
		OutVector.X = GetVectorDimension(XDimensionSpeed, Action[Offset++]);
	}
	if (bHasYDimension)
	{
		OutVector.Y = GetVectorDimension(YDimensionSpeed, Action[Offset++]);
	}
	if (bHasZDimension)
	{
		OutVector.Z = GetVectorDimension(ZDimensionSpeed, Action[Offset++]);
	}
	return OutVector;
}

float UTeleportActuator::GetVectorDimension(int Speed, int DiscretePointValue)
{

	switch ((ETeleportDirection)DiscretePointValue)
	{
		case ETeleportDirection::Nothing:
			return 0;
		case ETeleportDirection::Forward:
			return Speed;
		case ETeleportDirection::Backward:
			return -1 * Speed;
			break;
		default:
			return 0;
	}
}

void UTeleportActuator::TakeAction(const FDiscretePoint& Action)
{
	
	AActor* LocalTarget = TryGetOwner();
	
	if (LocalTarget != nullptr)
	{
		const FVector& ActionVector = ConvertActionToFVector(Action);
		this->OnTeleportDelegate.Broadcast(ActionVector);
		LocalTarget->SetActorLocation(LocalTarget->GetActorLocation() + ActionVector, this->bSweep, nullptr, this->TeleportType);
	}
	else
	{
		UE_LOG(LogSchola, Warning, TEXT("TeleportActuator %s: No Pawn found to apply action to."), *this->GetName());
	}
}

FString UTeleportActuator::GenerateId() const
{
	FString Output = FString("Teleport");

	if (bHasXDimension)
	{
		Output.Appendf(TEXT("_X_%.2f"), XDimensionSpeed);
	}

	if (bHasYDimension)
	{
		Output.Appendf(TEXT("_Y_%.2f"), YDimensionSpeed);
	}

	if (bHasZDimension)
	{
		Output.Appendf(TEXT("_Z_%.2f"), ZDimensionSpeed);
	}

	return Output;
}
