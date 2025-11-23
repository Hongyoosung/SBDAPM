// Copyright (c) 2024 Advanced Micro Devices, Inc. All Rights Reserved.

#include "Common/AbstractInteractor.h"

AActor* UAbstractInteractor::TryGetOwner() const
{
	// Four possible locations
	// 1. In Sensor/Actuator/InferenceAgent component on the Agent Pawn
	// 2. In Sensor/Actuator component on the Trainer/Controller
	// 3. In the Controller directly
	// 4. In InferencePawn directly
	// 5. *New* in BTTask directly

	UObject* Outer = this->GetOuter();
	if (Outer)
	{
		UActorComponent* Comp = Cast<UActorComponent>(Outer);
		AController*	 Controller = Cast<AController>(Outer);
		UBTTaskNode*	 BTTask = Cast<UBTTaskNode>(Outer);

		if (Comp)
		{
			// In a Component
			AActor* CompOwner = Comp->GetOwner();
			Controller = Cast<AController>(CompOwner);

			if (Controller)
			{
				// In a Component in the Trainer
				return Controller->GetPawn();
			}
			else
			{
				return CompOwner;
			}
		}
		else if (Controller)
		{
			// Directly in the Trainer
			return Controller->GetPawn();
		}
		else if (BTTask)
		{
			// In a BTTask
			return Cast<AActor>(Cast<AController>(Cast<UBehaviorTreeComponent>(BTTask->GetOuter())->GetAIOwner())->GetPawn());
		}
		else
		{
			// In the Pawn
			return Cast<AActor>(Outer);
		}
	}
	else
	{
		return nullptr;
	}
}

UObject* UAbstractInteractor::GetLocation() const
{
	UObject* Outer = this->GetOuter();
	if (Outer)
	{
		UActorComponent* Comp = Cast<UActorComponent>(Outer);
		AController*	 Controller = Cast<AController>(Outer);

		if (Comp)
		{
			return Comp->GetOwner(); // Return Either the Controller, if that is where this comp is, or the Pawn
		}
		else if (Controller)
		{
			return Controller->GetOuter(); // Return the Pawn or other entity owning this Controller
		}
		else
		{
			return Outer;
		}
	}
	else
	{
		return nullptr;
	}
}

FString UAbstractInteractor::GetLabel() const
{
	UObject* Outer = this->GetOuter();
	if (Outer)
	{
		UActorComponent* Comp = Cast<UActorComponent>(Outer);
		AController*	 Controller = Cast<AController>(Outer);

		if (Comp)
		{
			return Comp->GetName(); // Return Either the Controller, if that is where this comp is, or the Pawn
		}
		else
		{
			return Outer->GetClass()->GetName() + FString("_") + this->GetName(); // Return the Pawn or other entity owning this Controller
		}
	}
	else
	{
		return FString("None_") + this->GetName();
	}
}

FString UAbstractInteractor::GetId() const
{

	if (this->bUseCustomId)
	{
		return this->CustomId;
	}
	else
	{
		return this->GenerateId();
	}
}

FString UAbstractInteractor::GetSanitizedId() const
{
	FString Output = this->GetId();
	Output.ReplaceCharInline('.', ',');
	return Output;
}

FString UAbstractInteractor::GenerateId() const
{
	return this->GetLabel();
}
