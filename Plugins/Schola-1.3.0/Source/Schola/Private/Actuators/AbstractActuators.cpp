// Copyright (c) 2023 Advanced Micro Devices, Inc. All Rights Reserved.

#include "Actuators/AbstractActuators.h"


AActor* UActuator::SpawnActor(TSubclassOf<AActor> Class, const FTransform& SpawnTransform, ESpawnActorCollisionHandlingMethod CollisionHandlingOverride, ESpawnActorScaleMethod TransformScaleMethod, AActor* Owner, APawn* Instigator)
{
	FActorSpawnParameters Parameters = FActorSpawnParameters();
	Parameters.SpawnCollisionHandlingOverride = CollisionHandlingOverride;
	Parameters.TransformScaleMethod = TransformScaleMethod;
	Parameters.Instigator = Instigator;
	Parameters.Owner = Owner;

	return this->GetWorld()->SpawnActor<AActor>(Class, SpawnTransform, Parameters);
};


FString UBoxActuator::GetId() const
{
	if (this->bUseCustomId)
	{
		return this->CustomId;
	}
	else
	{
		FString Id = this->GenerateId();

		for (UObject* RawWrapper : this->Wrappers)
		{
			IBoxActuatorWrapper* Wrapper = Cast<IBoxActuatorWrapper>(RawWrapper);
			if (Wrapper)
			{
				Id.Append("_").Append(Wrapper->GenerateId());
			}
		}
		return this->GenerateId();
	}
}

FString UDiscreteActuator::GetId() const
{
	if (this->bUseCustomId)
	{
		return this->CustomId;
	}
	else
	{
		FString Id = this->GenerateId();

		for (UObject* RawWrapper : this->Wrappers)
		{
			IDiscreteActuatorWrapper* Wrapper = Cast<IDiscreteActuatorWrapper>(RawWrapper);
			if (Wrapper)
			{
				Id.Append("_").Append(Wrapper->GenerateId());
			}
		}
		return this->GenerateId();
	}
}

FString UBinaryActuator::GetId() const
{
	if (this->bUseCustomId)
	{
		return this->CustomId;
	}
	else
	{
		FString Id = this->GenerateId();

		for (UObject* RawWrapper : this->Wrappers)
		{
			IBinaryActuatorWrapper* Wrapper = Cast<IBinaryActuatorWrapper>(RawWrapper);
			if (Wrapper)
			{
				Id.Append("_").Append(Wrapper->GenerateId());
			}
		}
		return this->GenerateId();
	}
}