// Copyright (c) 2023 Advanced Micro Devices, Inc. All Rights Reserved.

#include "Observers/AbstractObservers.h"


#if WITH_EDITOR
void UAbstractObserver::TestObserverValidity()
{
	TPoint Observations;
	TSpace ObsSpace;
	FillObservationSpace(ObsSpace);
	CollectObservations(Observations);
	SetDebugObservations(Observations);
	this->ObservationShape = Visit([](auto& TypedObsSpace) { return TypedObsSpace.GetNumDimensions(); }, ObsSpace);
	this->ValidationResult = Visit([&Observations](auto& TypedObsSpace) { return TypedObsSpace.Validate(Observations); }, ObsSpace);
}

void UBoxObserver::SetDebugObservations(TPoint& Temp)
{
	this->DebugBoxPoint = Temp.Get<FBoxPoint>().Values;
}

void UBinaryObserver::SetDebugObservations(TPoint& Temp)
{
	this->DebugBinaryPoint = Temp.Get<FBinaryPoint>().Values;
}

void UDiscreteObserver::SetDebugObservations(TPoint& Temp)
{
	this->DebugDiscretePoint = Temp.Get<FDiscretePoint>().Values;
}

#endif

FString UBoxObserver::GetId() const
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
			IBoxObserverWrapper* Wrapper = Cast<IBoxObserverWrapper>(RawWrapper);
			if (Wrapper)
			{
				Id.Append("_").Append(Wrapper->GenerateId());
			}
		}
		return Id;
	}
}

FString UDiscreteObserver::GetId() const
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
			IDiscreteObserverWrapper* Wrapper = Cast<IDiscreteObserverWrapper>(RawWrapper);
			if (Wrapper)
			{
				Id.Append("_").Append(Wrapper->GenerateId());
			}
		}
		return Id;
	}
}

FString UBinaryObserver::GetId() const
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
			IBinaryObserverWrapper* Wrapper = Cast<IBinaryObserverWrapper>(RawWrapper);
			if (Wrapper)
			{
				Id.Append("_").Append(Wrapper->GenerateId());
			}
		}
		return Id;
	}
}
