// Copyright (c) 2024 Advanced Micro Devices, Inc. All Rights Reserved.
#pragma once

#include "CoreMinimal.h"
#include "UObject/Interface.h"
#include "Common/Spaces.h"
#include "Common/Points.h"
#include "ActuatorWrapperInterfaces.generated.h"

UINTERFACE(MinimalAPI)
class UBoxActuatorWrapper : public UInterface
{
	GENERATED_BODY()
};

class SCHOLA_API IBoxActuatorWrapper
{
	GENERATED_BODY()

public:
	
	virtual void Reset() {};

	virtual FBoxSpace WrapBoxActionSpace(const FBoxSpace& Space);

	virtual FBoxPoint WrapBoxAction(const FBoxPoint& Point) PURE_VIRTUAL(IBoxActuatorWrapper::WrapBoxAction, return FBoxPoint(););

	virtual FString GenerateId() const PURE_VIRTUAL(IBoxActuatorWrapper::GenerateId, return FString(););
};

UINTERFACE(MinimalAPI)
class UDiscreteActuatorWrapper : public UInterface
{
	GENERATED_BODY()
};

class SCHOLA_API IDiscreteActuatorWrapper
{
	GENERATED_BODY()

public:

	virtual void		   Reset(){};
	virtual FDiscreteSpace WrapDiscreteActionSpace(const FDiscreteSpace& Space);

	virtual FDiscretePoint WrapDiscreteAction(const FDiscretePoint& Point) PURE_VIRTUAL(IDiscreteActuatorWrapper::WrapDiscreteAction, return FDiscretePoint(););
	virtual FString		   GenerateId() const PURE_VIRTUAL(IDiscreteActuatorWrapper::GenerateId, return FString(););
};

UINTERFACE(MinimalAPI)
class UBinaryActuatorWrapper : public UInterface
{
	GENERATED_BODY()
};

class SCHOLA_API IBinaryActuatorWrapper
{
	GENERATED_BODY()

public:
	
	virtual void		 Reset(){};
	virtual FBinarySpace WrapBinaryActionSpace(const FBinarySpace& Space);

	virtual FBinaryPoint WrapBinaryAction(const FBinaryPoint& Point) PURE_VIRTUAL(IBinaryActuatorWrapper::WrapBinaryAction, return FBinaryPoint(););
	virtual FString		 GenerateId()  const PURE_VIRTUAL(IBinaryActuatorWrapper::GenerateId, return FString(););

};


//Blueprint Interfaces

UINTERFACE(MinimalAPI, Blueprintable)
class UBlueprintBoxActuatorWrapper : public UBoxActuatorWrapper
{
	GENERATED_BODY()
};

class SCHOLA_API IBlueprintBoxActuatorWrapper : public IBoxActuatorWrapper
{
	GENERATED_BODY()

public:
	
	UFUNCTION(BlueprintImplementableEvent)
	void Reset();
	
	UFUNCTION(BlueprintImplementableEvent)
	FBoxSpace WrapBoxActionSpace(const FBoxSpace& Space);
	
	UFUNCTION(BlueprintImplementableEvent)
	FBoxPoint WrapBoxAction(const FBoxPoint& Point);

	UFUNCTION(BlueprintNativeEvent)
	FString GenerateId() const;

	FString GenerateId_Implementation() const
	{
		return this->_getUObject()->GetClass()->GetName();
	};

};

UINTERFACE(MinimalAPI, Blueprintable)
class UBlueprintDiscreteActuatorWrapper : public UDiscreteActuatorWrapper
{
	GENERATED_BODY()
};

class SCHOLA_API IBlueprintDiscreteActuatorWrapper : public IDiscreteActuatorWrapper
{
	GENERATED_BODY()

public:

	UFUNCTION(BlueprintImplementableEvent)
	void Reset();

	UFUNCTION(BlueprintImplementableEvent)
	FDiscreteSpace WrapDiscreteActionSpace(const FDiscreteSpace& Space);

	UFUNCTION(BlueprintImplementableEvent)
	FDiscretePoint WrapDiscreteAction(const FDiscretePoint& Point);

	UFUNCTION(BlueprintNativeEvent)
	FString GenerateId() const;

	FString GenerateId_Implementation() const
	{
		return this->_getUObject()->GetClass()->GetName();
	};
};

UINTERFACE(MinimalAPI, Blueprintable)
class UBlueprintBinaryActuatorWrapper : public UBinaryActuatorWrapper
{
	GENERATED_BODY()
};

class SCHOLA_API IBlueprintBinaryActuatorWrapper : public IBinaryActuatorWrapper
{
	GENERATED_BODY()

public:
	UFUNCTION(BlueprintImplementableEvent)
	void		 Reset();

	UFUNCTION(BlueprintImplementableEvent)
	FBinarySpace WrapBinaryActionSpace(const FBinarySpace& Space);

	UFUNCTION(BlueprintImplementableEvent)
	FBinaryPoint WrapBinaryAction(const FBinaryPoint& Point);

	UFUNCTION(BlueprintNativeEvent)
	FString GenerateId() const;

	FString GenerateId_Implementation() const
	{
		return this->_getUObject()->GetClass()->GetName();
	};
};
