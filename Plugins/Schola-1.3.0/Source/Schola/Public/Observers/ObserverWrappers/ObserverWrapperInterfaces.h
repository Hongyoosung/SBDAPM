// Copyright (c) 2024 Advanced Micro Devices, Inc. All Rights Reserved.
#pragma once

#include "CoreMinimal.h"
#include "UObject/Interface.h"
#include "Common/Spaces.h"
#include "Common/Points.h"
#include "ObserverWrapperInterfaces.generated.h"

/**
 * @brief Interface for wrapping box observations.
 */
UINTERFACE(MinimalAPI)
class UBoxObserverWrapper : public UInterface
{
	GENERATED_BODY()
};

class SCHOLA_API IBoxObserverWrapper
{
	GENERATED_BODY()

public:
	
	virtual void Reset() {};

	virtual FBoxSpace WrapBoxObservationSpace(const FBoxSpace& Space);

	virtual FBoxPoint WrapBoxObservation(const FBoxPoint& Point) PURE_VIRTUAL(IBoxObserverWrapper::WrapBoxObservation, return FBoxPoint(););

	virtual FString GenerateId() const PURE_VIRTUAL(IBoxObserverWrapper::GenerateId, return FString(););
};

/**
 * @brief Interface for wrapping discrete observations.
 */
UINTERFACE(MinimalAPI)
class UDiscreteObserverWrapper : public UInterface
{
	GENERATED_BODY()
};

class SCHOLA_API IDiscreteObserverWrapper
{
	GENERATED_BODY()

public:

	virtual void		   Reset(){};
	virtual FDiscreteSpace WrapDiscreteObservationSpace(const FDiscreteSpace& Space);

	virtual FDiscretePoint WrapDiscreteObservation(const FDiscretePoint& Point) PURE_VIRTUAL(IDiscreteObserverWrapper::WrapDiscreteObservation, return FDiscretePoint(););

	virtual FString GenerateId() const PURE_VIRTUAL(IDiscreteObserverWrapper::GenerateId, return FString(););
};

/**
 * @brief Interface for wrapping binary observations.
 */
UINTERFACE(MinimalAPI)
class UBinaryObserverWrapper : public UInterface
{
	GENERATED_BODY()
};

class SCHOLA_API IBinaryObserverWrapper
{
	GENERATED_BODY()

public:
	
	virtual void		 Reset(){};
	virtual FBinarySpace WrapBinaryObservationSpace(const FBinarySpace& Space);

	virtual FBinaryPoint WrapBinaryObservation(const FBinaryPoint& Point) PURE_VIRTUAL(IBinaryObserverWrapper::WrapBinaryObservation, return FBinaryPoint(););

	virtual FString GenerateId() const PURE_VIRTUAL(IBinaryObserverWrapper::GenerateId, return FString(););
};


//Blueprint Interfaces

/**
 * @brief Blueprint interface for wrapping box observations.
 */
UINTERFACE(MinimalAPI, Blueprintable)
class UBlueprintBoxObserverWrapper : public UBoxObserverWrapper
{
	GENERATED_BODY()
};

class SCHOLA_API IBlueprintBoxObserverWrapper : public IBoxObserverWrapper
{
	GENERATED_BODY()

public:
	
	UFUNCTION(BlueprintImplementableEvent)
	void Reset();
	
	UFUNCTION(BlueprintImplementableEvent)
	FBoxSpace WrapBoxObservationSpace(const FBoxSpace& Space);
	
	UFUNCTION(BlueprintImplementableEvent)
	FBoxPoint WrapBoxObservation(const FBoxPoint& Point);

	UFUNCTION(BlueprintNativeEvent)
	FString GenerateId() const;

	FString GenerateId_Implementation() const
	{
		return this->_getUObject()->GetClass()->GetName();
	};
};

/**
 * @brief Blueprint interface for wrapping discrete observations.
 */
UINTERFACE(MinimalAPI, Blueprintable)
class UBlueprintDiscreteObserverWrapper : public UDiscreteObserverWrapper
{
	GENERATED_BODY()
};

class SCHOLA_API IBlueprintDiscreteObserverWrapper: public IDiscreteObserverWrapper
{
	GENERATED_BODY()

public:

	UFUNCTION(BlueprintImplementableEvent)
	void Reset();

	UFUNCTION(BlueprintImplementableEvent)
	FDiscreteSpace WrapDiscreteObservationSpace(const FDiscreteSpace& Space);

	UFUNCTION(BlueprintImplementableEvent)
	FDiscretePoint WrapDiscreteObservation(const FDiscretePoint& Point);

	UFUNCTION(BlueprintNativeEvent)
	FString GenerateId() const;

	FString GenerateId_Implementation() const
	{
		return this->_getUObject()->GetClass()->GetName();
	};
};

/**
 * @brief Blueprint interface for wrapping binary observations.
 */
UINTERFACE(MinimalAPI, Blueprintable)
class UBlueprintBinaryObserverWrapper : public UBinaryObserverWrapper
{
	GENERATED_BODY()
};

class SCHOLA_API IBlueprintBinaryObserverWrapper : public IBinaryObserverWrapper
{
	GENERATED_BODY()

public:
	UFUNCTION(BlueprintImplementableEvent)
	void		 Reset();

	UFUNCTION(BlueprintImplementableEvent)
	FBinarySpace WrapBinaryObservationSpace(const FBinarySpace& Space);

	UFUNCTION(BlueprintImplementableEvent)
	FBinaryPoint WrapBinaryObservation(const FBinaryPoint& Point);

	UFUNCTION(BlueprintNativeEvent)
	FString GenerateId() const;

	FString GenerateId_Implementation() const
	{
		return this->_getUObject()->GetClass()->GetName();
	};
};
