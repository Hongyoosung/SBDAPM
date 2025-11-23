// Copyright (c) 2023 Advanced Micro Devices, Inc. All Rights Reserved.

#pragma once

#include "CoreMinimal.h"
#include "Components/ActorComponent.h"
#include "Containers/Array.h"
#include "Common/Spaces.h"
#include "Common/Points.h"
#include "Common/IValidatable.h"
#include "Common/LogSchola.h"
#include "Common/AbstractInteractor.h"
#include "Observers/ObserverWrappers/ObserverWrapperInterfaces.h"
#include "AbstractObservers.generated.h"

/**
 * @brief An abstract class representing an observer that can collect observations about the environment state.
 */
UCLASS(Abstract, EditInlineNew)
class SCHOLA_API UAbstractObserver : public UAbstractInteractor
{
	GENERATED_BODY()

public:
	/**
	 * @brief Create a new instance of UAbstractObserver.
	 */
	UAbstractObserver() = default;
	
	/**
	 * @brief Get the ObservationSpace bounding the outputs of this sensor.
	 * @param[out] OutSpace The ObservationSpace bounding the outputs of this sensor.
	 * @note This function should be implemented by any derived classes.
	 */
	virtual void FillObservationSpace(TSpace& OutSpace) const PURE_VIRTUAL(UAbstractObserver::FillObservationSpace, return; );

	/**
	 * @brief Use this sensor to collect observations about the environment state.
	 * @param OutObservations - DataPoint that will be updated with the outputs of this sensor.
	 * @note This function should be implemented by any derived classes.
	 */
	virtual void CollectObservations(TPoint& OutObservations) PURE_VIRTUAL(UAbstractObserver::CollectObservations, return; );

	/**
	 * @brief Do any subclass-specific setup.
	 * @note This function should be implemented by any derived classes.
	 */
	virtual void InitializeObserver() {};

	/**
	* @brief Reset the state of this observer, called when an episode ends.
	* @note This function should be implemented by any derived classes.
	*/
	virtual void ResetObserver() {};

	
	virtual void Reset() override { this->ResetObserver(); };


#if WITH_EDITOR
	
	/**
	 * @brief Test whether this sensor's output matches the constraints defined by the return of GetObservationSpace.
	 * @note Sets the ValidationResult and ObservationShape properties.
	 */
	void TestObserverValidity();

	/**
	 * @brief Set the debug observations for this observer.
	 * @param[in] Temp The temporary point to copy to the debug observations.
	 * @note This function should be implemented by any derived classes, to use the correct type of point.
	 */
	virtual void SetDebugObservations(TPoint& Temp) PURE_VIRTUAL(UAbstractObserver::SetDebugObservations, return;);

#endif

#if WITH_EDITORONLY_DATA

	/** The results of the most recent validation. */
	UPROPERTY(VisibleInstanceOnly, Category = "Observer Utilities")
	ESpaceValidationResult ValidationResult = ESpaceValidationResult::NoResults;

	/** The number of dimensions in the space set on this object. */
	UPROPERTY(VisibleInstanceOnly, NoClear, Category = "Observer Utilities")
	int ObservationShape = 0;

#endif
};

/**
 * @brief An observer that collects box(continuous) observations about the environment state.
 * @note This class is abstract and should be derived from to implement the CollectObservations and GetObservationSpace functions.
 */
UCLASS(Abstract)
class SCHOLA_API UBoxObserver : public UAbstractObserver
{
	GENERATED_BODY()


public:
	/**
	 * @brief Create a new instance of UBoxObserver
	 */
	UBoxObserver(){};


	/** Wrappers for customizing the observations recorded by this observer. Applied in order. */
	UPROPERTY(EditAnywhere, Instanced, meta = (AllowedClasses = "/Script/Schola.BoxObserverWrapper", ExactClass = false), Category="Sensor Properties")
	TArray<UObject*> Wrappers;

	/**
	 * @brief Get the BoxSpace bounding the outputs of this sensor.
	 * @return The ObservationSpace bounding the outputs of this sensor.
	 * @note This function should be implemented by any derived classes
	 */
	virtual FBoxSpace GetObservationSpace() const PURE_VIRTUAL(UBoxObserver::GetObservationSpace, return FBoxSpace(););

	/**
	 * @brief Use this sensor to collect observations about the environment state
	 * @param[out] OutObservations DataPoint that will be updated with the outputs of this sensor.
	 * @note This function should be implemented by any derived classes
	 */
	virtual void CollectObservations(FBoxPoint& OutObservations) PURE_VIRTUAL(UBoxObserver::CollectObservations, return; );

	void Reset() override 
	{
		for (TScriptInterface<IBoxObserverWrapper> Wrapper : this->Wrappers)
		{
			Wrapper->Reset();
		}
		this->ResetObserver();
	};

	void CollectObservations(TPoint& OutObservations) override
	{
		
		FBoxPoint OutputPoint;
		this->CollectObservations(OutputPoint);
		for (TScriptInterface<IBoxObserverWrapper> Wrapper : this->Wrappers)
		{
			OutputPoint = Wrapper->WrapBoxObservation(OutputPoint);
		}
		OutObservations.Set<FBoxPoint>(OutputPoint);
		#if WITH_EDITOR
				this->SetDebugObservations(OutObservations);
		#endif
	}

	void FillObservationSpace(TSpace& OutSpaceGroup) const override
	{	
		FBoxSpace OutputSpace = this->GetObservationSpace();
		
		for (TScriptInterface<IBoxObserverWrapper> Wrapper : this->Wrappers)
		{
			OutputSpace = Wrapper->WrapBoxObservationSpace(OutputSpace);
		}

		OutSpaceGroup.Set<FBoxSpace>(OutputSpace);
	}

	FString GetId() const;


#if WITH_EDITOR
	void SetDebugObservations(TPoint& Temp) override;
#endif

#if WITH_EDITORONLY_DATA
	/** The debug observations for this observer. Shows the last collected observation  */
	UPROPERTY(VisibleInstanceOnly, Category = "Observer Utilities")
	TArray<float> DebugBoxPoint;
#endif
};

/**
 * @brief A blueprintable version of UBoxObserver
 */
UCLASS(Blueprintable, Abstract)
class SCHOLA_API UBlueprintBoxObserver : public UBoxObserver
{
	GENERATED_BODY()

public:
	UBlueprintBoxObserver(){};

	UFUNCTION(BlueprintImplementableEvent)
	FBoxSpace GetObservationSpace() const;

	UFUNCTION(BlueprintImplementableEvent)
	void CollectObservations(FBoxPoint& OutObservations);


	UFUNCTION(BlueprintImplementableEvent)
	void InitializeObserver() override;

};

/**
 * @brief An observer that collects binary observations about the environment state.
 * @note This class is abstract and should be derived from to implement the CollectObservations and GetObservationSpace functions.
 */
UCLASS(Abstract)
class SCHOLA_API UBinaryObserver : public UAbstractObserver
{
	GENERATED_BODY()

public:

	/** Wrappers for customizing the observations recorded by this observer. Applied in order. */
	UPROPERTY(EditAnywhere, Instanced, meta = (AllowedClasses = "/Script/Schola.BinaryObserverWrapper", ExactClass = false), Category = "Sensor Properties")
	TArray<UObject*> Wrappers;

	UBinaryObserver(){};

	/**
	 * @brief Get the BinarySpace bounding the outputs of this sensor.
	 * @return The ObservationSpace bounding the outputs of this sensor.
	 * @note This function should be implemented by any derived classes
	 */
	virtual FBinarySpace GetObservationSpace() const PURE_VIRTUAL(UBinaryObserver::GetObservationSpace, return FBinarySpace(););

	/**
	 * @brief Use this sensor to collect observations about the environment state
	 * @param[out] OutObservations DataPoint that will be updated with the outputs of this sensor.
	 * @note This function should be implemented by any derived classes
	 */
	virtual void CollectObservations(FBinaryPoint& OutObservations) PURE_VIRTUAL(UBinaryObserver::CollectObservations, return; );

	void CollectObservations(TPoint& OutObservations)
	{
		
		FBinaryPoint OutputPoint;
		this->CollectObservations(OutputPoint);
		for (TScriptInterface<IBinaryObserverWrapper> Wrapper : this->Wrappers)
		{
			OutputPoint = Wrapper->WrapBinaryObservation(OutputPoint);
		}
		OutObservations.Set<FBinaryPoint>(OutputPoint);
		#if WITH_EDITOR
			this->SetDebugObservations(OutObservations);
		#endif
	}

	void FillObservationSpace(TSpace& OutSpaceGroup) const
	{
		FBinarySpace OutputSpace = this->GetObservationSpace();

		for (TScriptInterface<IBinaryObserverWrapper> Wrapper : this->Wrappers)
		{
			OutputSpace = Wrapper->WrapBinaryObservationSpace(OutputSpace);
		}

		OutSpaceGroup.Set<FBinarySpace>(this->GetObservationSpace());
	}

	FString GetId() const;

#if WITH_EDITOR
	void SetDebugObservations(TPoint& Temp) override;
#endif

#if WITH_EDITORONLY_DATA
	/** The debug observations for this observer. Shows the last collected observation  */
	UPROPERTY(VisibleInstanceOnly, Category = "Observer Utilities")
	TArray<bool> DebugBinaryPoint;
#endif
};

/**
 * @brief A blueprintable version of UBinaryObserver
 * @note This class is abstract and should be derived from to implement the CollectObservations and GetObservationSpace functions.
 */
UCLASS(Blueprintable, Abstract)
class SCHOLA_API UBlueprintBinaryObserver : public UBinaryObserver
{
	GENERATED_BODY()

public:
	UBlueprintBinaryObserver(){};

	UFUNCTION(BlueprintImplementableEvent)
	FBinarySpace GetObservationSpace() const;

	UFUNCTION(BlueprintImplementableEvent)
	void CollectObservations(FBinaryPoint& OutObservations);

	UFUNCTION(BlueprintImplementableEvent)
	void InitializeObserver();

};

/**
 * @brief An observer that collects discrete observations about the environment state.
 * @note This class is abstract and should be derived from to implement the CollectObservations and GetObservationSpace functions.
 */
UCLASS(Abstract)
class SCHOLA_API UDiscreteObserver : public UAbstractObserver
{
	GENERATED_BODY()

public:

	/** Wrappers for customizing the observations recorded by this observer. Applied in order. */
	UPROPERTY(EditAnywhere, Instanced, meta = (AllowedClasses = "/Script/Schola.DiscreteObserverWrapper", ExactClass = false), Category = "Sensor Properties")
	TArray<UObject*> Wrappers;

	UDiscreteObserver(){};

	/**
	 * @brief Get the DiscreteSpace bounding the outputs of this sensor.
	 * @return The ObservationSpace bounding the outputs of this sensor.
	 * @note This function should be implemented by any derived classes
	 */
	virtual FDiscreteSpace GetObservationSpace() const PURE_VIRTUAL(UDiscreteObserver::GetObservationSpace, return FDiscreteSpace(););

	/**
	 * @brief Use this sensor to collect observations about the environment state
	 * @param[out] OutObservations DataPoint that will be updated with the outputs of this sensor.
	 * @note This function should be implemented by any derived classes
	 */
	virtual void CollectObservations(FDiscretePoint& OutObservations) PURE_VIRTUAL(UDiscreteObserver::CollectObservations, return; );

	void CollectObservations(TPoint& OutObservations)
	{
		FDiscretePoint OutputPoint;
		this->CollectObservations(OutputPoint);
		for (TScriptInterface<IDiscreteObserverWrapper> Wrapper : this->Wrappers)
		{
			OutputPoint = Wrapper->WrapDiscreteObservation(OutputPoint);
		}
		OutObservations.Set<FDiscretePoint>(OutputPoint);
		#if WITH_EDITOR
				this->SetDebugObservations(OutObservations);
		#endif
	}

	void FillObservationSpace(TSpace& OutSpaceGroup) const
	{
		FDiscreteSpace OutputSpace = this->GetObservationSpace();

		for (TScriptInterface<IDiscreteObserverWrapper> Wrapper : this->Wrappers)
		{
			OutputSpace = Wrapper->WrapDiscreteObservationSpace(OutputSpace);
		}

		OutSpaceGroup.Set<FDiscreteSpace>(this->GetObservationSpace());
		
	}

	FString GetId() const;

#if WITH_EDITOR
	void SetDebugObservations(TPoint& Temp) override;
#endif

#if WITH_EDITORONLY_DATA
	/** The debug observations for this observer. Shows the last collected observation  */
	UPROPERTY(VisibleInstanceOnly, Category = "Observer Utilities")
	TArray<int> DebugDiscretePoint;
#endif
};

/**
 * @brief Blueprintable version of UDiscreteObserver
 * @note This class is abstract and should be derived from to implement the CollectObservations and GetObservationSpace functions.
 */
UCLASS(Blueprintable, Abstract)
class SCHOLA_API UBlueprintDiscreteObserver : public UDiscreteObserver
{
	GENERATED_BODY()
public:
	UBlueprintDiscreteObserver(){};

	UFUNCTION(BlueprintImplementableEvent)
	FDiscreteSpace GetObservationSpace() const;

	UFUNCTION(BlueprintImplementableEvent)
	void CollectObservations(FDiscretePoint& OutObservations);

	UFUNCTION(BlueprintImplementableEvent)
	void InitializeObserver();
};
