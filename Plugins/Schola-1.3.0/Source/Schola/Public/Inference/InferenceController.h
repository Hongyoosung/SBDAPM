// Copyright (c) 2023 Advanced Micro Devices, Inc. All Rights Reserved.

#pragma once

#include "CoreMinimal.h"
#include <Kismet/GameplayStatics.h>
#include "Common/LogSchola.h"
#include "./IInferenceAgent.h"
#include "InferenceController.generated.h"

/**
 * @brief A controller that implements the IInferenceAgent interface, to control a Pawn with a Brain/Policy.
 */
UCLASS(Abstract, Blueprintable, ClassGroup = (Schola), meta = (BlueprintSpawnableComponent))
class SCHOLA_API AInferenceController : public AController, public IInferenceAgent
{
	GENERATED_BODY()

public:
	/** Object defining how the agent interacts with the environment. */
	UPROPERTY(EditAnywhere, NoClear, Instanced, meta = (ShowInnerProperties), Category = "Reinforcement Learning")
	UInteractionManager* InteractionManager = CreateDefaultSubobject<UInteractionManager>(TEXT("InteractionManager"));

	/** Object defining an asynchronous function f:Observations->Actions used to make decisions for the agent. */
	UPROPERTY(EditAnywhere, NoClear, Instanced, meta = (ShowInnerProperties), Category = "Reinforcement Learning")
	UAbstractPolicy* Policy;

	/** Object defining how decision requests are synchronized. */
	UPROPERTY(EditAnywhere, NoClear, Instanced, meta = (ShowInnerProperties), Category = "Reinforcement Learning")
	UAbstractBrain* Brain;

	/** List of observers that collect observations for the agent. */
	UPROPERTY(EditAnywhere, NoClear, Instanced, meta = (ShowInnerProperties), Category = "Reinforcement Learning")
	TArray<UAbstractObserver*> Observers;

	/** List of actuators that execute actions for the agent. */
	UPROPERTY(EditAnywhere, NoClear, Instanced, meta = (ShowInnerProperties), Category = "Reinforcement Learning")
	TArray<UActuator*> Actuators;

	/** The status of the agent. */
	UPROPERTY(BlueprintReadOnly, Category = "Reinforcement Learning")
	EAgentStatus Status = EAgentStatus::Stopped;

	/* Tick function for Think calls. */
	UPROPERTY()
	FThinkTickFunction ThinkTickFunction = FThinkTickFunction(this);

	/* Tick function for Act calls. */
	UPROPERTY()
	FActTickFunction ActTickFunction = FActTickFunction(this);

	/** Whether the agent should be set up to take actions automatically. */
	UPROPERTY(EditAnywhere, Category = "Reinforcement Learning")
	bool bRegisterAgentStep = true;

	/**
	 * @brief Get the controlled pawn of the agent.
	 * @return A pointer to the controlled pawn.
	 */
	virtual APawn* GetControlledPawn() override
	{
		return this->GetPawn();
	}

	/**
	 * @brief Get the interaction manager for the agent.
	 * @return A pointer to the interaction manager.
	 */
	virtual UInteractionManager* GetInteractionManager() override
	{
		return InteractionManager;
	}

	/**
	 * @brief Get the brain of the agent.
	 * @return A pointer to the brain.
	 */
	virtual UAbstractBrain* GetBrain() override
	{
		return Brain;
	}

	/**
	 * @brief Get the policy of the agent.
	 * @return A pointer to the policy.
	 */
	virtual UAbstractPolicy* GetPolicy() override
	{
		return Policy;
	}

	/**
	 * @brief Get all observers of the agent.
	 * @return An array of pointers to the observers.
	 */
	virtual TArray<UAbstractObserver*> GetAllObservers() override
	{
		TArray<UAbstractObserver*> AllObservers;
		AllObservers.Append(this->Observers);
		AllObservers.Append(GetObserversFromPawn());

		TArray<USensor*> SensorsTemp;
		this->GetComponents(SensorsTemp);
		for (USensor* Sensor : SensorsTemp)
		{
			AllObservers.Add(Sensor->Observer);
		}

		return AllObservers;
	}

	/**
	 * @brief Get all actuators of the agent.
	 * @return An array of pointers to the actuators.
	 */
	virtual TArray<UActuator*> GetAllActuators() override
	{

		TArray<UActuator*> AllActuators;
		AllActuators.Append(Actuators);
		AllActuators.Append(GetActuatorsFromPawn());

		TArray<UActuatorComponent*> ActuatorsTemp;
		this->GetComponents(ActuatorsTemp);
		for (UActuatorComponent* Actuator : ActuatorsTemp)
		{
			AllActuators.Add(Actuator->Actuator);
		}

		return AllActuators;
	}

	/**
	 * @brief Get the status of the agent.
	 * @return The status of the agent.
	 */
	virtual EAgentStatus GetStatus() override
	{
		return Status;
	}

	/**
	 * @brief Set the status of the agent.
	 * @param[in] NewStatus The new status to set.
	 */
	virtual void SetStatus(EAgentStatus NewStatus) override
	{
		Status = NewStatus;
	}

	/**
	 * @brief Register or unregister the tick functions for the agent.
	 * @param[in] bRegister Whether to register the tick functions.
	 */
	void RegisterActorTickFunctions(bool bRegister) override
	{
		Super::RegisterActorTickFunctions(bRegister);
		if (bRegister && bRegisterAgentStep)
		{
			//Pass in this as the target actor since the object can exist outside of the controlled actor
			this->SetupDefaultTicking(this->ThinkTickFunction, this->ActTickFunction,this);
		}
		else
		{
			this->ThinkTickFunction.UnRegisterTickFunction();
			this->ActTickFunction.UnRegisterTickFunction();
		}
	}

	/**
	 * @brief Called when the game starts or when spawned.
	 */
	virtual void BeginPlay() override {
		Super::BeginPlay();
		this->Initialize();
	}
};