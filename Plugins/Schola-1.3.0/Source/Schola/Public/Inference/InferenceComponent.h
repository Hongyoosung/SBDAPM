// Copyright (c) 2023 Advanced Micro Devices, Inc. All Rights Reserved.

#pragma once

#include "CoreMinimal.h"
#include <Kismet/GameplayStatics.h>
#include <Kismet/GameplayStatics.h>
#include "Common/LogSchola.h"
#include "./IInferenceAgent.h"
#include "InferenceComponent.generated.h"

/**
 * @brief A component that implements the IInferenceAgent interface, to control a Pawn with a Brain/Policy.
 */
UCLASS(Abstract, Blueprintable, ClassGroup = Schola)
class SCHOLA_API UInferenceComponent : public UActorComponent, public IInferenceAgent
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
	UPROPERTY(BlueprintReadOnly, VisibleAnywhere, Category = "Reinforcement Learning")
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
		return Cast<APawn>(this->GetOwner());
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
	 * @brief Get all the observers attached to the agent.
	 * @return An array of pointers to the observers.
	 */
	virtual TArray<UAbstractObserver*> GetAllObservers() override
	{
		TArray<UAbstractObserver*> AllObservers;
		AllObservers.Append(GetObserversFromPawn());
		AllObservers.Append(Observers);
		return AllObservers;
	}

	/**
	 * @brief Get all actuators of the agent.
	 * @return An array of pointers to the actuators.
	 */
	virtual TArray<UActuator*> GetAllActuators() override
	{
		TArray<UActuator*> AllActuators;
		AllActuators.Append(GetActuatorsFromPawn());
		AllActuators.Append(Actuators);
		// Log the number of actuators collected
		UE_LOG(LogSchola, Warning, TEXT("Number of Actuators: %d"), AllActuators.Num());
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
	void RegisterComponentTickFunctions(bool bRegister) override
	{
		UActorComponent::RegisterComponentTickFunctions(bRegister);
		if (bRegister && bRegisterAgentStep)
		{
			assert(this->GetControlledPawn() != nullptr);
			this->SetupDefaultTicking(ThinkTickFunction, ActTickFunction,this->GetControlledPawn());
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
	virtual void BeginPlay() override
	{
		Super::BeginPlay();
		this->Initialize();
	}
};