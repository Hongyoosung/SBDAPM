// Copyright (c) 2025 Advanced Micro Devices, Inc. All Rights Reserved.

#pragma once

#include "CoreMinimal.h"
#include "BehaviorTree/BTTaskNode.h"
#include "BehaviorTree/ValueOrBBKey.h"
#include "BehaviorTree/BehaviorTreeComponent.h"
#include "Inference/IInferenceAgent.h"
#include "AIController.h"
#include "Common/LogSchola.h"
#include "Common/InteractionManager.h"

#include "BTT_AgentStep.generated.h"

/**
 * @brief A BTTask that runs a step of inference to determine the action taken by the model.
 */
UCLASS()
class SCHOLA_API UBTTask_AgentStep : public UBTTaskNode, public IInferenceAgent
{
	GENERATED_BODY()

	EBTNodeResult::Type ExecuteTask(UBehaviorTreeComponent& OwnerComp, uint8* NodeMemory);

public:
	/** Task Constructor */
	UBTTask_AgentStep(FObjectInitializer const& ObjectInitializer);

	/** Object defining how the agent interacts with the environment. */
	UPROPERTY(EditAnywhere, NoClear, Instanced, meta = (ShowInnerProperties), Category = "Reinforcement Learning")
	UInteractionManager* InteractionManager = CreateDefaultSubobject<UInteractionManager>(TEXT("InteractionManager"));

	/** Object defining an asynchronous function f:Observations->Actions used to make decisions for the agent. */
	UPROPERTY(EditAnywhere, NoClear, Instanced, meta = (ShowInnerProperties), Category = "Reinforcement Learning")
	UAbstractPolicy* Policy;

	/** Object defining how decisions requests are synchronized. */
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
	EAgentStatus Status = EAgentStatus::Running;

	/** Whether the agent should be setup to take actions automatically. */
	UPROPERTY(EditAnywhere, Category = "Reinforcement Learning")
	bool bRegisterAgentStep = true;

	/* Tick Function for think calls. */
	UPROPERTY()
	FThinkTickFunction ThinkTickFunction = FThinkTickFunction(this);

	/* Tick Function for act calls. */
	UPROPERTY()
	FActTickFunction ActTickFunction = FActTickFunction(this, true);

	/** Whether to tick the task instead of executing it. 	 */
	UPROPERTY(EditAnywhere, Category = "Reinforcement Learning")
	bool bUseTickTask = false;

	/**
	 * @brief Get the controlled pawn of the agent.
	 * @return A pointer to the controlled pawn.
	 */
	virtual APawn* GetControlledPawn() override
	{
		return OwnerPawn;
	}

	virtual UInteractionManager* GetInteractionManager() override
	{
		return InteractionManager;
	}

	virtual UAbstractBrain* GetBrain() override
	{
		return Brain;
	}

	virtual UAbstractPolicy* GetPolicy() override
	{
		return Policy;
	}

	virtual TArray<UAbstractObserver*> GetAllObservers() override
	{
		TArray<UAbstractObserver*> AllObservers;
		AllObservers.Append(GetObserversFromPawn());
		AllObservers.Append(Observers);
		return AllObservers;
	}

	virtual TArray<UActuator*> GetAllActuators() override
	{
		TArray<UActuator*> AllActuators;
		AllActuators.Append(GetActuatorsFromPawn());
		AllActuators.Append(Actuators);
		return AllActuators;
	}

	virtual EAgentStatus GetStatus() override
	{
		return Status;
	}

	virtual void SetStatus(EAgentStatus NewStatus) override
	{
		Status = NewStatus;
	}

	virtual void TickTask(UBehaviorTreeComponent& OwnerComp, uint8* NodeMemory, float DeltaSeconds) override;

private:
	/** Whether the connection to owner pawn has been established */
	UPROPERTY(EditAnywhere, Category = "Reinforcement Learning")
	bool bConnected = false;

	/** The pawn that performs this task */
	UPROPERTY(VisibleAnywhere, Category = "Reinforcement Learning")
	APawn* OwnerPawn;

	void Step(UBehaviorTreeComponent& OwnerComp);

	EBTNodeResult::Type TaskResult = EBTNodeResult::InProgress;
};
