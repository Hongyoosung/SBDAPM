// Copyright (c) 2023 Advanced Micro Devices, Inc. All Rights Reserved.

#pragma once

#include "CoreMinimal.h"
#include "Components/ActorComponent.h"
#include "Agent/AgentAction.h"
#include "AbstractEnvironmentUtilityComponent.generated.h"

/**
 * @brief An Abstract Base class for an ActorComponent that provides utility functions for an environment.
 */
UCLASS(Abstract, EditInlineNew, ClassGroup = Schola)
class SCHOLA_API UAbstractEnvironmentUtilityComponent : public UActorComponent
{
	GENERATED_BODY()

public:
	/**
	 * @brief Callback for when an agent takes a step in the environment.
	 * @param[in] AgentID The ID of the agent that took the step.
	 * @param[in] State The state of the agent after taking the step.
	 */
	virtual void OnEnvironmentStep(int AgentID, FTrainerState& State){};

	/**
	 * @brief Callback for when the environment is reset.
	 */
	virtual void OnEnvironmentReset(){};

	/**
	 * @brief Callback for when an agent is registered in the environment.
	 * @param[in] AgentID The ID of the agent that was registered.
	 */
	virtual void OnAgentRegister(int AgentID){};

	/**
	 * @brief Callback for when the environment is initialized.
	 * @param[in] Id The ID of the environment.
	 */
	virtual void OnEnvironmentInit(int Id);

	/**
	 * @brief Callback for when the environment ends.
	 */
	virtual void OnEnvironmentEnd(){};

protected:
	/** The ID of the environment this component is attached to. */
	UPROPERTY()
	int EnvId;
};

/**
 * @brief A blueprint implementable version of the AbstractEnvironmentUtilityComponent.
 */
UCLASS(Blueprintable)
class SCHOLA_API UBlueprintEnvironmentUtilityComponent : public UAbstractEnvironmentUtilityComponent
{
	GENERATED_BODY()

public:

	UFUNCTION(BlueprintImplementableEvent, Category = "Reinforcement Learning")
	void OnEnvironmentStep(int AgentID, FTrainerState& State);

	UFUNCTION(BlueprintImplementableEvent, Category = "Reinforcement Learning")
	void OnEnvironmentReset();

	UFUNCTION(BlueprintImplementableEvent, Category = "Reinforcement Learning")
	void OnAgentRegister(int AgentID);

	UFUNCTION(BlueprintImplementableEvent, Category = "Reinforcement Learning")
	void OnEnvironmentInit(int Id);

	UFUNCTION(BlueprintImplementableEvent, Category = "Reinforcement Learning")
	void OnEnvironmentEnd();
};
