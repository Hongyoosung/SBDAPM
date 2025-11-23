// Copyright (c) 2024 Advanced Micro Devices, Inc. All Rights Reserved.

#pragma once

#include "CoreMinimal.h"
#include "AgentId.generated.h"

/**
 * @brief Struct containing the Id of an agent.
 */
USTRUCT(BlueprintType)
struct SCHOLA_API FAgentId
{
	GENERATED_BODY()
	
	/* The unique identifier of this agent. */
	UPROPERTY(EditAnywhere, Category="Agent ID")
	int UniqueId;

	/** The unique identifier of the agent with respect to it's environment */
	UPROPERTY(EditAnywhere, Category="Agent ID")
	int AgentId;

	/** The unique identifier of the environment the agent is in. */
	UPROPERTY(EditAnywhere, Category="Agent ID")
	int EnvId;

	/**
	 * @brief Construct a new default FAgentId object
	 */
	FAgentId()
	{	
		UniqueId = -1;
		AgentId = 0;
		EnvId = 0;
	}

	/**
	 * @brief Construct a new FAgentId object
	 * @param[in] UniqueId The unique identifier
	 * @param[in] AgentId The unique identifier of the agent
	 * @param[in] EnvId The unique identifier of the environment the agent is in
	 */
	FAgentId(int UniqueId, int EnvId,  int AgentId )
	{	
		this->UniqueId = UniqueId;
		this->EnvId = EnvId;
		this->AgentId = AgentId;
	}

	/**
	 * @brief Copy construct a new FAgentId object
	 * @param Other An existing FAgentId object to copy
	 */
	FAgentId(const FAgentId& Other)
	{	
		this->UniqueId = Other.UniqueId;
		this->AgentId = Other.AgentId;
		this->EnvId = Other.EnvId;
	}
};

