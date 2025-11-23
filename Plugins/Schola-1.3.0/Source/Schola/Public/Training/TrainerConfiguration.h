// Copyright (c) 2025 Advanced Micro Devices, Inc. All Rights Reserved.

#pragma once
#include "CoreMinimal.h"
#include "Brains/AbstractBrain.h"
#include "Common/InteractionManager.h"
#include "TrainerConfiguration.generated.h"

/**
 * @brief Struct containing the properties that define a trainer.
 */
USTRUCT(BlueprintType)
struct SCHOLA_API FTrainerConfiguration 
{
	GENERATED_BODY()


    /** If true the agent will repeat its last action each step between decision requests */
	UPROPERTY(EditAnywhere, Category = "Reinforcement Learning")
	bool bTakeActionBetweenDecisions = true;

    /** The name of the agent, used for logging, and grouping agents in rllib */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Meta = (InlineEditConditionToggle), Category = "Reinforcement Learning")
	bool bUseCustomName = false;

	/** The name of the agent, used for logging, and grouping agents in rllib */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, meta = (EditCondition = "bUseCustomName"), Category = "Reinforcement Learning")
	FString Name;

    /** The number of steps between requests for new actions. If this is different across agents it may cause issues training in some frameworks (e.g. Stable Baselines 3). */
	UPROPERTY(EditAnywhere, Category = "Reinforcement Learning")
	int DecisionRequestFrequency = 5;

    /** The type of validation to perform on this agent. Fail means agent is skipped on any error. Warning means just warn about non-fatal errors. Set to No Validation to skip validation */
	UPROPERTY(EditAnywhere, Category = "Reinforcement Learning")
	EValidationType Validation = EValidationType::FAIL;
};
