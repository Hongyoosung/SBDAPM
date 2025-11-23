// Copyright (c) 2023-2025 Advanced Micro Devices, Inc. All Rights Reserved.

#pragma once
#include "Environment/AbstractEnvironment.h"
#include "DynamicEnvironment.generated.h"

USTRUCT(BlueprintType)
struct SCHOLA_API FDynamicAgentStruct
{
	GENERATED_BODY()

	UPROPERTY(BlueprintReadWrite, EditAnywhere, Category = "Dynamic Agent Struct")
	TSubclassOf<APawn> AgentClass;

	UPROPERTY(BlueprintReadWrite, EditAnywhere, Category = "Dynamic Agent Struct")
	TSubclassOf<AAbstractTrainer> TrainerClass;

	FDynamicAgentStruct(TSubclassOf<APawn> AgentClass, TSubclassOf<AAbstractTrainer> TrainerClass)
	{
		this->AgentClass = AgentClass;
		this->TrainerClass = TrainerClass;
	};

	FDynamicAgentStruct()
	{
		this->AgentClass = nullptr;
		this->TrainerClass = nullptr;
	};
};



UCLASS(Abstract)
class ADynamicScholaEnvironment : public AAbstractScholaEnvironment
{
	GENERATED_BODY()

public:


	virtual void InternalRegisterAgents(TArray<FTrainerAgentPair>& OutAgentTrainerPairs);
	

	virtual void RegisterAgents(TArray<FDynamicAgentStruct>& OutAgentSpawnStructs){};

	UFUNCTION(BlueprintCallable, Category = "Reinforcement Learning")
	APawn* SpawnAgent(int AgentId, FTransform InAgentPosition);

};

/**
 * @brief A blueprintable version of the DynamicScholaEnvironment, that features trainers spawned by the environment or other objects
 */
UCLASS(Blueprintable,Abstract)
class SCHOLA_API ABlueprintDynamicScholaEnvironment : public ADynamicScholaEnvironment
{
	GENERATED_BODY()

public:
	UFUNCTION(BlueprintImplementableEvent, Category = "Reinforcement Learning")
	void ResetEnvironment();

	UFUNCTION(BlueprintImplementableEvent, Category = "Reinforcement Learning")
	void RegisterAgents(TArray<FDynamicAgentStruct>& OutAgentSpawnStructs);

	UFUNCTION(BlueprintImplementableEvent, Category = "Reinforcement Learning")
	void InitializeEnvironment();

	UFUNCTION(BlueprintImplementableEvent, Category = "Reinforcement Learning")
	void SetEnvironmentOptions(const TMap<FString, FString>& InOptions);

	UFUNCTION(BlueprintImplementableEvent, Category = "Reinforcement Learning")
	void SeedEnvironment(int Seed);
};