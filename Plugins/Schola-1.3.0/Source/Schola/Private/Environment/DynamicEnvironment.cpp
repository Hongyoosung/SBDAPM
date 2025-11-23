// Copyright (c) 2025 Advanced Micro Devices, Inc. All Rights Reserved.

#include "Environment/DynamicEnvironment.h"

void ADynamicScholaEnvironment::InternalRegisterAgents(TArray<FTrainerAgentPair>& OutAgentTrainerPairs)
{
	TArray<FDynamicAgentStruct> DynamicAgentSpawnStructs;
	RegisterAgents(DynamicAgentSpawnStructs);
	for (const FDynamicAgentStruct& AgentTrainerPair : DynamicAgentSpawnStructs)
	{
		AAbstractTrainer* Trainer = GetWorld()->SpawnActor<AAbstractTrainer>(AgentTrainerPair.TrainerClass);
		OutAgentTrainerPairs.Add({ AgentTrainerPair.AgentClass.GetDefaultObject(), Trainer });
	}
}

APawn* ADynamicScholaEnvironment::SpawnAgent(int AgentId, FTransform AgentPosition)
{
	AAbstractTrainer*	  Trainer = *Trainers.Find(AgentId);
	FActorSpawnParameters SpawnParams;
	APawn*				  Agent = GetWorld()->SpawnActor<APawn>(Trainer->AgentClass, AgentPosition);
	Trainer->Possess(Agent);
	return Agent;
}
