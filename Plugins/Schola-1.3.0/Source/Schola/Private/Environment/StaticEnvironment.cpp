// Copyright (c) 2025 Advanced Micro Devices, Inc. All Rights Reserved.

#include "Environment/StaticEnvironment.h"

AStaticScholaEnvironment::AStaticScholaEnvironment(const FObjectInitializer& ObjectInitializer)
	: Super(ObjectInitializer)
{

}

AStaticScholaEnvironment::~AStaticScholaEnvironment()
{
}

void AStaticScholaEnvironment::InternalRegisterAgents(TArray<FTrainerAgentPair>& OutAgentTrainerPairs)
{
	TArray<APawn*> TrainerControlledPawns;
	RegisterAgents(TrainerControlledPawns);
	for (APawn* ControlledPawn : TrainerControlledPawns)
	{
		AAbstractTrainer* Trainer = Cast<AAbstractTrainer>(ControlledPawn->GetController());
		if (Trainer == nullptr)
		{
			UE_LOG(LogSchola, Warning, TEXT("Pawn %s is not controlled by a Trainer. Skipping."), *ControlledPawn->GetName());
		}
		else
		{
			OutAgentTrainerPairs.Add({ ControlledPawn, Trainer });
		}
	}
}
