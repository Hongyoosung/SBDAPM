// Copyright (c) 2023-2025 Advanced Micro Devices, Inc. All Rights Reserved.

#include "Environment/AbstractEnvironment.h"

void AAbstractScholaEnvironment::Initialize()
{
	this->InitializeEnvironment();
	TArray<FTrainerAgentPair> TrainersTemp;
	InternalRegisterAgents(TrainersTemp);
	RetrieveUtilityComponents();

	//Spawn Trainers
	for (const FTrainerAgentPair& TrainerAgentPair : TrainersTemp)
	{
		if (TrainerAgentPair.Trainer->Initialize(this->EnvId, MaxId, TrainerAgentPair.AgentCDO))
		{
			Trainers.Add(MaxId, TrainerAgentPair.Trainer);
			for (UAbstractEnvironmentUtilityComponent* Component : UtilityComponents)
			{
				Component->OnAgentRegister(MaxId);
			}
			// Increment so it is always largest value+1, that way RegisterAgent never clobbers
			MaxId++;
		}
		else
		{
			UE_LOG(LogSchola, Warning, TEXT("Initializing Agent Failed in Environment %s. Skipping."), *this->GetName());
		}
	}

	for (UAbstractEnvironmentUtilityComponent* Component : UtilityComponents)
	{
		Component->OnEnvironmentInit(EnvId);
	}

	if (this->Trainers.Num() <= 0)
	{
		UE_LOG(LogSchola, Warning, TEXT("Environment %s has No Agents. Are you sure this is correct? See previous logs for potential errors while adding agents."), *this->GetName());
	}
}

void AAbstractScholaEnvironment::RetrieveUtilityComponents()
{
	this->GetComponents(UtilityComponents);
}

void AAbstractScholaEnvironment::PopulateAgentStatePointers(FSharedEnvironmentState& OutSharedEnvironmentState)
{
	for (TPair<int,AAbstractTrainer*>& IdAgentPair : this->Trainers)
	{
		OutSharedEnvironmentState.AddSharedAgentState(IdAgentPair.Key, IdAgentPair.Value);
	}
}

void AAbstractScholaEnvironment::PopulateAgentDefinitionPointers(FEnvironmentDefinition& OutEnvDefn)
{
	for (auto& IdAgentPair : this->Trainers)
	{
		// TODO make a method for getting the agent defn
		OutEnvDefn.AddSharedAgentDefn(IdAgentPair.Key, &IdAgentPair.Value->TrainerDefn);
	}
}

int AAbstractScholaEnvironment::GetNumAgents()
{
	return Trainers.Num();
}

FVector AAbstractScholaEnvironment::GetEnvironmentCenterPoint()
{
	return this->GetLevelTransform().GetLocation();
}

void AAbstractScholaEnvironment::Reset()
{
	
	ResetEnvironment();
	for (auto& IdAgentPair : Trainers)
	{
		AAbstractTrainer* Trainer = IdAgentPair.Value;
		Trainer->Reset();
	}

	for (UAbstractEnvironmentUtilityComponent* Component : UtilityComponents)
	{
		Component->OnEnvironmentReset();
	}


	// Make sure that the user is warned if there are no active agents after reset
	bool bAnyActiveAgents = false;
	for (auto& IdAgentPair : Trainers)
	{
		AAbstractTrainer* Trainer = IdAgentPair.Value;
		if (Trainer->State.bExists)
		{
			bAnyActiveAgents = true;
			break;
		}
	}
	
	if (!bAnyActiveAgents)
	{
		UE_LOG(LogSchola, Warning, TEXT("No Active Agents in Environment %s, after reset. Are you sure this is correct?"), *this->GetName());
		this->EnvironmentStatus = EEnvironmentStatus::Completed;
	}

}

void AAbstractScholaEnvironment::MarkCompleted()
{
	this->EnvironmentStatus = EEnvironmentStatus::Completed;
}

void AAbstractScholaEnvironment::AllAgentsThink()
{
	bool AllDone = true;

	for (auto& IdAgentPair : Trainers)
	{	
		FTrainerState* TrainerState = IdAgentPair.Value->Think();

		// Pass agent state to Utility Components for calculations.
		for (UAbstractEnvironmentUtilityComponent* Component : UtilityComponents)
		{
			Component->OnEnvironmentStep(IdAgentPair.Key, *TrainerState);
		}

		if (!TrainerState->IsDone())
		{
			AllDone = false;
		}
	}

	if (AllDone)
	{
		this->EnvironmentStatus = EEnvironmentStatus::Completed;
	}

	// If all agents are done, mark the environment as completed
}

void AAbstractScholaEnvironment::AllAgentsAct(const FEnvStep& EnvUpdate)
{
	for (const TTuple<int, FAction>& IdActionPair : EnvUpdate.Actions)
	{
		// We only act if we are running
		// Lookup is done on State, because that can outlast TrainerObjects
		if (State->AgentStates[IdActionPair.Key]->TrainingStatus == EAgentTrainingStatus::Running)
		{
			Trainers[IdActionPair.Key]->Act(IdActionPair.Value);
		}
	}
}

void AAbstractScholaEnvironment::SetEnvId(int EnvironmentId)
{
	this->EnvId = EnvironmentId;
}

void AAbstractScholaEnvironment::UpdateStatus(EEnvironmentStatus NewStatus)
{
	this->EnvironmentStatus = NewStatus;
}

EEnvironmentStatus AAbstractScholaEnvironment::GetStatus()
{
	return this->EnvironmentStatus;
}