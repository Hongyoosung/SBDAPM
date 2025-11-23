// Copyright (c) 2023 Advanced Micro Devices, Inc. All Rights Reserved.

#include "Training/AbstractTrainer.h"
#include "Subsystem/ScholaManagerSubsystem.h"
#include "Agent/AgentComponents/SensorComponent.h"

const FString AGENT_ACTION_ID = FString("__AGENT__");

void AAbstractTrainer::OnPossess(APawn* InPawn)
{
	Super::OnPossess(InPawn);
	if (this->HasAgentClass())
	{
		// If we have attached before, check if the class is the same as the attached pawn
		if (this->AgentClass != InPawn->GetClass())
		{
			UE_LOG(LogSchola, Warning, TEXT("Agent %s has attached to a different pawn class than before. This may cause issues."), *this->GetName())
		}
	}
	else //We haven't attached before, so set the class to the attached pawn
	{
		this->AgentClass = InPawn->GetClass();
	}
	this->State.bExists = true;
	this->SetTrainingStatus(EAgentTrainingStatus::Running);
}


void AAbstractTrainer::PawnPendingDestroy(APawn* inPawn)
{
	//Code is based on AActor::PawnPendingDestroy but we don't destroy the trainer at the end
	if (IsInState(NAME_Inactive))
	{
		UE_LOG(LogController, Log, TEXT("PawnPendingDestroy while inactive %s"), *GetName());
	}

	if (inPawn != this->GetPawn())
	{
		return;
	}

	UnPossess();
	ChangeState(NAME_Inactive);
	
	//Normally we would destroy ourselves here, but Trainers persist beyond their pawns
}

void AAbstractTrainer::OnUnPossess()
{
	//We directly set here, so that in the event of UnPossess->Possess in one tick we can continue.
	Super::OnUnPossess();
	this->State.bExists = false;
}

AAbstractTrainer::AAbstractTrainer()
{
	
}

bool AAbstractTrainer::Initialize(int EnvId, int AgentId, APawn* TargetPawn)
{
	UE_LOG(LogSchola, Log, TEXT("Starting Initialization of Agent %s"), *this->GetName())

	this->AgentClass = TargetPawn->GetClass();

	//UE_LOG(LogSchola, Log, TEXT("Agent is Controlling Pawn %s "), *this->TargetPawn->GetName());

	// Collect all the observers and actuators
	TArray<UActuatorComponent*> ActuatorComponentsTemp;
	TArray<UActuator*>			AllActuators = this->Actuators;
	this->GetPawn()->GetComponents(ActuatorComponentsTemp);
	for (UActuatorComponent* Actuator : ActuatorComponentsTemp)
	{
		AllActuators.Add(Actuator->Actuator);
	}
	ActuatorComponentsTemp.Reset();
	this->GetComponents(ActuatorComponentsTemp);
	for (UActuatorComponent* Actuator : ActuatorComponentsTemp)
	{
		AllActuators.Add(Actuator->Actuator);
	}

	TArray<USensor*> SensorsTemp;
	TArray<UAbstractObserver*> AllObservers = this->Observers;
	this->GetPawn()->GetComponents(SensorsTemp);
	for (USensor* Sensor : SensorsTemp)
	{
		AllObservers.Add(Sensor->Observer);
	}
	SensorsTemp.Reset();
	this->GetComponents(SensorsTemp);
	for (USensor* Sensor : SensorsTemp)
	{
		AllObservers.Add(Sensor->Observer);
	}

	// Initialize the Interaction Manager with the Observers and Actuators
	this->InteractionManager->Initialize(AllObservers, AllActuators);

	// Set the agent state's observation as a pointer to the Interaction Manager's observation
	this->State.Observations = &this->InteractionManager->Observations;
	// Set the ID for the Agent
	UAgentUIDSubsystem* UIDManager = GetWorld()->GetSubsystem<UAgentUIDSubsystem>();
	this->TrainerDefn.Id = {UIDManager->GetId(), EnvId, AgentId};
	
	if (this->TrainerConfiguration.bUseCustomName)
	{
		this->TrainerDefn.Name = this->TrainerConfiguration.Name;
	}
	else
	{
		this->GetName(this->TrainerDefn.Name);
	}

	this->TrainerDefn.PolicyDefinition = &this->InteractionManager->InteractionDefn;

	UE_LOG(LogSchola, Warning, TEXT("Initialization finished"));
	return true;
}

FTrainerState* AAbstractTrainer::Think()
{

	TRACE_CPUPROFILER_EVENT_SCOPE_STR("Schola: Agent Thinking");
	// If we entered the think done, we can skip
	if (this->IsDone())
	{
		// Set the training status so that, LastStatus now shows a terminal status as well
		this->SetTrainingStatus(this->State.TrainingStatus);
	}
	else
	{
		if(this->State.bExists)
		{
			this->SetTrainingStatus(this->ComputeStatus());
			// Set the reward.
			this->State.Reward = this->ComputeReward();
			// Update the info field
			this->State.Info.Reset();
			this->GetInfo(this->State.Info);
			this->InteractionManager->AggregateObservations();
		}
		else 
		{
			//We entered this think and the agent no longer exists, so reuse the last obs/reward and complete
			this->SetTrainingStatus(EAgentTrainingStatus::Completed);
		}
		
		if (this->IsDone())
		{
			OnCompletion();
		}
	}
	
	return &State;
}

void AAbstractTrainer::Act(const FAction& Action)
{
	TRACE_CPUPROFILER_EVENT_SCOPE_STR("Schola: Agent Acting");
	this->InteractionManager->DistributeActions(Action.Values);
	this->IncrementStep();
}

void AAbstractTrainer::Reset()
{
	this->ResetTrainer();
	State.Observations->Reset();
	State.Info.Reset();
	State.Step = 0;
	this->InteractionManager->Reset();
	this->InteractionManager->AggregateObservations();
	this->GetInfo(this->State.Info);
	this->SetTrainingStatus(EAgentTrainingStatus::Running);
	
}

void AAbstractTrainer::SetTrainingStatus(EAgentTrainingStatus NewStatus)
{
	this->State.LastStatus = this->State.TrainingStatus;
	this->State.TrainingStatus = NewStatus;
}

EAgentTrainingStatus AAbstractTrainer::GetTrainingStatus()
{
	return this->State.TrainingStatus;
}

bool AAbstractTrainer::IsRunning()
{
	return this->State.TrainingStatus == EAgentTrainingStatus::Running;
}

bool AAbstractTrainer::IsDone() const
{
	return this->State.IsDone();
}

bool AAbstractTrainer::IsActionStep()
{
	return this->IsDecisionStep() || this->TrainerConfiguration.bTakeActionBetweenDecisions;
}

bool AAbstractTrainer::IsDecisionStep(int StepToCheck)
{
	return (StepToCheck % this->TrainerConfiguration.DecisionRequestFrequency) == 0;
}

bool AAbstractTrainer::IsDecisionStep()
{
	return this->IsDecisionStep(this->State.Step);
}
