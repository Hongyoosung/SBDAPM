// FollowerAgentTrainer.cpp - Trainer wrapper implementation

#include "Schola/FollowerAgentTrainer.h"
#include "Schola/ScholaAgentComponent.h"
#include "Schola/TacticalRewardProvider.h"
#include "Team/FollowerAgentComponent.h"
#include "Combat/HealthComponent.h"

AFollowerAgentTrainer::AFollowerAgentTrainer()
{
	PrimaryActorTick.bCanEverTick = false;
}

void AFollowerAgentTrainer::Initialize(UScholaAgentComponent* InAgent)
{
	if (!InAgent)
	{
		UE_LOG(LogTemp, Error, TEXT("[FollowerTrainer] Initialize called with null agent"));
		return;
	}

	ScholaAgent = InAgent;
	FollowerAgent = InAgent->FollowerAgent;
	RewardProvider = InAgent->RewardProvider;

	// Copy observers and actuators from ScholaAgentComponent to this trainer
	// This is required by Schola's architecture
	Observers = ScholaAgent->Observers;
	Actuators = ScholaAgent->Actuators;

	// Use ScholaAgent's InteractionManager
	InteractionManager = ScholaAgent->InteractionManager;

	// Set trainer configuration
	TrainerConfiguration.DecisionFrequency = 1; // Every step (real-time RL)
	TrainerConfiguration.TrainerName = FString::Printf(TEXT("Follower_%s"), *InAgent->GetOwner()->GetName());

	UE_LOG(LogTemp, Log, TEXT("[FollowerTrainer] Initialized for %s"), *InAgent->GetOwner()->GetName());
}

//------------------------------------------------------------------------------
// ABSTRACT TRAINER INTERFACE
//------------------------------------------------------------------------------

float AFollowerAgentTrainer::ComputeReward()
{
	if (!RewardProvider)
	{
		return 0.0f;
	}

	// Get reward from TacticalRewardProvider
	float StepReward = RewardProvider->GetReward();
	EpisodeReward += StepReward;

	return StepReward;
}

EAgentTrainingStatus AFollowerAgentTrainer::ComputeStatus()
{
	// Check if agent is dead
	if (IsAgentDead())
	{
		UE_LOG(LogTemp, Log, TEXT("[FollowerTrainer] %s - Agent died (Episode reward: %.2f, Steps: %d)"),
			*TrainerConfiguration.TrainerName, EpisodeReward, EpisodeSteps);
		return EAgentTrainingStatus::Complete;
	}

	// Check if reward provider says episode terminated
	if (RewardProvider && RewardProvider->IsTerminated())
	{
		UE_LOG(LogTemp, Log, TEXT("[FollowerTrainer] %s - Episode terminated by RewardProvider"),
			*TrainerConfiguration.TrainerName);
		return EAgentTrainingStatus::Complete;
	}

	// Check for timeout
	if (IsEpisodeTimeout())
	{
		UE_LOG(LogTemp, Log, TEXT("[FollowerTrainer] %s - Episode timeout"),
			*TrainerConfiguration.TrainerName);
		return EAgentTrainingStatus::Truncated;
	}

	// Still running
	return EAgentTrainingStatus::Running;
}

void AFollowerAgentTrainer::GetInfo(TMap<FString, FString>& Info)
{
	// Provide debug info for logging/monitoring
	Info.Add(TEXT("agent_name"), TrainerConfiguration.TrainerName);
	Info.Add(TEXT("episode_reward"), FString::SanitizeFloat(EpisodeReward));
	Info.Add(TEXT("episode_steps"), FString::FromInt(EpisodeSteps));

	if (FollowerAgent)
	{
		Info.Add(TEXT("is_alive"), FollowerAgent->bIsAlive ? TEXT("true") : TEXT("false"));
	}

	if (RewardProvider)
	{
		Info.Add(TEXT("current_reward"), FString::SanitizeFloat(RewardProvider->GetReward()));
	}
}

void AFollowerAgentTrainer::ResetTrainer()
{
	// Reset episode counters
	EpisodeReward = 0.0f;
	EpisodeSteps = 0;

	// Reset ScholaAgent episode state
	if (ScholaAgent)
	{
		ScholaAgent->ResetEpisode();
	}

	UE_LOG(LogTemp, Log, TEXT("[FollowerTrainer] %s - Reset for new episode"), *TrainerConfiguration.TrainerName);
}

void AFollowerAgentTrainer::OnCompletion()
{
	// Called when episode ends
	UE_LOG(LogTemp, Log, TEXT("[FollowerTrainer] %s - Episode completed (Total reward: %.2f, Steps: %d)"),
		*TrainerConfiguration.TrainerName, EpisodeReward, EpisodeSteps);
}

//------------------------------------------------------------------------------
// INTERNAL HELPERS
//------------------------------------------------------------------------------

bool AFollowerAgentTrainer::IsAgentDead() const
{
	if (!FollowerAgent)
	{
		return true;
	}

	// Check FollowerAgentComponent's alive flag
	if (!FollowerAgent->bIsAlive)
	{
		return true;
	}

	// Double-check with HealthComponent
	AActor* OwnerActor = FollowerAgent->GetOwner();
	if (OwnerActor)
	{
		UHealthComponent* HealthComp = OwnerActor->FindComponentByClass<UHealthComponent>();
		if (HealthComp && HealthComp->IsDead())
		{
			return true;
		}
	}

	return false;
}

bool AFollowerAgentTrainer::IsEpisodeTimeout() const
{
	// Check for max episode steps (configured in TrainerConfiguration)
	const int32 MaxSteps = 10000; // ~5 minutes at 30 FPS
	return EpisodeSteps >= MaxSteps;
}
