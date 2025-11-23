// TacticalRewardProvider.cpp - Schola reward provider for combat events

#include "Schola/TacticalRewardProvider.h"
#include "Team/FollowerAgentComponent.h"

UTacticalRewardProvider::UTacticalRewardProvider()
{
}

void UTacticalRewardProvider::Initialize()
{
	if (bAutoFindFollower && FollowerAgent == nullptr)
	{
		FollowerAgent = FindFollowerAgent();
	}

	if (FollowerAgent)
	{
		LastRewardValue = FollowerAgent->GetAccumulatedReward();
		bTerminated = !FollowerAgent->bIsAlive;

		UE_LOG(LogTemp, Log, TEXT("[TacticalRewardProvider] Initialized with FollowerAgent"));
	}
	else
	{
		UE_LOG(LogTemp, Warning, TEXT("[TacticalRewardProvider] No FollowerAgent found"));
	}
}

float UTacticalRewardProvider::GetReward()
{
	if (!FollowerAgent)
	{
		return 0.0f;
	}

	// Get current accumulated reward
	float CurrentReward = FollowerAgent->GetAccumulatedReward();

	// Calculate delta since last query
	float DeltaReward = CurrentReward - LastRewardValue;
	LastRewardValue = CurrentReward;

	// Check for termination
	bTerminated = !FollowerAgent->bIsAlive;

	return DeltaReward;
}

void UTacticalRewardProvider::Reset()
{
	LastRewardValue = 0.0f;
	bTerminated = false;

	if (FollowerAgent)
	{
		FollowerAgent->ResetEpisode();
		LastRewardValue = FollowerAgent->GetAccumulatedReward();
	}
}

UFollowerAgentComponent* UTacticalRewardProvider::FindFollowerAgent() const
{
	AActor* Owner = Cast<AActor>(GetOuter());
	if (!Owner)
	{
		UActorComponent* OuterComponent = Cast<UActorComponent>(GetOuter());
		if (OuterComponent)
		{
			Owner = OuterComponent->GetOwner();
		}
	}

	if (Owner)
	{
		return Owner->FindComponentByClass<UFollowerAgentComponent>();
	}

	return nullptr;
}
