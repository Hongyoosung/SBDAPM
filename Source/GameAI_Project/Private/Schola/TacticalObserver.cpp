// TacticalObserver.cpp - Schola observer for 71-feature tactical observation

#include "Schola/TacticalObserver.h"
#include "Team/FollowerAgentComponent.h"
#include "Observation/ObservationElement.h"
#include "Common/Spaces/BoxSpace.h"
#include "Common/Points/BoxPoint.h"

UTacticalObserver::UTacticalObserver()
{
	// Build observation space (78 continuous features = 71 tactical + 7 objective embedding)
	TArray<FBoxSpaceDimension> Dimensions;
	Dimensions.Reserve(78);

	// All features normalized to [-1, 1] or [0, 1]
	for (int32 i = 0; i < 78; ++i)
	{
		FBoxSpaceDimension Dim;
		Dim.Low = -1.0f;
		Dim.High = 1.0f;
		Dimensions.Add(Dim);
	}

	CachedObservationSpace = FBoxSpace(Dimensions);
}

void UTacticalObserver::InitializeObserver()
{
	if (bAutoFindFollower && FollowerAgent == nullptr)
	{
		FollowerAgent = FindFollowerAgent();
	}

	if (FollowerAgent)
	{
		UE_LOG(LogTemp, Log, TEXT("[TacticalObserver] Initialized with FollowerAgent on %s"),
			*GetOuter()->GetName());
	}
	else
	{
		UE_LOG(LogTemp, Warning, TEXT("[TacticalObserver] No FollowerAgent found on %s"),
			*GetOuter()->GetName());
	}
}

void UTacticalObserver::ResetObserver()
{
	// Re-find follower if needed
	if (bAutoFindFollower && FollowerAgent == nullptr)
	{
		FollowerAgent = FindFollowerAgent();
	}
}

FBoxSpace UTacticalObserver::GetObservationSpace() const
{
	return CachedObservationSpace;
}

void UTacticalObserver::CollectObservations(FBoxPoint& OutObservations)
{
	static int32 CallCount = 0;
	CallCount++;

	OutObservations.Values.SetNum(78);

	if (!FollowerAgent)
	{
		if (CallCount % 100 == 1) // Log every 100th call to avoid spam
		{
			UE_LOG(LogTemp, Error, TEXT("[TacticalObserver] CollectObservations called but FollowerAgent is NULL! (Call #%d)"), CallCount);
		}
		// Return zeros if no follower
		for (int32 i = 0; i < 78; ++i)
		{
			OutObservations.Values[i] = 0.0f;
		}
		return;
	}

	if (CallCount % 100 == 1)
	{
		UE_LOG(LogTemp, Warning, TEXT("[TacticalObserver] CollectObservations #%d for %s"),
			CallCount, *FollowerAgent->GetOwner()->GetName());
	}

	// Get observation from follower and convert to feature vector (71 features)
	const FObservationElement& Obs = FollowerAgent->GetLocalObservation();
	TArray<float> Features = Obs.ToFeatureVector();

	// Copy all 71 tactical features
	check(Features.Num() == 71);
	for (int32 i = 0; i < 71; ++i)
	{
		OutObservations.Values[i] = Features[i];
	}

	// Add 7-dimensional objective embedding (one-hot encoding of current state)
	// States mapped to embedding indices:
	// Idle=0, Assault=1, Defend=2, Support=3, Move=4, Retreat=5, Dead=6
	for (int32 i = 0; i < 7; ++i)
	{
		OutObservations.Values[71 + i] = 0.0f;
	}

	// Get current state from follower and encode as one-hot
	EFollowerState CurrentState = FollowerAgent->GetCurrentState();
	int32 StateIndex = static_cast<int32>(CurrentState);

	// Ensure index is valid (Dead state should map to index 6)
	if (StateIndex >= 0 && StateIndex < 7)
	{
		OutObservations.Values[71 + StateIndex] = 1.0f;
	}
	else
	{
		// Fallback to Idle (index 0)
		OutObservations.Values[71] = 1.0f;
	}
}

UFollowerAgentComponent* UTacticalObserver::FindFollowerAgent() const
{
	AActor* Owner = Cast<AActor>(GetOuter());
	if (!Owner)
	{
		// Try to find through actor component hierarchy
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
