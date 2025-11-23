// TacticalObserver.cpp - Schola observer for 71-feature tactical observation

#include "Schola/TacticalObserver.h"
#include "Team/FollowerAgentComponent.h"
#include "Observation/ObservationElement.h"
#include "Common/Spaces/BoxSpace.h"
#include "Common/Points/BoxPoint.h"

UTacticalObserver::UTacticalObserver()
{
	// Build observation space (71 continuous features)
	TArray<FBoxSpaceDimension> Dimensions;
	Dimensions.Reserve(71);

	// All features normalized to [-1, 1] or [0, 1]
	for (int32 i = 0; i < 71; ++i)
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
	OutObservations.Values.SetNum(71);

	if (!FollowerAgent)
	{
		// Return zeros if no follower
		for (int32 i = 0; i < 71; ++i)
		{
			OutObservations.Values[i] = 0.0f;
		}
		return;
	}

	// Get observation from follower and convert to feature vector
	const FObservationElement& Obs = FollowerAgent->GetLocalObservation();
	TArray<float> Features = Obs.ToFeatureVector();

	// Copy all 71 features
	check(Features.Num() == 71);
	OutObservations.Values = MoveTemp(Features);
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
