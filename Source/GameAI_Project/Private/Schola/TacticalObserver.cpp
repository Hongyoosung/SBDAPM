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

	// CRITICAL: Add safety checks to prevent crash during initialization
	if (!FollowerAgent || !FollowerAgent->IsValidLowLevel() || !FollowerAgent->GetOwner())
	{
		if (CallCount % 100 == 1) // Log every 100th call to avoid spam
		{
			UE_LOG(LogTemp, Error, TEXT("[TacticalObserver] CollectObservations called but FollowerAgent is NULL or invalid! (Call #%d)"), CallCount);
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

	// Add 7-dimensional objective embedding (one-hot encoding of current objective type)
	// Objective types mapped to embedding indices (v3.0 fix):
	// Eliminate=0, CaptureObjective=1, DefendObjective=2, SupportAlly=3,
	// FormationMove=4, Retreat=5, RescueAlly=6
	// None = all zeros
	for (int32 i = 0; i < 7; ++i)
	{
		OutObservations.Values[71 + i] = 0.0f;
	}

	// Get current objective from follower and encode as one-hot
	UObjective* CurrentObjective = FollowerAgent->GetCurrentObjective();
	if (CurrentObjective && CurrentObjective->IsActive())
	{
		// EObjectiveType enum: None=0, Eliminate=1, CaptureObjective=2, DefendObjective=3,
		// SupportAlly=4, FormationMove=5, Retreat=6, RescueAlly=7
		int32 ObjectiveIndex = static_cast<int32>(CurrentObjective->Type);

		// Map to 0-6 range (skip None=0 by subtracting 1)
		if (ObjectiveIndex >= 1 && ObjectiveIndex <= 7)
		{
			OutObservations.Values[71 + (ObjectiveIndex - 1)] = 1.0f;
		}
		// If None (0) or invalid: keep all zeros (already initialized)
	}
	// If no objective or inactive: keep all zeros (already initialized)
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
