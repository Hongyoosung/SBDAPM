// TacticalActuator.cpp - Schola actuator for 16 discrete tactical actions

#include "Schola/TacticalActuator.h"
#include "Team/FollowerAgentComponent.h"
#include "StateTree/FollowerStateTreeComponent.h"
#include "Common/Spaces/DiscreteSpace.h"
#include "Common/Points/DiscretePoint.h"

UTacticalActuator::UTacticalActuator()
{
	// 16 discrete actions (ETacticalAction enum)
	TArray<int> High;
	High.Add(16);
	CachedActionSpace = FDiscreteSpace(High);
}

void UTacticalActuator::InitializeActuator()
{
	if (bAutoFindFollower && FollowerAgent == nullptr)
	{
		FollowerAgent = FindFollowerAgent();
	}

	if (FollowerAgent)
	{
		UE_LOG(LogTemp, Log, TEXT("[TacticalActuator] Initialized with FollowerAgent on %s"),
			*GetOuter()->GetName());
	}
	else
	{
		UE_LOG(LogTemp, Warning, TEXT("[TacticalActuator] No FollowerAgent found on %s"),
			*GetOuter()->GetName());
	}
}

void UTacticalActuator::ResetActuator()
{
	LastAction = ETacticalAction::DefensiveHold;

	// Re-find follower if needed
	if (bAutoFindFollower && FollowerAgent == nullptr)
	{
		FollowerAgent = FindFollowerAgent();
	}
}

FDiscreteSpace UTacticalActuator::GetActionSpace()
{
	return CachedActionSpace;
}

void UTacticalActuator::TakeAction(const FDiscretePoint& Action)
{
	if (Action.Values.Num() == 0)
	{
		UE_LOG(LogTemp, Warning, TEXT("[TacticalActuator] Empty action received"));
		return;
	}

	int32 ActionIndex = Action.Values[0];
	LastAction = MapActionToTactical(ActionIndex);

	if (!FollowerAgent)
	{
		UE_LOG(LogTemp, Warning, TEXT("[TacticalActuator] No FollowerAgent, action %d ignored"), ActionIndex);
		return;
	}

	// Update the follower's last tactical action
	FollowerAgent->LastTacticalAction = LastAction;

	// If the follower has a StateTree component, the action will be picked up
	// by STTask_QueryRLPolicy or similar task

	UE_LOG(LogTemp, Verbose, TEXT("[TacticalActuator] Action: %d -> %s"),
		ActionIndex, *UEnum::GetValueAsString(LastAction));
}

UFollowerAgentComponent* UTacticalActuator::FindFollowerAgent() const
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

ETacticalAction UTacticalActuator::MapActionToTactical(int32 ActionIndex) const
{
	// Direct mapping to ETacticalAction enum values
	switch (ActionIndex)
	{
		case 0:  return ETacticalAction::Advance;
		case 1:  return ETacticalAction::Retreat;
		case 2:  return ETacticalAction::FlankLeft;
		case 3:  return ETacticalAction::FlankRight;
		case 4:  return ETacticalAction::TakeCover;
		case 5:  return ETacticalAction::SuppressiveFire;
		case 6:  return ETacticalAction::HoldPosition;
		case 7:  return ETacticalAction::AssaultTarget;
		case 8:  return ETacticalAction::DefendPosition;
		case 9:  return ETacticalAction::Regroup;
		case 10: return ETacticalAction::Heal;
		case 11: return ETacticalAction::Reload;
		case 12: return ETacticalAction::CallSupport;
		case 13: return ETacticalAction::SetAmbush;
		case 14: return ETacticalAction::AggressivePush;
		case 15: return ETacticalAction::DefensiveHold;
		default:
			UE_LOG(LogTemp, Warning, TEXT("[TacticalActuator] Invalid action index %d, defaulting to DefensiveHold"), ActionIndex);
			return ETacticalAction::DefensiveHold;
	}
}
