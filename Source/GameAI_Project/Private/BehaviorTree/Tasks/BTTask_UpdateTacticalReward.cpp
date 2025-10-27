// Copyright Epic Games, Inc. All Rights Reserved.

#include "BehaviorTree/Tasks/BTTask_UpdateTacticalReward.h"
#include "Team/FollowerAgentComponent.h"
#include "BehaviorTree/BlackboardComponent.h"
#include "BehaviorTree/BehaviorTreeComponent.h"
#include "AIController.h"

UBTTask_UpdateTacticalReward::UBTTask_UpdateTacticalReward()
{
	NodeName = "Update Tactical Reward";
	bNotifyTick = false;

	// Set default blackboard keys
	RewardBlackboardKey.SelectedKeyName = "CurrentReward";
	TerminalStateKey.SelectedKeyName = "bTerminalState";
}

EBTNodeResult::Type UBTTask_UpdateTacticalReward::ExecuteTask(UBehaviorTreeComponent& OwnerComp, uint8* NodeMemory)
{
	// Get follower component
	UFollowerAgentComponent* FollowerComp = GetFollowerComponent(OwnerComp);
	if (!FollowerComp)
	{
		UE_LOG(LogTemp, Error, TEXT("BTTask_UpdateTacticalReward: No FollowerAgentComponent found"));
		return EBTNodeResult::Failed;
	}

	// Get blackboard
	UBlackboardComponent* Blackboard = OwnerComp.GetBlackboardComponent();

	// Determine reward value
	float Reward = 0.0f;
	switch (RewardMode)
	{
		case ERewardMode::Manual:
			Reward = ManualReward;
			break;

		case ERewardMode::FromBlackboard:
			if (Blackboard)
			{
				Reward = Blackboard->GetValueAsFloat(RewardBlackboardKey.SelectedKeyName);
			}
			break;

		case ERewardMode::AutoCalculate:
			Reward = CalculateReward(OwnerComp, FollowerComp);
			break;
	}

	// Determine terminal state
	bool bIsTerminal = bTerminalState;
	if (!bIsTerminal && Blackboard && TerminalStateKey.SelectedKeyType == UBlackboardKeyType_Bool::StaticClass())
	{
		bIsTerminal = Blackboard->GetValueAsBool(TerminalStateKey.SelectedKeyName);
	}

	// Provide reward to RL policy
	FollowerComp->ProvideReward(Reward, bIsTerminal);

	// Log reward
	if (bLogReward)
	{
		UE_LOG(LogTemp, Log, TEXT("BTTask_UpdateTacticalReward: Provided reward %.2f to %s (Terminal: %s)"),
			Reward,
			*FollowerComp->GetOwner()->GetName(),
			bIsTerminal ? TEXT("Yes") : TEXT("No"));
	}

	return EBTNodeResult::Succeeded;
}

FString UBTTask_UpdateTacticalReward::GetStaticDescription() const
{
	FString ModeStr;
	switch (RewardMode)
	{
		case ERewardMode::Manual:
			ModeStr = FString::Printf(TEXT("Manual (%.2f)"), ManualReward);
			break;
		case ERewardMode::FromBlackboard:
			ModeStr = FString::Printf(TEXT("From Blackboard '%s'"), *RewardBlackboardKey.SelectedKeyName.ToString());
			break;
		case ERewardMode::AutoCalculate:
			ModeStr = TEXT("Auto-Calculate");
			break;
	}

	return FString::Printf(TEXT("Update reward: %s"), *ModeStr);
}

UFollowerAgentComponent* UBTTask_UpdateTacticalReward::GetFollowerComponent(UBehaviorTreeComponent& OwnerComp) const
{
	AAIController* AIController = OwnerComp.GetAIOwner();
	if (!AIController)
	{
		return nullptr;
	}

	APawn* ControlledPawn = AIController->GetPawn();
	if (!ControlledPawn)
	{
		return nullptr;
	}

	return ControlledPawn->FindComponentByClass<UFollowerAgentComponent>();
}

float UBTTask_UpdateTacticalReward::CalculateReward(UBehaviorTreeComponent& OwnerComp, UFollowerAgentComponent* FollowerComp) const
{
	float TotalReward = 0.0f;

	// Combat rewards
	if (bCalculateCombatRewards)
	{
		TotalReward += CalculateCombatReward(OwnerComp, FollowerComp);
	}

	// Tactical rewards
	if (bCalculateTacticalRewards)
	{
		TotalReward += CalculateTacticalReward(OwnerComp, FollowerComp);
	}

	// Command rewards
	if (bCalculateCommandRewards)
	{
		TotalReward += CalculateCommandReward(OwnerComp, FollowerComp);
	}

	return TotalReward;
}

float UBTTask_UpdateTacticalReward::CalculateCombatReward(UBehaviorTreeComponent& OwnerComp, UFollowerAgentComponent* FollowerComp) const
{
	float CombatReward = 0.0f;

	UBlackboardComponent* Blackboard = OwnerComp.GetBlackboardComponent();
	if (!Blackboard)
	{
		return CombatReward;
	}

	// Check for kills
	if (Blackboard->GetValueAsBool("bJustKilledEnemy"))
	{
		CombatReward += FTacticalRewards::KILL_ENEMY;
		Blackboard->SetValueAsBool("bJustKilledEnemy", false);
	}

	// Check for damage dealt
	float DamageDealt = Blackboard->GetValueAsFloat("RecentDamageDealt");
	if (DamageDealt > 0.0f)
	{
		CombatReward += FTacticalRewards::DAMAGE_ENEMY * (DamageDealt / 100.0f);  // Normalize by 100 HP
		Blackboard->SetValueAsFloat("RecentDamageDealt", 0.0f);
	}

	// Check for damage taken
	float DamageTaken = Blackboard->GetValueAsFloat("RecentDamageTaken");
	if (DamageTaken > 0.0f)
	{
		CombatReward += FTacticalRewards::TAKE_DAMAGE * (DamageTaken / 100.0f);  // Penalty
		Blackboard->SetValueAsFloat("RecentDamageTaken", 0.0f);
	}

	// Check for suppression
	if (Blackboard->GetValueAsBool("bSuppressingEnemy"))
	{
		CombatReward += FTacticalRewards::SUPPRESS_ENEMY;
	}

	return CombatReward;
}

float UBTTask_UpdateTacticalReward::CalculateTacticalReward(UBehaviorTreeComponent& OwnerComp, UFollowerAgentComponent* FollowerComp) const
{
	float TacticalReward = 0.0f;

	UBlackboardComponent* Blackboard = OwnerComp.GetBlackboardComponent();
	if (!Blackboard)
	{
		return TacticalReward;
	}

	// Get observation
	FObservationElement Observation = FollowerComp->GetLocalObservation();

	// Reward for being in cover
	if (Observation.bHasCover)
	{
		TacticalReward += FTacticalRewards::REACH_COVER * 0.1f;  // Small per-tick reward
	}

	// Reward for maintaining formation
	if (Blackboard->GetValueAsBool("bInFormation"))
	{
		TacticalReward += FTacticalRewards::MAINTAIN_FORMATION * 0.1f;
	}

	// Penalty for breaking formation
	if (Blackboard->GetValueAsBool("bBrokeFormation"))
	{
		TacticalReward += FTacticalRewards::BREAK_FORMATION;
		Blackboard->SetValueAsBool("bBrokeFormation", false);
	}

	// Penalty for being out of position
	if (Blackboard->GetValueAsBool("bOutOfPosition"))
	{
		TacticalReward += FTacticalRewards::OUT_OF_POSITION * 0.1f;
	}

	return TacticalReward;
}

float UBTTask_UpdateTacticalReward::CalculateCommandReward(UBehaviorTreeComponent& OwnerComp, UFollowerAgentComponent* FollowerComp) const
{
	float CommandReward = 0.0f;

	UBlackboardComponent* Blackboard = OwnerComp.GetBlackboardComponent();
	if (!Blackboard)
	{
		return CommandReward;
	}

	// Get current command
	FStrategicCommand Command = FollowerComp->GetCurrentCommand();

	// Reward for command progress
	if (Command.Progress > 0.0f && Command.Progress < 1.0f)
	{
		CommandReward += FTacticalRewards::FOLLOW_COMMAND * Command.Progress * 0.1f;
	}

	// Reward for command completion
	if (Blackboard->GetValueAsBool("bCommandCompleted"))
	{
		CommandReward += FTacticalRewards::FOLLOW_COMMAND;
		Blackboard->SetValueAsBool("bCommandCompleted", false);
	}

	// Penalty for ignoring command
	if (Blackboard->GetValueAsBool("bIgnoredCommand"))
	{
		CommandReward += FTacticalRewards::IGNORE_COMMAND;
		Blackboard->SetValueAsBool("bIgnoredCommand", false);
	}

	// Support action rewards
	if (Blackboard->GetValueAsBool("bJustRescuedAlly"))
	{
		CommandReward += FTacticalRewards::RESCUE_ALLY;
		Blackboard->SetValueAsBool("bJustRescuedAlly", false);
	}

	if (Blackboard->GetValueAsBool("bProvidingCoveringFire"))
	{
		CommandReward += FTacticalRewards::COVERING_FIRE * 0.1f;
	}

	return CommandReward;
}
