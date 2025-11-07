// Copyright Epic Games, Inc. All Rights Reserved.

#include "BehaviorTree/Tasks/BTTask_ExecuteSupport.h"
#include "Team/FollowerAgentComponent.h"
#include "RL/RLPolicyNetwork.h"
#include "Interfaces/CombatStatsInterface.h"
#include "BehaviorTree/BlackboardComponent.h"
#include "BehaviorTree/BehaviorTreeComponent.h"
#include "AIController.h"
#include "GameFramework/Character.h"
#include "GameFramework/CharacterMovementComponent.h"
#include "DrawDebugHelpers.h"
#include "Kismet/GameplayStatics.h"

UBTTask_ExecuteSupport::UBTTask_ExecuteSupport()
{
	NodeName = "Execute Support";
	bNotifyTick = true;
	bNotifyTaskFinished = true;

	// Set default blackboard keys
	CurrentCommandKey.SelectedKeyName = "CurrentCommand";
	AllyToSupportKey.SelectedKeyName = "AllyToSupport";
	SupportLocationKey.SelectedKeyName = "SupportLocation";
	TacticalActionKey.SelectedKeyName = "TacticalAction";
	SupportTargetKey.SelectedKeyName = "SupportTarget";
	ActionProgressKey.SelectedKeyName = "ActionProgress";
}

EBTNodeResult::Type UBTTask_ExecuteSupport::ExecuteTask(UBehaviorTreeComponent& OwnerComp, uint8* NodeMemory)
{
	FBTExecuteSupportMemory* Memory = CastInstanceNodeMemory<FBTExecuteSupportMemory>(NodeMemory);
	if (!Memory)
	{
		return EBTNodeResult::Failed;
	}

	// Get follower component
	UFollowerAgentComponent* FollowerComp = GetFollowerComponent(OwnerComp);
	if (!FollowerComp)
	{
		UE_LOG(LogTemp, Error, TEXT("BTTask_ExecuteSupport: No FollowerAgentComponent found"));
		return EBTNodeResult::Failed;
	}

	// Initialize memory
	Memory->CurrentTactic = ETacticalAction::ProvideCoveringFire;
	Memory->TimeInCurrentTactic = 0.0f;
	Memory->TimeSinceLastRLQuery = 0.0f;
	Memory->bReloadInProgress = false;
	Memory->bAllyRescued = false;
	Memory->CoveringFireShots = 0;
	Memory->ThreatsNeutralized = 0;

	// Get ally to support
	Memory->AllyBeingSupported = GetAllyToSupport(OwnerComp);
	if (Memory->AllyBeingSupported)
	{
		Memory->InitialAllyHealth = GetAllyHealthPercentage(Memory->AllyBeingSupported);

		APawn* ControlledPawn = OwnerComp.GetAIOwner()->GetPawn();
		if (ControlledPawn)
		{
			Memory->DistanceToAlly = FVector::Dist(ControlledPawn->GetActorLocation(), Memory->AllyBeingSupported->GetActorLocation());
		}
	}

	// Query initial tactical action from RL policy
	Memory->CurrentTactic = QueryTacticalAction(OwnerComp);

	if (bLogActions)
	{
		UE_LOG(LogTemp, Log, TEXT("BTTask_ExecuteSupport: Starting support with tactic '%s'"),
			*URLPolicyNetwork::GetActionName(Memory->CurrentTactic));
	}

	// Set action in blackboard
	UBlackboardComponent* Blackboard = OwnerComp.GetBlackboardComponent();
	if (Blackboard)
	{
		Blackboard->SetValueAsEnum(TacticalActionKey.SelectedKeyName, static_cast<uint8>(Memory->CurrentTactic));
		UpdateActionProgress(OwnerComp, 0.0f);
	}

	return EBTNodeResult::InProgress;
}

EBTNodeResult::Type UBTTask_ExecuteSupport::AbortTask(UBehaviorTreeComponent& OwnerComp, uint8* NodeMemory)
{
	FBTExecuteSupportMemory* Memory = CastInstanceNodeMemory<FBTExecuteSupportMemory>(NodeMemory);
	if (!Memory)
	{
		return EBTNodeResult::Aborted;
	}

	// Provide negative reward for abortion
	UFollowerAgentComponent* FollowerComp = GetFollowerComponent(OwnerComp);
	if (FollowerComp)
	{
		FollowerComp->ProvideReward(-2.0f, false);
	}

	if (bLogActions)
	{
		UE_LOG(LogTemp, Warning, TEXT("BTTask_ExecuteSupport: Support aborted after %.1f seconds"),
			Memory->TimeInCurrentTactic);
	}

	return EBTNodeResult::Aborted;
}

void UBTTask_ExecuteSupport::TickTask(UBehaviorTreeComponent& OwnerComp, uint8* NodeMemory, float DeltaSeconds)
{
	FBTExecuteSupportMemory* Memory = CastInstanceNodeMemory<FBTExecuteSupportMemory>(NodeMemory);
	if (!Memory)
	{
		FinishLatentTask(OwnerComp, EBTNodeResult::Failed);
		return;
	}

	Memory->TimeInCurrentTactic += DeltaSeconds;
	Memory->TimeSinceLastRLQuery += DeltaSeconds;

	// Re-query RL policy if interval exceeded
	if (RLQueryInterval > 0.0f && Memory->TimeSinceLastRLQuery >= RLQueryInterval)
	{
		ETacticalAction NewTactic = QueryTacticalAction(OwnerComp);
		if (NewTactic != Memory->CurrentTactic)
		{
			// Calculate reward for previous tactic
			float Reward = CalculateTacticalReward(OwnerComp, Memory);
			UFollowerAgentComponent* FollowerComp = GetFollowerComponent(OwnerComp);
			if (FollowerComp)
			{
				FollowerComp->ProvideReward(Reward, false);
			}

			// Switch to new tactic
			Memory->CurrentTactic = NewTactic;
			Memory->TimeInCurrentTactic = 0.0f;

			if (bLogActions)
			{
				UE_LOG(LogTemp, Log, TEXT("BTTask_ExecuteSupport: Switching to tactic '%s' (Reward: %.2f)"),
					*URLPolicyNetwork::GetActionName(NewTactic), Reward);
			}

			// Update blackboard
			UBlackboardComponent* Blackboard = OwnerComp.GetBlackboardComponent();
			if (Blackboard)
			{
				Blackboard->SetValueAsEnum(TacticalActionKey.SelectedKeyName, static_cast<uint8>(NewTactic));
			}
		}

		Memory->TimeSinceLastRLQuery = 0.0f;
	}

	// Execute current tactical action
	ExecuteTacticalAction(OwnerComp, Memory, Memory->CurrentTactic, DeltaSeconds);

	// Draw debug info
	if (bDrawDebugInfo)
	{
		DrawDebugInfo(OwnerComp, Memory);
	}

	// Check completion conditions
	if (ShouldCompleteSupport(OwnerComp, Memory))
	{
		// Calculate final reward
		float FinalReward = CalculateTacticalReward(OwnerComp, Memory);
		UFollowerAgentComponent* FollowerComp = GetFollowerComponent(OwnerComp);
		if (FollowerComp)
		{
			FollowerComp->ProvideReward(FinalReward, false);
		}

		if (bLogActions)
		{
			UE_LOG(LogTemp, Log, TEXT("BTTask_ExecuteSupport: Support complete (Final Reward: %.2f, Time: %.1f seconds)"),
				FinalReward, Memory->TimeInCurrentTactic);
		}

		FinishLatentTask(OwnerComp, EBTNodeResult::Succeeded);
	}
}

FString UBTTask_ExecuteSupport::GetStaticDescription() const
{
	return FString::Printf(TEXT("Execute support tactics via RL policy\nQuery Interval: %.1fs\nSupport Range: %.0fcm"),
		RLQueryInterval, MaxSupportRange);
}

UFollowerAgentComponent* UBTTask_ExecuteSupport::GetFollowerComponent(UBehaviorTreeComponent& OwnerComp) const
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

ETacticalAction UBTTask_ExecuteSupport::QueryTacticalAction(UBehaviorTreeComponent& OwnerComp) const
{
	UFollowerAgentComponent* FollowerComp = GetFollowerComponent(OwnerComp);
	if (!FollowerComp)
	{
		return ETacticalAction::ProvideCoveringFire;
	}

	return FollowerComp->QueryRLPolicy();
}

void UBTTask_ExecuteSupport::ExecuteTacticalAction(UBehaviorTreeComponent& OwnerComp, FBTExecuteSupportMemory* Memory, ETacticalAction Action, float DeltaSeconds)
{
	switch (Action)
	{
	case ETacticalAction::ProvideCoveringFire:
		ExecuteCoveringFire(OwnerComp, Memory, DeltaSeconds);
		break;

	case ETacticalAction::Reload:
		ExecuteReload(OwnerComp, Memory, DeltaSeconds);
		break;

	case ETacticalAction::UseAbility:
		ExecuteUseAbility(OwnerComp, Memory, DeltaSeconds);
		break;

	case ETacticalAction::CautiousAdvance:
	case ETacticalAction::AggressiveAssault:
		// Rescue/assist ally
		ExecuteRescueAlly(OwnerComp, Memory, DeltaSeconds);
		break;

	default:
		// Fallback to covering fire
		ExecuteCoveringFire(OwnerComp, Memory, DeltaSeconds);
		break;
	}
}

void UBTTask_ExecuteSupport::ExecuteCoveringFire(UBehaviorTreeComponent& OwnerComp, FBTExecuteSupportMemory* Memory, float DeltaSeconds)
{
	// Find threats to ally
	AActor* Ally = Memory->AllyBeingSupported;
	if (!Ally)
	{
		Ally = GetAllyToSupport(OwnerComp);
		Memory->AllyBeingSupported = Ally;
	}

	if (!Ally)
	{
		return;
	}

	// Find threats engaging ally
	TArray<AActor*> Threats = FindThreatsEngagingAlly(OwnerComp, Ally);
	if (Threats.Num() > 0)
	{
		Memory->ThreatToSuppress = Threats[0];

		// Position ourselves at optimal distance from ally
		APawn* ControlledPawn = OwnerComp.GetAIOwner()->GetPawn();
		if (ControlledPawn)
		{
			float DistanceToAlly = FVector::Dist(ControlledPawn->GetActorLocation(), Ally->GetActorLocation());

			if (DistanceToAlly > OptimalSupportDistance * 1.5f)
			{
				// Too far, move closer
				MoveToSupportPosition(OwnerComp, Ally->GetActorLocation(), DeltaSeconds);
			}
			else if (DistanceToAlly < OptimalSupportDistance * 0.5f)
			{
				// Too close, back off slightly
				FVector AwayDirection = (ControlledPawn->GetActorLocation() - Ally->GetActorLocation()).GetSafeNormal();
				FVector BackOffPosition = ControlledPawn->GetActorLocation() + AwayDirection * 200.0f;
				MoveToSupportPosition(OwnerComp, BackOffPosition, DeltaSeconds);
			}

			// Provide covering fire
			ProvideCoveringFireAtThreat(OwnerComp, Memory->ThreatToSuppress, CoveringFireAccuracy, CoveringFireRateMultiplier);
			Memory->CoveringFireShots++;
		}
	}

	// Update progress
	float Progress = FMath::Clamp(Memory->TimeInCurrentTactic / 10.0f, 0.0f, 1.0f);
	UpdateActionProgress(OwnerComp, Progress);
}

void UBTTask_ExecuteSupport::ExecuteReload(UBehaviorTreeComponent& OwnerComp, FBTExecuteSupportMemory* Memory, float DeltaSeconds)
{
	if (!Memory->bReloadInProgress)
	{
		Memory->bReloadInProgress = true;

		if (bLogActions)
		{
			UE_LOG(LogTemp, Log, TEXT("BTTask_ExecuteSupport: Starting reload"));
		}
	}

	// TODO: Implement actual reload logic via weapon component
	// For now, simulate reload time

	// Move to safe position while reloading
	AActor* Ally = Memory->AllyBeingSupported;
	if (Ally)
	{
		// Move closer to ally for protection
		MoveToSupportPosition(OwnerComp, Ally->GetActorLocation(), DeltaSeconds);
	}

	// Reload takes ~2 seconds
	float ReloadDuration = 2.0f;
	float Progress = FMath::Clamp(Memory->TimeInCurrentTactic / ReloadDuration, 0.0f, 1.0f);
	UpdateActionProgress(OwnerComp, Progress);

	// Complete reload after duration
	if (Memory->TimeInCurrentTactic >= ReloadDuration)
	{
		Memory->bReloadInProgress = false;

		if (bLogActions)
		{
			UE_LOG(LogTemp, Log, TEXT("BTTask_ExecuteSupport: Reload complete"));
		}

		// Switch back to covering fire
		Memory->CurrentTactic = ETacticalAction::ProvideCoveringFire;
		Memory->TimeInCurrentTactic = 0.0f;
	}
}

void UBTTask_ExecuteSupport::ExecuteUseAbility(UBehaviorTreeComponent& OwnerComp, FBTExecuteSupportMemory* Memory, float DeltaSeconds)
{
	// TODO: Implement ability usage
	// This would check for available abilities and use them appropriately

	if (bLogActions)
	{
		UE_LOG(LogTemp, Log, TEXT("BTTask_ExecuteSupport: Using ability (placeholder)"));
	}

	// For now, fall back to covering fire after attempting ability
	if (Memory->TimeInCurrentTactic > 1.0f)
	{
		Memory->CurrentTactic = ETacticalAction::ProvideCoveringFire;
		Memory->TimeInCurrentTactic = 0.0f;
	}

	float Progress = FMath::Clamp(Memory->TimeInCurrentTactic / 1.0f, 0.0f, 1.0f);
	UpdateActionProgress(OwnerComp, Progress);
}

void UBTTask_ExecuteSupport::ExecuteRescueAlly(UBehaviorTreeComponent& OwnerComp, FBTExecuteSupportMemory* Memory, float DeltaSeconds)
{
	AActor* Ally = Memory->AllyBeingSupported;
	if (!Ally)
	{
		return;
	}

	APawn* ControlledPawn = OwnerComp.GetAIOwner()->GetPawn();
	if (!ControlledPawn)
	{
		return;
	}

	float DistanceToAlly = FVector::Dist(ControlledPawn->GetActorLocation(), Ally->GetActorLocation());
	Memory->DistanceToAlly = DistanceToAlly;

	// Move toward ally
	if (DistanceToAlly > 200.0f)
	{
		MoveToSupportPosition(OwnerComp, Ally->GetActorLocation(), DeltaSeconds);

		// Provide covering fire while moving
		TArray<AActor*> Threats = FindThreatsEngagingAlly(OwnerComp, Ally);
		if (Threats.Num() > 0)
		{
			ProvideCoveringFireAtThreat(OwnerComp, Threats[0], 0.6f, CoveringFireRateMultiplier * 0.8f);
		}
	}
	else
	{
		// Reached ally
		Memory->bAllyRescued = true;

		if (bLogActions)
		{
			UE_LOG(LogTemp, Log, TEXT("BTTask_ExecuteSupport: Successfully reached ally for rescue"));
		}

		// Now provide covering fire from ally's position
		ExecuteCoveringFire(OwnerComp, Memory, DeltaSeconds);
	}

	// Update progress
	float Progress = 1.0f - FMath::Clamp(DistanceToAlly / Memory->DistanceToAlly, 0.0f, 1.0f);
	UpdateActionProgress(OwnerComp, Progress);
}

AActor* UBTTask_ExecuteSupport::GetAllyToSupport(UBehaviorTreeComponent& OwnerComp) const
{
	UBlackboardComponent* Blackboard = OwnerComp.GetBlackboardComponent();
	if (Blackboard)
	{
		AActor* Ally = Cast<AActor>(Blackboard->GetValueAsObject(AllyToSupportKey.SelectedKeyName));
		if (Ally)
		{
			return Ally;
		}
	}

	// Find nearest ally in danger
	return FindNearestAllyInDanger(OwnerComp);
}

AActor* UBTTask_ExecuteSupport::FindNearestAllyInDanger(UBehaviorTreeComponent& OwnerComp) const
{
	UWorld* World = OwnerComp.GetWorld();
	if (!World)
	{
		return nullptr;
	}

	APawn* ControlledPawn = OwnerComp.GetAIOwner()->GetPawn();
	if (!ControlledPawn)
	{
		return nullptr;
	}

	// Find all allies (actors with "Ally" tag)
	TArray<AActor*> AllAllies;
	UGameplayStatics::GetAllActorsWithTag(World, FName("Ally"), AllAllies);

	AActor* NearestAllyInDanger = nullptr;
	float NearestDistance = MaxSupportRange;

	for (AActor* Ally : AllAllies)
	{
		if (Ally && Ally != ControlledPawn)
		{
			float Health = GetAllyHealthPercentage(Ally);
			if (Health < AllyDangerHealthThreshold && Health > 0.0f)
			{
				float Distance = FVector::Dist(ControlledPawn->GetActorLocation(), Ally->GetActorLocation());
				if (Distance < NearestDistance)
				{
					NearestDistance = Distance;
					NearestAllyInDanger = Ally;
				}
			}
		}
	}

	return NearestAllyInDanger;
}

TArray<AActor*> UBTTask_ExecuteSupport::FindThreatsEngagingAlly(UBehaviorTreeComponent& OwnerComp, AActor* Ally) const
{
	TArray<AActor*> Threats;

	if (!Ally)
	{
		return Threats;
	}

	UWorld* World = OwnerComp.GetWorld();
	if (!World)
	{
		return Threats;
	}

	// Find all enemies
	TArray<AActor*> AllEnemies;
	UGameplayStatics::GetAllActorsWithTag(World, FName("Enemy"), AllEnemies);

	// Filter to enemies near ally
	for (AActor* Enemy : AllEnemies)
	{
		if (Enemy)
		{
			float Distance = FVector::Dist(Ally->GetActorLocation(), Enemy->GetActorLocation());
			if (Distance < 2000.0f)  // Threat range
			{
				Threats.Add(Enemy);
			}
		}
	}

	return Threats;
}

FVector UBTTask_ExecuteSupport::GetSupportLocation(UBehaviorTreeComponent& OwnerComp) const
{
	UBlackboardComponent* Blackboard = OwnerComp.GetBlackboardComponent();
	if (Blackboard)
	{
		FVector Location = Blackboard->GetValueAsVector(SupportLocationKey.SelectedKeyName);
		if (!Location.IsNearlyZero())
		{
			return Location;
		}
	}

	// Fallback to ally location
	AActor* Ally = GetAllyToSupport(OwnerComp);
	if (Ally)
	{
		return Ally->GetActorLocation();
	}

	return FVector::ZeroVector;
}


float UBTTask_ExecuteSupport::GetAllyHealthPercentage(AActor* Ally) const
{
	if (!Ally)
	{
		return 0.0f;
	}

	// Try to get health via CombatStatsInterface
	ICombatStatsInterface* CombatStats = Cast<ICombatStatsInterface>(Ally);
	if (CombatStats)
	{
		return CombatStats->Execute_GetHealthPercentage(Ally) / 100.0f;
	}

	// Default: assume healthy
	return 1.0f;
}

float UBTTask_ExecuteSupport::CalculateTacticalReward(UBehaviorTreeComponent& OwnerComp, const FBTExecuteSupportMemory* Memory) const
{
	float Reward = 0.0f;

	// Reward for providing covering fire
	Reward += (Memory->CoveringFireShots / 10.0f) * CoveringFireReward;

	// Reward for safe reload
	if (Memory->bReloadInProgress && Memory->TimeInCurrentTactic > 1.5f)
	{
		Reward += SafeReloadReward;
	}

	// Reward for rescuing ally
	if (Memory->bAllyRescued)
	{
		Reward += RescueAllyReward;

		// Bonus if ally health improved
		if (Memory->AllyBeingSupported)
		{
			float CurrentAllyHealth = GetAllyHealthPercentage(Memory->AllyBeingSupported);
			if (CurrentAllyHealth > Memory->InitialAllyHealth)
			{
				Reward += 5.0f;
			}
		}
	}
	else if (Memory->AllyBeingSupported)
	{
		// Penalty if ally died before rescue
		float CurrentAllyHealth = GetAllyHealthPercentage(Memory->AllyBeingSupported);
		if (CurrentAllyHealth <= 0.0f)
		{
			Reward += FailedRescuePenalty;
		}
	}

	// Reward for neutralizing threats
	Reward += Memory->ThreatsNeutralized * 5.0f;

	return Reward;
}

void UBTTask_ExecuteSupport::UpdateActionProgress(UBehaviorTreeComponent& OwnerComp, float Progress) const
{
	UBlackboardComponent* Blackboard = OwnerComp.GetBlackboardComponent();
	if (Blackboard)
	{
		Blackboard->SetValueAsFloat(ActionProgressKey.SelectedKeyName, Progress);
	}
}

void UBTTask_ExecuteSupport::DrawDebugInfo(UBehaviorTreeComponent& OwnerComp, const FBTExecuteSupportMemory* Memory) const
{
	AAIController* AIController = OwnerComp.GetAIOwner();
	if (!AIController || !AIController->GetPawn())
	{
		return;
	}

	UWorld* World = AIController->GetWorld();
	if (!World)
	{
		return;
	}

	APawn* ControlledPawn = AIController->GetPawn();
	FVector PawnLocation = ControlledPawn->GetActorLocation();

	// Draw current tactic
	FString TacticText = FString::Printf(TEXT("Support Tactic:\n%s\nTime: %.1fs"),
		*URLPolicyNetwork::GetActionName(Memory->CurrentTactic),
		Memory->TimeInCurrentTactic);
	DrawDebugString(World, PawnLocation + FVector(0, 0, 150), TacticText, nullptr, FColor::Green, 0.0f, true);

	// Draw ally connection
	if (Memory->AllyBeingSupported)
	{
		DrawDebugLine(World, PawnLocation, Memory->AllyBeingSupported->GetActorLocation(), FColor::Green, false, 0.0f, 0, 2.0f);
		DrawDebugSphere(World, Memory->AllyBeingSupported->GetActorLocation(), 100.0f, 12, FColor::Green, false, 0.0f, 0, 2.0f);
	}

	// Draw threat
	if (Memory->ThreatToSuppress)
	{
		DrawDebugLine(World, PawnLocation, Memory->ThreatToSuppress->GetActorLocation(), FColor::Orange, false, 0.0f, 0, 1.0f);
	}
}

bool UBTTask_ExecuteSupport::ShouldCompleteSupport(UBehaviorTreeComponent& OwnerComp, const FBTExecuteSupportMemory* Memory) const
{
	// Check if command has changed
	UFollowerAgentComponent* FollowerComp = GetFollowerComponent(OwnerComp);
	if (FollowerComp)
	{
		EFollowerState CurrentState = FollowerComp->GetCurrentState();
		if (CurrentState != EFollowerState::Support)
		{
			return true;
		}
	}

	// Check if ally is no longer in danger
	if (Memory->AllyBeingSupported)
	{
		float AllyHealth = GetAllyHealthPercentage(Memory->AllyBeingSupported);
		if (AllyHealth > 0.7f || AllyHealth <= 0.0f)
		{
			// Ally either recovered or died
			return true;
		}
	}

	return false;
}

void UBTTask_ExecuteSupport::MoveToSupportPosition(UBehaviorTreeComponent& OwnerComp, const FVector& Destination, float DeltaSeconds)
{
	AAIController* AIController = OwnerComp.GetAIOwner();
	if (!AIController)
	{
		return;
	}

	// Use AI movement
	AIController->MoveToLocation(Destination, 100.0f, true, true, false, true, 0, true);

	// Moderate speed
	ACharacter* Character = Cast<ACharacter>(AIController->GetPawn());
	if (Character && Character->GetCharacterMovement())
	{
		UCharacterMovementComponent* Movement = Character->GetCharacterMovement();
		float BaseSpeed = Movement->GetMaxSpeed();
		Movement->MaxWalkSpeed = BaseSpeed * 0.8f;
	}
}

void UBTTask_ExecuteSupport::ProvideCoveringFireAtThreat(UBehaviorTreeComponent& OwnerComp, AActor* Threat, float AccuracyModifier, float FireRateMultiplier)
{
	if (!Threat)
	{
		return;
	}

	AAIController* AIController = OwnerComp.GetAIOwner();
	if (!AIController)
	{
		return;
	}

	// Set focus on threat
	AIController->SetFocus(Threat);

	// TODO: Implement actual weapon firing logic

	// Update blackboard with fire command parameters
	UBlackboardComponent* Blackboard = OwnerComp.GetBlackboardComponent();
	if (Blackboard)
	{
		Blackboard->SetValueAsObject("FireTarget", Threat);
		Blackboard->SetValueAsFloat("FireAccuracy", AccuracyModifier);
		Blackboard->SetValueAsFloat("FireRate", FireRateMultiplier);
		Blackboard->SetValueAsObject(SupportTargetKey.SelectedKeyName, Threat);
	}
}
