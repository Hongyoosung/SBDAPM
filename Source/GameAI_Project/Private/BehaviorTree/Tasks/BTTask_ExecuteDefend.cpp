// Copyright Epic Games, Inc. All Rights Reserved.

#include "BehaviorTree/Tasks/BTTask_ExecuteDefend.h"
#include "Team/FollowerAgentComponent.h"
#include "RL/RLPolicyNetwork.h"
#include "BehaviorTree/BlackboardComponent.h"
#include "BehaviorTree/BehaviorTreeComponent.h"
#include "AIController.h"
#include "GameFramework/Character.h"
#include "GameFramework/CharacterMovementComponent.h"
#include "DrawDebugHelpers.h"
#include "NavigationSystem.h"
#include "Kismet/GameplayStatics.h"

UBTTask_ExecuteDefend::UBTTask_ExecuteDefend()
{
	NodeName = "Execute Defend";
	bNotifyTick = true;
	bNotifyTaskFinished = true;

	// Set default blackboard keys
	CurrentCommandKey.SelectedKeyName = "CurrentCommand";
	DefendLocationKey.SelectedKeyName = "DefendLocation";
	ThreatActorsKey.SelectedKeyName = "ThreatActors";
	TacticalActionKey.SelectedKeyName = "TacticalAction";
	CoverActorKey.SelectedKeyName = "CoverActor";
	ActionProgressKey.SelectedKeyName = "ActionProgress";
}

EBTNodeResult::Type UBTTask_ExecuteDefend::ExecuteTask(UBehaviorTreeComponent& OwnerComp, uint8* NodeMemory)
{
	FBTExecuteDefendMemory* Memory = CastInstanceNodeMemory<FBTExecuteDefendMemory>(NodeMemory);
	if (!Memory)
	{
		return EBTNodeResult::Failed;
	}

	// Get follower component
	UFollowerAgentComponent* FollowerComp = GetFollowerComponent(OwnerComp);
	if (!FollowerComp)
	{
		UE_LOG(LogTemp, Error, TEXT("BTTask_ExecuteDefend: No FollowerAgentComponent found"));
		return EBTNodeResult::Failed;
	}

	// Initialize memory
	Memory->CurrentTactic = ETacticalAction::DefensiveHold;
	Memory->TimeInCurrentTactic = 0.0f;
	Memory->TimeSinceLastRLQuery = 0.0f;
	Memory->TimeInDefensivePosition = 0.0f;
	Memory->ShotsBlockedByCover = 0;
	Memory->DamageTaken = 0;
	Memory->bHasCover = false;
	Memory->CurrentCover = nullptr;
	Memory->VisibleThreats = 0;

	// Get defend location
	Memory->DefendPosition = GetDefendLocation(OwnerComp);

	// Query initial tactical action from RL policy
	Memory->CurrentTactic = QueryTacticalAction(OwnerComp);

	if (bLogActions)
	{
		UE_LOG(LogTemp, Log, TEXT("BTTask_ExecuteDefend: Starting defense with tactic '%s'"),
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

EBTNodeResult::Type UBTTask_ExecuteDefend::AbortTask(UBehaviorTreeComponent& OwnerComp, uint8* NodeMemory)
{
	FBTExecuteDefendMemory* Memory = CastInstanceNodeMemory<FBTExecuteDefendMemory>(NodeMemory);
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
		UE_LOG(LogTemp, Warning, TEXT("BTTask_ExecuteDefend: Defense aborted after %.1f seconds"),
			Memory->TimeInCurrentTactic);
	}

	return EBTNodeResult::Aborted;
}

void UBTTask_ExecuteDefend::TickTask(UBehaviorTreeComponent& OwnerComp, uint8* NodeMemory, float DeltaSeconds)
{
	FBTExecuteDefendMemory* Memory = CastInstanceNodeMemory<FBTExecuteDefendMemory>(NodeMemory);
	if (!Memory)
	{
		FinishLatentTask(OwnerComp, EBTNodeResult::Failed);
		return;
	}

	Memory->TimeInCurrentTactic += DeltaSeconds;
	Memory->TimeSinceLastRLQuery += DeltaSeconds;
	Memory->TimeInDefensivePosition += DeltaSeconds;

	// Update cover status
	UpdateCoverStatus(OwnerComp, Memory);

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
				UE_LOG(LogTemp, Log, TEXT("BTTask_ExecuteDefend: Switching to tactic '%s' (Reward: %.2f)"),
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
	if (ShouldCompleteDefense(OwnerComp, Memory))
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
			UE_LOG(LogTemp, Log, TEXT("BTTask_ExecuteDefend: Defense complete (Final Reward: %.2f, Time: %.1f seconds)"),
				FinalReward, Memory->TimeInCurrentTactic);
		}

		FinishLatentTask(OwnerComp, EBTNodeResult::Succeeded);
	}
}

FString UBTTask_ExecuteDefend::GetStaticDescription() const
{
	return FString::Printf(TEXT("Execute defensive tactics via RL policy\nQuery Interval: %.1fs\nDefend Radius: %.0fcm"),
		RLQueryInterval, MaxDefendRadius);
}

UFollowerAgentComponent* UBTTask_ExecuteDefend::GetFollowerComponent(UBehaviorTreeComponent& OwnerComp) const
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

ETacticalAction UBTTask_ExecuteDefend::QueryTacticalAction(UBehaviorTreeComponent& OwnerComp) const
{
	UFollowerAgentComponent* FollowerComp = GetFollowerComponent(OwnerComp);
	if (!FollowerComp)
	{
		return ETacticalAction::DefensiveHold;
	}

	return FollowerComp->QueryRLPolicy();
}

void UBTTask_ExecuteDefend::ExecuteTacticalAction(UBehaviorTreeComponent& OwnerComp, FBTExecuteDefendMemory* Memory, ETacticalAction Action, float DeltaSeconds)
{
	switch (Action)
	{
	case ETacticalAction::DefensiveHold:
		ExecuteDefensiveHold(OwnerComp, Memory, DeltaSeconds);
		break;

	case ETacticalAction::SeekCover:
		ExecuteSeekCover(OwnerComp, Memory, DeltaSeconds);
		break;

	case ETacticalAction::SuppressiveFire:
		ExecuteSuppressiveFire(OwnerComp, Memory, DeltaSeconds);
		break;

	case ETacticalAction::TacticalRetreat:
		ExecuteTacticalRetreat(OwnerComp, Memory, DeltaSeconds);
		break;

	case ETacticalAction::MaintainDistance:
	case ETacticalAction::Hold:
		// Similar to defensive hold
		ExecuteDefensiveHold(OwnerComp, Memory, DeltaSeconds);
		break;

	default:
		// Fallback to defensive hold
		ExecuteDefensiveHold(OwnerComp, Memory, DeltaSeconds);
		break;
	}
}

void UBTTask_ExecuteDefend::ExecuteDefensiveHold(UBehaviorTreeComponent& OwnerComp, FBTExecuteDefendMemory* Memory, float DeltaSeconds)
{
	APawn* ControlledPawn = OwnerComp.GetAIOwner()->GetPawn();
	if (!ControlledPawn)
	{
		return;
	}

	// Check if within defensive radius
	if (!IsWithinDefensiveRadius(OwnerComp, ControlledPawn->GetActorLocation()))
	{
		// Move back to defensive position
		MoveToDefensivePosition(OwnerComp, Memory->DefendPosition, false, DeltaSeconds);
	}
	else
	{
		// Hold position, scan for threats
		AAIController* AIController = OwnerComp.GetAIOwner();
		if (AIController)
		{
			// Face nearest threat
			AActor* NearestThreat = GetNearestThreat(OwnerComp);
			if (NearestThreat)
			{
				AIController->SetFocus(NearestThreat);
				Memory->LastKnownThreatLocation = NearestThreat->GetActorLocation();
			}
		}
	}

	// Engage visible threats
	EngageThreats(OwnerComp, 1.0f, DefensiveFireRateMultiplier);

	// Update progress based on time held
	float Progress = FMath::Clamp(Memory->TimeInDefensivePosition / DefensiveHoldDuration, 0.0f, 1.0f);
	UpdateActionProgress(OwnerComp, Progress);
}

void UBTTask_ExecuteDefend::ExecuteSeekCover(UBehaviorTreeComponent& OwnerComp, FBTExecuteDefendMemory* Memory, float DeltaSeconds)
{
	APawn* ControlledPawn = OwnerComp.GetAIOwner()->GetPawn();
	if (!ControlledPawn)
	{
		return;
	}

	// Find cover if we don't have any
	if (!Memory->CurrentCover || !Memory->bHasCover)
	{
		Memory->CurrentCover = FindNearestCover(OwnerComp, ControlledPawn->GetActorLocation());

		if (Memory->CurrentCover)
		{
			if (bLogActions)
			{
				UE_LOG(LogTemp, Log, TEXT("BTTask_ExecuteDefend: Found cover: %s"), *Memory->CurrentCover->GetName());
			}

			// Update blackboard
			UBlackboardComponent* Blackboard = OwnerComp.GetBlackboardComponent();
			if (Blackboard)
			{
				Blackboard->SetValueAsObject(CoverActorKey.SelectedKeyName, Memory->CurrentCover);
			}
		}
	}

	// Move to cover
	if (Memory->CurrentCover)
	{
		FVector CoverLocation = Memory->CurrentCover->GetActorLocation();
		float DistanceToCover = FVector::Dist(ControlledPawn->GetActorLocation(), CoverLocation);

		if (DistanceToCover > 100.0f)
		{
			MoveToDefensivePosition(OwnerComp, CoverLocation, true, DeltaSeconds);
		}
		else
		{
			Memory->bHasCover = true;

			// Stay in cover and engage threats
			EngageThreats(OwnerComp, CoverAccuracyBonus, DefensiveFireRateMultiplier);
		}

		// Update progress
		float Progress = 1.0f - FMath::Clamp(DistanceToCover / CoverSearchRadius, 0.0f, 1.0f);
		UpdateActionProgress(OwnerComp, Progress);
	}
	else
	{
		// No cover found, fall back to defensive hold
		ExecuteDefensiveHold(OwnerComp, Memory, DeltaSeconds);
	}
}

void UBTTask_ExecuteDefend::ExecuteSuppressiveFire(UBehaviorTreeComponent& OwnerComp, FBTExecuteDefendMemory* Memory, float DeltaSeconds)
{
	APawn* ControlledPawn = OwnerComp.GetAIOwner()->GetPawn();
	if (!ControlledPawn)
	{
		return;
	}

	// Maintain position
	if (!IsWithinDefensiveRadius(OwnerComp, ControlledPawn->GetActorLocation()))
	{
		MoveToDefensivePosition(OwnerComp, Memory->DefendPosition, Memory->bHasCover, DeltaSeconds);
	}

	// Fire at threats with high rate, lower accuracy
	TArray<AActor*> Threats = GetThreatActors(OwnerComp);
	if (Threats.Num() > 0)
	{
		AActor* Target = Threats[0];

		AAIController* AIController = OwnerComp.GetAIOwner();
		if (AIController)
		{
			AIController->SetFocus(Target);
		}

		// Suppressive fire: high rate, low accuracy
		float AccuracyModifier = Memory->bHasCover ? 0.5f : 0.3f;
		EngageThreats(OwnerComp, AccuracyModifier, DefensiveFireRateMultiplier * 2.0f);
	}

	// Update progress based on time
	float Progress = FMath::Clamp(Memory->TimeInCurrentTactic / 10.0f, 0.0f, 1.0f);
	UpdateActionProgress(OwnerComp, Progress);
}

void UBTTask_ExecuteDefend::ExecuteTacticalRetreat(UBehaviorTreeComponent& OwnerComp, FBTExecuteDefendMemory* Memory, float DeltaSeconds)
{
	APawn* ControlledPawn = OwnerComp.GetAIOwner()->GetPawn();
	if (!ControlledPawn)
	{
		return;
	}

	// Calculate retreat position (away from threats, toward defend location)
	FVector CurrentLocation = ControlledPawn->GetActorLocation();
	FVector RetreatDirection = FVector::ZeroVector;

	// Move away from threats
	TArray<AActor*> Threats = GetThreatActors(OwnerComp);
	if (Threats.Num() > 0)
	{
		for (AActor* Threat : Threats)
		{
			if (Threat)
			{
				FVector AwayFromThreat = (CurrentLocation - Threat->GetActorLocation()).GetSafeNormal();
				RetreatDirection += AwayFromThreat;
			}
		}

		RetreatDirection.Normalize();
	}
	else
	{
		// No immediate threats, retreat toward defend location
		RetreatDirection = (Memory->DefendPosition - CurrentLocation).GetSafeNormal();
	}

	// Calculate retreat destination
	FVector RetreatDestination = CurrentLocation + (RetreatDirection * 500.0f);

	// Ensure destination is navigable
	UNavigationSystemV1* NavSys = UNavigationSystemV1::GetCurrent(OwnerComp.GetWorld());
	if (NavSys)
	{
		FNavLocation NavLocation;
		if (NavSys->ProjectPointToNavigation(RetreatDestination, NavLocation, FVector(500.0f, 500.0f, 500.0f)))
		{
			RetreatDestination = NavLocation.Location;
		}
	}

	// Move to retreat position
	MoveToDefensivePosition(OwnerComp, RetreatDestination, false, DeltaSeconds);

	// Fire while retreating (reduced accuracy)
	EngageThreats(OwnerComp, 0.5f, DefensiveFireRateMultiplier * 0.6f);

	// Update progress based on distance from threats
	float AverageDistanceFromThreats = 0.0f;
	if (Threats.Num() > 0)
	{
		for (AActor* Threat : Threats)
		{
			if (Threat)
			{
				AverageDistanceFromThreats += FVector::Dist(CurrentLocation, Threat->GetActorLocation());
			}
		}
		AverageDistanceFromThreats /= Threats.Num();
	}

	float Progress = FMath::Clamp(AverageDistanceFromThreats / MinSafeDistance, 0.0f, 1.0f);
	UpdateActionProgress(OwnerComp, Progress);
}

FVector UBTTask_ExecuteDefend::GetDefendLocation(UBehaviorTreeComponent& OwnerComp) const
{
	UBlackboardComponent* Blackboard = OwnerComp.GetBlackboardComponent();
	if (Blackboard)
	{
		FVector Location = Blackboard->GetValueAsVector(DefendLocationKey.SelectedKeyName);
		if (!Location.IsNearlyZero())
		{
			return Location;
		}
	}

	// Fallback to current position
	APawn* ControlledPawn = OwnerComp.GetAIOwner()->GetPawn();
	if (ControlledPawn)
	{
		return ControlledPawn->GetActorLocation();
	}

	return FVector::ZeroVector;
}

AActor* UBTTask_ExecuteDefend::FindNearestCover(UBehaviorTreeComponent& OwnerComp, const FVector& FromLocation) const
{
	UWorld* World = OwnerComp.GetWorld();
	if (!World)
	{
		return nullptr;
	}

	// Find actors with "Cover" tag
	TArray<AActor*> CoverActors;
	UGameplayStatics::GetAllActorsWithTag(World, FName("Cover"), CoverActors);

	AActor* NearestCover = nullptr;
	float NearestDistance = CoverSearchRadius;

	for (AActor* CoverActor : CoverActors)
	{
		if (CoverActor)
		{
			float Distance = FVector::Dist(FromLocation, CoverActor->GetActorLocation());
			if (Distance < NearestDistance)
			{
				// Check if cover is within defensive radius
				if (IsWithinDefensiveRadius(OwnerComp, CoverActor->GetActorLocation()))
				{
					NearestDistance = Distance;
					NearestCover = CoverActor;
				}
			}
		}
	}

	return NearestCover;
}

bool UBTTask_ExecuteDefend::IsWithinDefensiveRadius(UBehaviorTreeComponent& OwnerComp, const FVector& Position) const
{
	FVector DefendLocation = GetDefendLocation(OwnerComp);
	float Distance = FVector::Dist(Position, DefendLocation);
	return Distance <= MaxDefendRadius;
}

TArray<AActor*> UBTTask_ExecuteDefend::GetThreatActors(UBehaviorTreeComponent& OwnerComp) const
{
	TArray<AActor*> Threats;

	UBlackboardComponent* Blackboard = OwnerComp.GetBlackboardComponent();
	if (Blackboard)
	{
		// TODO: Implement proper threat array retrieval from blackboard
		// For now, find enemies by tag
		UWorld* World = OwnerComp.GetWorld();
		if (World)
		{
			TArray<AActor*> EnemyActors;
			UGameplayStatics::GetAllActorsWithTag(World, FName("Enemy"), EnemyActors);

			APawn* ControlledPawn = OwnerComp.GetAIOwner()->GetPawn();
			if (ControlledPawn)
			{
				// Filter to nearby threats
				for (AActor* Enemy : EnemyActors)
				{
					if (Enemy)
					{
						float Distance = FVector::Dist(ControlledPawn->GetActorLocation(), Enemy->GetActorLocation());
						if (Distance < MinSafeDistance * 2.0f)
						{
							Threats.Add(Enemy);
						}
					}
				}
			}
		}
	}

	return Threats;
}

AActor* UBTTask_ExecuteDefend::GetNearestThreat(UBehaviorTreeComponent& OwnerComp) const
{
	TArray<AActor*> Threats = GetThreatActors(OwnerComp);
	if (Threats.Num() == 0)
	{
		return nullptr;
	}

	APawn* ControlledPawn = OwnerComp.GetAIOwner()->GetPawn();
	if (!ControlledPawn)
	{
		return nullptr;
	}

	AActor* NearestThreat = nullptr;
	float NearestDistance = MAX_FLT;

	for (AActor* Threat : Threats)
	{
		if (Threat)
		{
			float Distance = FVector::Dist(ControlledPawn->GetActorLocation(), Threat->GetActorLocation());
			if (Distance < NearestDistance)
			{
				NearestDistance = Distance;
				NearestThreat = Threat;
			}
		}
	}

	return NearestThreat;
}

float UBTTask_ExecuteDefend::CalculateTacticalReward(UBehaviorTreeComponent& OwnerComp, const FBTExecuteDefendMemory* Memory) const
{
	float Reward = 0.0f;

	// Reward for time in defensive position
	Reward += (Memory->TimeInDefensivePosition / 10.0f) * PositionHeldReward;

	// Reward for using cover
	if (Memory->bHasCover)
	{
		Reward += CoverUsageReward;
	}

	// Reward for survival (no damage or minimal damage)
	if (Memory->DamageTaken == 0)
	{
		Reward += SurvivalReward;
	}
	else if (Memory->DamageTaken < 3)
	{
		Reward += SurvivalReward * 0.5f;
	}

	// Bonus for blocking shots with cover
	Reward += Memory->ShotsBlockedByCover * 1.0f;

	// Check if position was abandoned
	APawn* ControlledPawn = OwnerComp.GetAIOwner()->GetPawn();
	if (ControlledPawn && !IsWithinDefensiveRadius(OwnerComp, ControlledPawn->GetActorLocation()))
	{
		Reward += PositionAbandonedPenalty;
	}

	return Reward;
}

void UBTTask_ExecuteDefend::UpdateActionProgress(UBehaviorTreeComponent& OwnerComp, float Progress) const
{
	UBlackboardComponent* Blackboard = OwnerComp.GetBlackboardComponent();
	if (Blackboard)
	{
		Blackboard->SetValueAsFloat(ActionProgressKey.SelectedKeyName, Progress);
	}
}

void UBTTask_ExecuteDefend::DrawDebugInfo(UBehaviorTreeComponent& OwnerComp, const FBTExecuteDefendMemory* Memory) const
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
	FString TacticText = FString::Printf(TEXT("Defend Tactic:\n%s\nTime: %.1fs\nCover: %s"),
		*URLPolicyNetwork::GetActionName(Memory->CurrentTactic),
		Memory->TimeInCurrentTactic,
		Memory->bHasCover ? TEXT("Yes") : TEXT("No"));
	DrawDebugString(World, PawnLocation + FVector(0, 0, 150), TacticText, nullptr, FColor::Blue, 0.0f, true);

	// Draw defensive radius
	DrawDebugCircle(World, Memory->DefendPosition, MaxDefendRadius, 32, FColor::Green, false, 0.0f, 0, 2.0f, FVector(0, 1, 0), FVector(1, 0, 0), false);

	// Draw defend position
	DrawDebugSphere(World, Memory->DefendPosition, 100.0f, 12, FColor::Green, false, 0.0f, 0, 2.0f);

	// Draw cover
	if (Memory->CurrentCover)
	{
		DrawDebugLine(World, PawnLocation, Memory->CurrentCover->GetActorLocation(), FColor::Cyan, false, 0.0f, 0, 2.0f);
		DrawDebugSphere(World, Memory->CurrentCover->GetActorLocation(), 80.0f, 12, FColor::Cyan, false, 0.0f, 0, 2.0f);
	}

	// Draw threats
	TArray<AActor*> Threats = GetThreatActors(OwnerComp);
	for (AActor* Threat : Threats)
	{
		if (Threat)
		{
			DrawDebugLine(World, PawnLocation, Threat->GetActorLocation(), FColor::Red, false, 0.0f, 0, 1.0f);
		}
	}
}

bool UBTTask_ExecuteDefend::ShouldCompleteDefense(UBehaviorTreeComponent& OwnerComp, const FBTExecuteDefendMemory* Memory) const
{
	// Check if command has changed
	UFollowerAgentComponent* FollowerComp = GetFollowerComponent(OwnerComp);
	if (FollowerComp)
	{
		EFollowerState CurrentState = FollowerComp->GetCurrentState();
		if (CurrentState != EFollowerState::Defend)
		{
			return true;
		}
	}

	// Check if defensive hold duration met (for DefensiveHold tactic)
	if (Memory->CurrentTactic == ETacticalAction::DefensiveHold)
	{
		if (Memory->TimeInDefensivePosition >= DefensiveHoldDuration)
		{
			return true;
		}
	}

	return false;
}

void UBTTask_ExecuteDefend::MoveToDefensivePosition(UBehaviorTreeComponent& OwnerComp, const FVector& Destination, bool bUseCover, float DeltaSeconds)
{
	AAIController* AIController = OwnerComp.GetAIOwner();
	if (!AIController)
	{
		return;
	}

	// Use AI movement with defensive posture
	AIController->MoveToLocation(Destination, 50.0f, true, true, false, true, 0, true);

	// Adjust movement speed (slower when using cover)
	ACharacter* Character = Cast<ACharacter>(AIController->GetPawn());
	if (Character && Character->GetCharacterMovement())
	{
		UCharacterMovementComponent* Movement = Character->GetCharacterMovement();
		float BaseSpeed = Movement->GetMaxSpeed();
		float SpeedMultiplier = bUseCover ? 0.7f : 0.9f;
		Movement->MaxWalkSpeed = BaseSpeed * SpeedMultiplier;
	}
}

void UBTTask_ExecuteDefend::EngageThreats(UBehaviorTreeComponent& OwnerComp, float AccuracyModifier, float FireRateMultiplier)
{
	TArray<AActor*> Threats = GetThreatActors(OwnerComp);
	if (Threats.Num() == 0)
	{
		return;
	}

	AActor* PriorityThreat = GetNearestThreat(OwnerComp);
	if (!PriorityThreat)
	{
		return;
	}

	AAIController* AIController = OwnerComp.GetAIOwner();
	if (!AIController)
	{
		return;
	}

	// Set focus on threat
	AIController->SetFocus(PriorityThreat);

	// TODO: Implement actual weapon firing logic
	// This would typically call a weapon component or fire function

	// Update blackboard with fire command parameters
	UBlackboardComponent* Blackboard = OwnerComp.GetBlackboardComponent();
	if (Blackboard)
	{
		Blackboard->SetValueAsObject("FireTarget", PriorityThreat);
		Blackboard->SetValueAsFloat("FireAccuracy", AccuracyModifier);
		Blackboard->SetValueAsFloat("FireRate", FireRateMultiplier);
	}
}

void UBTTask_ExecuteDefend::UpdateCoverStatus(UBehaviorTreeComponent& OwnerComp, FBTExecuteDefendMemory* Memory)
{
	APawn* ControlledPawn = OwnerComp.GetAIOwner()->GetPawn();
	if (!ControlledPawn || !Memory->CurrentCover)
	{
		Memory->bHasCover = false;
		return;
	}

	// Check if still near cover
	float DistanceToCover = FVector::Dist(ControlledPawn->GetActorLocation(), Memory->CurrentCover->GetActorLocation());
	Memory->bHasCover = (DistanceToCover < 150.0f);

	// Update blackboard
	UBlackboardComponent* Blackboard = OwnerComp.GetBlackboardComponent();
	if (Blackboard)
	{
		Blackboard->SetValueAsObject(CoverActorKey.SelectedKeyName, Memory->bHasCover ? Memory->CurrentCover : nullptr);
	}
}
