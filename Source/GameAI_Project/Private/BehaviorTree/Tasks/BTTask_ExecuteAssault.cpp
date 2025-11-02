// Copyright Epic Games, Inc. All Rights Reserved.

#include "BehaviorTree/Tasks/BTTask_ExecuteAssault.h"
#include "Team/FollowerAgentComponent.h"
#include "RL/RLPolicyNetwork.h"
#include "BehaviorTree/BlackboardComponent.h"
#include "BehaviorTree/BehaviorTreeComponent.h"
#include "AIController.h"
#include "GameFramework/Character.h"
#include "GameFramework/CharacterMovementComponent.h"
#include "DrawDebugHelpers.h"
#include "NavigationSystem.h"
#include "Kismet/KismetMathLibrary.h"

UBTTask_ExecuteAssault::UBTTask_ExecuteAssault()
{
	NodeName = "Execute Assault";
	bNotifyTick = true;
	bNotifyTaskFinished = true;

	// Set default blackboard keys
	CurrentCommandKey.SelectedKeyName = "CurrentCommand";
	TargetActorKey.SelectedKeyName = "TargetActor";
	TargetLocationKey.SelectedKeyName = "TargetLocation";
	TacticalActionKey.SelectedKeyName = "TacticalAction";
	ActionProgressKey.SelectedKeyName = "ActionProgress";
}

EBTNodeResult::Type UBTTask_ExecuteAssault::ExecuteTask(UBehaviorTreeComponent& OwnerComp, uint8* NodeMemory)
{
	FBTExecuteAssaultMemory* Memory = CastInstanceNodeMemory<FBTExecuteAssaultMemory>(NodeMemory);
	if (!Memory)
	{
		return EBTNodeResult::Failed;
	}

	// Get follower component
	UFollowerAgentComponent* FollowerComp = GetFollowerComponent(OwnerComp);
	if (!FollowerComp)
	{
		UE_LOG(LogTemp, Error, TEXT("BTTask_ExecuteAssault: No FollowerAgentComponent found"));
		return EBTNodeResult::Failed;
	}

	// Initialize memory
	Memory->CurrentTactic = ETacticalAction::AggressiveAssault;
	Memory->TimeInCurrentTactic = 0.0f;
	Memory->TimeSinceLastRLQuery = 0.0f;
	Memory->HitsLanded = 0;
	Memory->DamageTaken = 0;
	Memory->bHasTarget = false;

	// Get initial target information
	AActor* Target = GetTargetActor(OwnerComp);
	if (Target)
	{
		Memory->bHasTarget = true;
		APawn* ControlledPawn = OwnerComp.GetAIOwner()->GetPawn();
		if (ControlledPawn)
		{
			Memory->InitialDistanceToTarget = FVector::Dist(ControlledPawn->GetActorLocation(), Target->GetActorLocation());
			Memory->LastDistanceToTarget = Memory->InitialDistanceToTarget;
			Memory->LastPosition = ControlledPawn->GetActorLocation();
		}
	}

	// Query initial tactical action from RL policy
	Memory->CurrentTactic = QueryTacticalAction(OwnerComp);

	if (bLogActions)
	{
		UE_LOG(LogTemp, Log, TEXT("BTTask_ExecuteAssault: Starting assault with tactic '%s'"),
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

EBTNodeResult::Type UBTTask_ExecuteAssault::AbortTask(UBehaviorTreeComponent& OwnerComp, uint8* NodeMemory)
{
	FBTExecuteAssaultMemory* Memory = CastInstanceNodeMemory<FBTExecuteAssaultMemory>(NodeMemory);
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
		UE_LOG(LogTemp, Warning, TEXT("BTTask_ExecuteAssault: Assault aborted after %.1f seconds"),
			Memory->TimeInCurrentTactic);
	}

	return EBTNodeResult::Aborted;
}

void UBTTask_ExecuteAssault::TickTask(UBehaviorTreeComponent& OwnerComp, uint8* NodeMemory, float DeltaSeconds)
{
	FBTExecuteAssaultMemory* Memory = CastInstanceNodeMemory<FBTExecuteAssaultMemory>(NodeMemory);
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
				UE_LOG(LogTemp, Log, TEXT("BTTask_ExecuteAssault: Switching to tactic '%s' (Reward: %.2f)"),
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
	if (ShouldCompleteAssault(OwnerComp, Memory))
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
			UE_LOG(LogTemp, Log, TEXT("BTTask_ExecuteAssault: Assault complete (Final Reward: %.2f, Time: %.1f seconds)"),
				FinalReward, Memory->TimeInCurrentTactic);
		}

		FinishLatentTask(OwnerComp, EBTNodeResult::Succeeded);
	}
}

FString UBTTask_ExecuteAssault::GetStaticDescription() const
{
	return FString::Printf(TEXT("Execute assault tactics via RL policy\nQuery Interval: %.1fs\nEngagement Range: %.0fcm"),
		RLQueryInterval, OptimalEngagementDistance);
}

UFollowerAgentComponent* UBTTask_ExecuteAssault::GetFollowerComponent(UBehaviorTreeComponent& OwnerComp) const
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

ETacticalAction UBTTask_ExecuteAssault::QueryTacticalAction(UBehaviorTreeComponent& OwnerComp) const
{
	UFollowerAgentComponent* FollowerComp = GetFollowerComponent(OwnerComp);
	if (!FollowerComp)
	{
		return ETacticalAction::AggressiveAssault;
	}

	return FollowerComp->QueryRLPolicy();
}

void UBTTask_ExecuteAssault::ExecuteTacticalAction(UBehaviorTreeComponent& OwnerComp, FBTExecuteAssaultMemory* Memory, ETacticalAction Action, float DeltaSeconds)
{
	switch (Action)
	{
	case ETacticalAction::AggressiveAssault:
		ExecuteAggressiveAssault(OwnerComp, Memory, DeltaSeconds);
		break;

	case ETacticalAction::CautiousAdvance:
		ExecuteCautiousAdvance(OwnerComp, Memory, DeltaSeconds);
		break;

	case ETacticalAction::FlankLeft:
		ExecuteFlankingManeuver(OwnerComp, Memory, true, DeltaSeconds);
		break;

	case ETacticalAction::FlankRight:
		ExecuteFlankingManeuver(OwnerComp, Memory, false, DeltaSeconds);
		break;

	case ETacticalAction::SuppressiveFire:
		ExecuteSuppressiveFire(OwnerComp, Memory, DeltaSeconds);
		break;

	case ETacticalAction::DefensiveHold:
	case ETacticalAction::TacticalRetreat:
		// Defensive actions in assault state - use cautious advance
		ExecuteCautiousAdvance(OwnerComp, Memory, DeltaSeconds);
		break;

	default:
		// Fallback to aggressive assault
		ExecuteAggressiveAssault(OwnerComp, Memory, DeltaSeconds);
		break;
	}
}

void UBTTask_ExecuteAssault::ExecuteAggressiveAssault(UBehaviorTreeComponent& OwnerComp, FBTExecuteAssaultMemory* Memory, float DeltaSeconds)
{
	AActor* Target = GetTargetActor(OwnerComp);
	if (!Target)
	{
		return;
	}

	APawn* ControlledPawn = OwnerComp.GetAIOwner()->GetPawn();
	if (!ControlledPawn)
	{
		return;
	}

	float DistanceToTarget = FVector::Dist(ControlledPawn->GetActorLocation(), Target->GetActorLocation());
	Memory->LastDistanceToTarget = DistanceToTarget;

	// Move aggressively toward target
	if (DistanceToTarget > OptimalEngagementDistance * 0.7f)
	{
		MoveTowardTarget(OwnerComp, Target->GetActorLocation(), AssaultSpeedMultiplier, DeltaSeconds);
	}

	// Fire at target with high rate
	FireAtTarget(OwnerComp, Target, 0.8f, FireRateMultiplier);

	// Update progress
	float Progress = 1.0f - FMath::Clamp(DistanceToTarget / Memory->InitialDistanceToTarget, 0.0f, 1.0f);
	UpdateActionProgress(OwnerComp, Progress);
}

void UBTTask_ExecuteAssault::ExecuteCautiousAdvance(UBehaviorTreeComponent& OwnerComp, FBTExecuteAssaultMemory* Memory, float DeltaSeconds)
{
	AActor* Target = GetTargetActor(OwnerComp);
	if (!Target)
	{
		return;
	}

	APawn* ControlledPawn = OwnerComp.GetAIOwner()->GetPawn();
	if (!ControlledPawn)
	{
		return;
	}

	float DistanceToTarget = FVector::Dist(ControlledPawn->GetActorLocation(), Target->GetActorLocation());
	Memory->LastDistanceToTarget = DistanceToTarget;

	// Move cautiously, maintaining ideal distance
	if (DistanceToTarget > OptimalEngagementDistance * 1.2f)
	{
		// Too far, advance slowly
		MoveTowardTarget(OwnerComp, Target->GetActorLocation(), AssaultSpeedMultiplier * 0.6f, DeltaSeconds);
	}
	else if (DistanceToTarget < OptimalEngagementDistance * 0.8f)
	{
		// Too close, back off
		FVector BackOffDirection = (ControlledPawn->GetActorLocation() - Target->GetActorLocation()).GetSafeNormal();
		FVector BackOffLocation = ControlledPawn->GetActorLocation() + BackOffDirection * 200.0f;
		MoveTowardTarget(OwnerComp, BackOffLocation, 0.5f, DeltaSeconds);
	}

	// Fire with normal accuracy
	FireAtTarget(OwnerComp, Target, 1.0f, FireRateMultiplier * 0.8f);

	// Update progress
	float Progress = FMath::Clamp(1.0f - FMath::Abs(DistanceToTarget - OptimalEngagementDistance) / OptimalEngagementDistance, 0.0f, 1.0f);
	UpdateActionProgress(OwnerComp, Progress);
}

void UBTTask_ExecuteAssault::ExecuteFlankingManeuver(UBehaviorTreeComponent& OwnerComp, FBTExecuteAssaultMemory* Memory, bool bFlankLeft, float DeltaSeconds)
{
	AActor* Target = GetTargetActor(OwnerComp);
	if (!Target)
	{
		return;
	}

	APawn* ControlledPawn = OwnerComp.GetAIOwner()->GetPawn();
	if (!ControlledPawn)
	{
		return;
	}

	// Calculate flanking destination if not set
	if (Memory->FlankDestination.IsNearlyZero() || Memory->TimeInCurrentTactic < 0.1f)
	{
		Memory->FlankDestination = CalculateFlankingPosition(OwnerComp, Target, bFlankLeft);
	}

	float DistanceToFlankPosition = FVector::Dist(ControlledPawn->GetActorLocation(), Memory->FlankDestination);

	// Move to flanking position
	if (DistanceToFlankPosition > 100.0f)
	{
		MoveTowardTarget(OwnerComp, Memory->FlankDestination, AssaultSpeedMultiplier * 0.9f, DeltaSeconds);
	}

	// Fire while flanking
	float DistanceToTarget = FVector::Dist(ControlledPawn->GetActorLocation(), Target->GetActorLocation());
	if (DistanceToTarget < MaxPursuitDistance)
	{
		FireAtTarget(OwnerComp, Target, 0.7f, FireRateMultiplier * 0.7f);
	}

	Memory->LastDistanceToTarget = DistanceToTarget;

	// Update progress
	float Progress = 1.0f - FMath::Clamp(DistanceToFlankPosition / FlankingDistance, 0.0f, 1.0f);
	UpdateActionProgress(OwnerComp, Progress);
}

void UBTTask_ExecuteAssault::ExecuteSuppressiveFire(UBehaviorTreeComponent& OwnerComp, FBTExecuteAssaultMemory* Memory, float DeltaSeconds)
{
	AActor* Target = GetTargetActor(OwnerComp);
	if (!Target)
	{
		return;
	}

	APawn* ControlledPawn = OwnerComp.GetAIOwner()->GetPawn();
	if (!ControlledPawn)
	{
		return;
	}

	float DistanceToTarget = FVector::Dist(ControlledPawn->GetActorLocation(), Target->GetActorLocation());
	Memory->LastDistanceToTarget = DistanceToTarget;

	// Stay at current position or move to better position
	if (DistanceToTarget > OptimalEngagementDistance * 1.5f)
	{
		MoveTowardTarget(OwnerComp, Target->GetActorLocation(), AssaultSpeedMultiplier * 0.4f, DeltaSeconds);
	}

	// Fire with high rate but low accuracy (suppression)
	FireAtTarget(OwnerComp, Target, SuppressiveAccuracyModifier, FireRateMultiplier * 2.0f);

	// Update progress based on time
	float Progress = FMath::Clamp(Memory->TimeInCurrentTactic / 5.0f, 0.0f, 1.0f);
	UpdateActionProgress(OwnerComp, Progress);
}

AActor* UBTTask_ExecuteAssault::GetTargetActor(UBehaviorTreeComponent& OwnerComp) const
{
	UBlackboardComponent* Blackboard = OwnerComp.GetBlackboardComponent();
	if (!Blackboard)
	{
		return nullptr;
	}

	return Cast<AActor>(Blackboard->GetValueAsObject(TargetActorKey.SelectedKeyName));
}

FVector UBTTask_ExecuteAssault::GetTargetLocation(UBehaviorTreeComponent& OwnerComp) const
{
	UBlackboardComponent* Blackboard = OwnerComp.GetBlackboardComponent();
	if (!Blackboard)
	{
		return FVector::ZeroVector;
	}

	return Blackboard->GetValueAsVector(TargetLocationKey.SelectedKeyName);
}

float UBTTask_ExecuteAssault::CalculateTacticalReward(UBehaviorTreeComponent& OwnerComp, const FBTExecuteAssaultMemory* Memory) const
{
	float Reward = 0.0f;

	// Reward for closing distance
	if (Memory->InitialDistanceToTarget > 0.0f)
	{
		float DistanceClosed = Memory->InitialDistanceToTarget - Memory->LastDistanceToTarget;
		if (DistanceClosed > 0.0f)
		{
			Reward += (DistanceClosed / 100.0f) * ClosingDistanceReward;
		}
	}

	// Reward for hits landed
	Reward += Memory->HitsLanded * CombatHitReward;

	// Penalty for damage taken
	Reward += Memory->DamageTaken * DamageTakenPenalty;

	// Time efficiency bonus (faster = better)
	if (Memory->TimeInCurrentTactic < 5.0f)
	{
		Reward += 1.0f;
	}

	return Reward;
}

void UBTTask_ExecuteAssault::UpdateActionProgress(UBehaviorTreeComponent& OwnerComp, float Progress) const
{
	UBlackboardComponent* Blackboard = OwnerComp.GetBlackboardComponent();
	if (Blackboard)
	{
		Blackboard->SetValueAsFloat(ActionProgressKey.SelectedKeyName, Progress);
	}
}

void UBTTask_ExecuteAssault::DrawDebugInfo(UBehaviorTreeComponent& OwnerComp, const FBTExecuteAssaultMemory* Memory) const
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
	FString TacticText = FString::Printf(TEXT("Assault Tactic:\n%s\nTime: %.1fs"),
		*URLPolicyNetwork::GetActionName(Memory->CurrentTactic),
		Memory->TimeInCurrentTactic);
	DrawDebugString(World, PawnLocation + FVector(0, 0, 150), TacticText, nullptr, FColor::Red, 0.0f, true);

	// Draw target line
	AActor* Target = GetTargetActor(OwnerComp);
	if (Target)
	{
		DrawDebugLine(World, PawnLocation, Target->GetActorLocation(), FColor::Red, false, 0.0f, 0, 2.0f);
		DrawDebugSphere(World, Target->GetActorLocation(), 50.0f, 12, FColor::Red, false, 0.0f, 0, 2.0f);
	}

	// Draw flanking destination
	if (!Memory->FlankDestination.IsNearlyZero())
	{
		DrawDebugSphere(World, Memory->FlankDestination, 100.0f, 12, FColor::Orange, false, 0.0f, 0, 2.0f);
		DrawDebugLine(World, PawnLocation, Memory->FlankDestination, FColor::Orange, false, 0.0f, 0, 1.0f);
	}
}

bool UBTTask_ExecuteAssault::ShouldCompleteAssault(UBehaviorTreeComponent& OwnerComp, const FBTExecuteAssaultMemory* Memory) const
{
	// Check if target is lost
	AActor* Target = GetTargetActor(OwnerComp);
	if (!Target || !Target->IsValidLowLevel())
	{
		return true;
	}

	// Check if target is too far away
	APawn* ControlledPawn = OwnerComp.GetAIOwner()->GetPawn();
	if (ControlledPawn)
	{
		float DistanceToTarget = FVector::Dist(ControlledPawn->GetActorLocation(), Target->GetActorLocation());
		if (DistanceToTarget > MaxPursuitDistance)
		{
			return true;
		}
	}

	// Check if command has changed
	UFollowerAgentComponent* FollowerComp = GetFollowerComponent(OwnerComp);
	if (FollowerComp)
	{
		EFollowerState CurrentState = FollowerComp->GetCurrentState();
		if (CurrentState != EFollowerState::Assault)
		{
			return true;
		}
	}

	return false;
}

void UBTTask_ExecuteAssault::MoveTowardTarget(UBehaviorTreeComponent& OwnerComp, const FVector& Destination, float SpeedMultiplier, float DeltaSeconds)
{
	AAIController* AIController = OwnerComp.GetAIOwner();
	if (!AIController)
	{
		return;
	}

	// Use AI movement
	AIController->MoveToLocation(Destination, 50.0f, true, true, false, true, 0, true);

	// Adjust movement speed
	ACharacter* Character = Cast<ACharacter>(AIController->GetPawn());
	if (Character && Character->GetCharacterMovement())
	{
		UCharacterMovementComponent* Movement = Character->GetCharacterMovement();
		float BaseSpeed = Movement->GetMaxSpeed();
		Movement->MaxWalkSpeed = BaseSpeed * SpeedMultiplier;
	}
}

void UBTTask_ExecuteAssault::FireAtTarget(UBehaviorTreeComponent& OwnerComp, AActor* Target, float AccuracyModifier, float InFireRateMultiplier)
{
	if (!Target)
	{
		return;
	}

	AAIController* AIController = OwnerComp.GetAIOwner();
	if (!AIController || !AIController->GetPawn())
	{
		return;
	}

	// Set focus on target
	AIController->SetFocus(Target);

	// TODO: Implement actual weapon firing logic
	// This would typically call a weapon component or fire function
	// For now, this is a placeholder that sets up the AI to face the target

	// Update blackboard with fire command parameters
	UBlackboardComponent* Blackboard = OwnerComp.GetBlackboardComponent();
	if (Blackboard)
	{
		Blackboard->SetValueAsObject("FireTarget", Target);
		Blackboard->SetValueAsFloat("FireAccuracy", AccuracyModifier);
		Blackboard->SetValueAsFloat("FireRate", InFireRateMultiplier);
	}
}

FVector UBTTask_ExecuteAssault::CalculateFlankingPosition(UBehaviorTreeComponent& OwnerComp, AActor* Target, bool bFlankLeft) const
{
	if (!Target)
	{
		return FVector::ZeroVector;
	}

	AAIController* AIController = OwnerComp.GetAIOwner();
	if (!AIController || !AIController->GetPawn())
	{
		return FVector::ZeroVector;
	}

	APawn* ControlledPawn = AIController->GetPawn();
	FVector PawnLocation = ControlledPawn->GetActorLocation();
	FVector TargetLocation = Target->GetActorLocation();

	// Calculate perpendicular vector to target
	FVector ToTarget = (TargetLocation - PawnLocation).GetSafeNormal2D();
	FVector FlankDirection = FVector::CrossProduct(ToTarget, FVector::UpVector);

	if (!bFlankLeft)
	{
		FlankDirection *= -1.0f;
	}

	// Calculate flanking position
	FVector FlankPosition = PawnLocation + (FlankDirection * FlankingDistance);

	// Try to find navigable position
	UNavigationSystemV1* NavSys = UNavigationSystemV1::GetCurrent(AIController->GetWorld());
	if (NavSys)
	{
		FNavLocation NavLocation;
		if (NavSys->ProjectPointToNavigation(FlankPosition, NavLocation, FVector(500.0f, 500.0f, 500.0f)))
		{
			return NavLocation.Location;
		}
	}

	return FlankPosition;
}
