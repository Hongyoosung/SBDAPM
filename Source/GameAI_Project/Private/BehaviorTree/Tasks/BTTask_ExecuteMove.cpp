// Copyright Epic Games, Inc. All Rights Reserved.

#include "BehaviorTree/Tasks/BTTask_ExecuteMove.h"
#include "Team/FollowerAgentComponent.h"
#include "RL/RLPolicyNetwork.h"
#include "BehaviorTree/BlackboardComponent.h"
#include "BehaviorTree/BehaviorTreeComponent.h"
#include "AIController.h"
#include "GameFramework/Character.h"
#include "GameFramework/CharacterMovementComponent.h"
#include "DrawDebugHelpers.h"
#include "Kismet/GameplayStatics.h"

UBTTask_ExecuteMove::UBTTask_ExecuteMove()
{
	NodeName = "Execute Move";
	bNotifyTick = true;
	bNotifyTaskFinished = true;

	// Set default blackboard keys
	CurrentCommandKey.SelectedKeyName = "CurrentCommand";
	MoveDestinationKey.SelectedKeyName = "MoveDestination";
	PatrolPointsKey.SelectedKeyName = "PatrolPoints";
	TacticalActionKey.SelectedKeyName = "TacticalAction";
	ActionProgressKey.SelectedKeyName = "ActionProgress";
}

EBTNodeResult::Type UBTTask_ExecuteMove::ExecuteTask(UBehaviorTreeComponent& OwnerComp, uint8* NodeMemory)
{
	FBTExecuteMoveMemory* Memory = CastInstanceNodeMemory<FBTExecuteMoveMemory>(NodeMemory);
	if (!Memory)
	{
		return EBTNodeResult::Failed;
	}

	// Get follower component
	UFollowerAgentComponent* FollowerComp = GetFollowerComponent(OwnerComp);
	if (!FollowerComp)
	{
		UE_LOG(LogTemp, Error, TEXT("BTTask_ExecuteMove: No FollowerAgentComponent found"));
		return EBTNodeResult::Failed;
	}

	// Initialize memory
	Memory->CurrentTactic = ETacticalAction::Sprint;
	Memory->TimeInCurrentTactic = 0.0f;
	Memory->TimeSinceLastRLQuery = 0.0f;
	Memory->CurrentPatrolIndex = 0;
	Memory->TimeAtPatrolPoint = 0.0f;
	Memory->bIsPatrolling = false;
	Memory->bReachedDestination = false;
	Memory->bEnemyDetected = false;
	Memory->DamageTaken = 0;
	Memory->TimeStuck = 0.0f;

	// Get movement destination
	Memory->Destination = GetMoveDestination(OwnerComp);
	if (Memory->Destination.IsNearlyZero())
	{
		UE_LOG(LogTemp, Warning, TEXT("BTTask_ExecuteMove: No valid destination"));
		return EBTNodeResult::Failed;
	}

	// Check if this is a patrol command
	TArray<FVector> PatrolPoints = GetPatrolPoints(OwnerComp);
	Memory->bIsPatrolling = (PatrolPoints.Num() > 0);

	// Get start location
	APawn* ControlledPawn = OwnerComp.GetAIOwner()->GetPawn();
	if (ControlledPawn)
	{
		Memory->StartLocation = ControlledPawn->GetActorLocation();
		Memory->LastPosition = Memory->StartLocation;
		Memory->InitialDistance = FVector::Dist(Memory->StartLocation, Memory->Destination);
		Memory->LastDistanceToDestination = Memory->InitialDistance;
	}

	// Query initial tactical action from RL policy
	Memory->CurrentTactic = QueryTacticalAction(OwnerComp);

	if (bLogActions)
	{
		UE_LOG(LogTemp, Log, TEXT("BTTask_ExecuteMove: Starting movement with tactic '%s' to destination %.0f units away"),
			*URLPolicyNetwork::GetActionName(Memory->CurrentTactic), Memory->InitialDistance);
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

EBTNodeResult::Type UBTTask_ExecuteMove::AbortTask(UBehaviorTreeComponent& OwnerComp, uint8* NodeMemory)
{
	FBTExecuteMoveMemory* Memory = CastInstanceNodeMemory<FBTExecuteMoveMemory>(NodeMemory);
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
		UE_LOG(LogTemp, Warning, TEXT("BTTask_ExecuteMove: Movement aborted after %.1f seconds"),
			Memory->TimeInCurrentTactic);
	}

	return EBTNodeResult::Aborted;
}

void UBTTask_ExecuteMove::TickTask(UBehaviorTreeComponent& OwnerComp, uint8* NodeMemory, float DeltaSeconds)
{
	FBTExecuteMoveMemory* Memory = CastInstanceNodeMemory<FBTExecuteMoveMemory>(NodeMemory);
	if (!Memory)
	{
		FinishLatentTask(OwnerComp, EBTNodeResult::Failed);
		return;
	}

	Memory->TimeInCurrentTactic += DeltaSeconds;
	Memory->TimeSinceLastRLQuery += DeltaSeconds;

	// Scan for enemies if enabled
	if (bScanForEnemies)
	{
		AActor* Enemy = ScanForEnemies(OwnerComp);
		if (Enemy && !Memory->bEnemyDetected)
		{
			Memory->bEnemyDetected = true;

			if (bLogActions)
			{
				UE_LOG(LogTemp, Warning, TEXT("BTTask_ExecuteMove: Enemy detected during movement!"));
			}

			// Signal to team leader
			UFollowerAgentComponent* FollowerComp = GetFollowerComponent(OwnerComp);
			if (FollowerComp)
			{
				FollowerComp->SignalEventToLeader(EStrategicEvent::EnemyEncounter, Enemy, Enemy->GetActorLocation(), 7);
			}
		}
	}

	// Check if stuck
	if (IsStuck(OwnerComp, Memory))
	{
		Memory->TimeStuck += DeltaSeconds;

		if (Memory->TimeStuck > 3.0f)
		{
			if (bLogActions)
			{
				UE_LOG(LogTemp, Warning, TEXT("BTTask_ExecuteMove: Agent appears stuck, re-querying RL policy"));
			}

			// Try a different tactic
			Memory->CurrentTactic = QueryTacticalAction(OwnerComp);
			Memory->TimeStuck = 0.0f;
		}
	}
	else
	{
		Memory->TimeStuck = 0.0f;
	}

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
				UE_LOG(LogTemp, Log, TEXT("BTTask_ExecuteMove: Switching to tactic '%s' (Reward: %.2f)"),
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

	// Update last position for stuck detection
	APawn* ControlledPawn = OwnerComp.GetAIOwner()->GetPawn();
	if (ControlledPawn)
	{
		Memory->LastPosition = ControlledPawn->GetActorLocation();
	}

	// Draw debug info
	if (bDrawDebugInfo)
	{
		DrawDebugInfo(OwnerComp, Memory);
	}

	// Check completion conditions
	if (ShouldCompleteMovement(OwnerComp, Memory))
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
			UE_LOG(LogTemp, Log, TEXT("BTTask_ExecuteMove: Movement complete (Final Reward: %.2f, Time: %.1f seconds)"),
				FinalReward, Memory->TimeInCurrentTactic);
		}

		FinishLatentTask(OwnerComp, EBTNodeResult::Succeeded);
	}
}

FString UBTTask_ExecuteMove::GetStaticDescription() const
{
	return FString::Printf(TEXT("Execute movement tactics via RL policy\nQuery Interval: %.1fs\nAcceptance Radius: %.0fcm"),
		RLQueryInterval, AcceptanceRadius);
}

UFollowerAgentComponent* UBTTask_ExecuteMove::GetFollowerComponent(UBehaviorTreeComponent& OwnerComp) const
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

ETacticalAction UBTTask_ExecuteMove::QueryTacticalAction(UBehaviorTreeComponent& OwnerComp) const
{
	UFollowerAgentComponent* FollowerComp = GetFollowerComponent(OwnerComp);
	if (!FollowerComp)
	{
		return ETacticalAction::Sprint;
	}

	return FollowerComp->QueryRLPolicy();
}

void UBTTask_ExecuteMove::ExecuteTacticalAction(UBehaviorTreeComponent& OwnerComp, FBTExecuteMoveMemory* Memory, ETacticalAction Action, float DeltaSeconds)
{
	switch (Action)
	{
	case ETacticalAction::Sprint:
		ExecuteSprint(OwnerComp, Memory, DeltaSeconds);
		break;

	case ETacticalAction::Crouch:
		ExecuteCrouch(OwnerComp, Memory, DeltaSeconds);
		break;

	case ETacticalAction::Patrol:
		ExecutePatrol(OwnerComp, Memory, DeltaSeconds);
		break;

	case ETacticalAction::Hold:
		ExecuteHold(OwnerComp, Memory, DeltaSeconds);
		break;

	default:
		// Fallback to sprint
		ExecuteSprint(OwnerComp, Memory, DeltaSeconds);
		break;
	}
}

void UBTTask_ExecuteMove::ExecuteSprint(UBehaviorTreeComponent& OwnerComp, FBTExecuteMoveMemory* Memory, float DeltaSeconds)
{
	// Move quickly to destination
	MoveToLocation(OwnerComp, Memory->Destination, SprintSpeedMultiplier, false, DeltaSeconds);

	// Update progress
	APawn* ControlledPawn = OwnerComp.GetAIOwner()->GetPawn();
	if (ControlledPawn)
	{
		float CurrentDistance = FVector::Dist(ControlledPawn->GetActorLocation(), Memory->Destination);
		Memory->LastDistanceToDestination = CurrentDistance;

		float Progress = 1.0f - FMath::Clamp(CurrentDistance / Memory->InitialDistance, 0.0f, 1.0f);
		UpdateActionProgress(OwnerComp, Progress);

		// Check if reached
		if (HasReachedDestination(OwnerComp, Memory->Destination))
		{
			Memory->bReachedDestination = true;
		}
	}
}

void UBTTask_ExecuteMove::ExecuteCrouch(UBehaviorTreeComponent& OwnerComp, FBTExecuteMoveMemory* Memory, float DeltaSeconds)
{
	// Move slowly and stealthily
	MoveToLocation(OwnerComp, Memory->Destination, CrouchSpeedMultiplier, true, DeltaSeconds);

	// Update progress
	APawn* ControlledPawn = OwnerComp.GetAIOwner()->GetPawn();
	if (ControlledPawn)
	{
		float CurrentDistance = FVector::Dist(ControlledPawn->GetActorLocation(), Memory->Destination);
		Memory->LastDistanceToDestination = CurrentDistance;

		float Progress = 1.0f - FMath::Clamp(CurrentDistance / Memory->InitialDistance, 0.0f, 1.0f);
		UpdateActionProgress(OwnerComp, Progress);

		// Check if reached
		if (HasReachedDestination(OwnerComp, Memory->Destination))
		{
			Memory->bReachedDestination = true;
		}
	}
}

void UBTTask_ExecuteMove::ExecutePatrol(UBehaviorTreeComponent& OwnerComp, FBTExecuteMoveMemory* Memory, float DeltaSeconds)
{
	APawn* ControlledPawn = OwnerComp.GetAIOwner()->GetPawn();
	if (!ControlledPawn)
	{
		return;
	}

	// Check if at current patrol point
	if (HasReachedDestination(OwnerComp, Memory->Destination))
	{
		// Pause at patrol point
		Memory->TimeAtPatrolPoint += DeltaSeconds;

		if (Memory->TimeAtPatrolPoint >= PatrolPauseDuration)
		{
			// Move to next patrol point
			Memory->Destination = GetNextPatrolPoint(OwnerComp, Memory);
			Memory->TimeAtPatrolPoint = 0.0f;

			if (bLogActions)
			{
				UE_LOG(LogTemp, Log, TEXT("BTTask_ExecuteMove: Moving to next patrol point (%d)"), Memory->CurrentPatrolIndex);
			}
		}
	}
	else
	{
		// Move to current patrol point
		MoveToLocation(OwnerComp, Memory->Destination, PatrolSpeedMultiplier, false, DeltaSeconds);
	}

	// Update progress (based on patrol completion)
	TArray<FVector> PatrolPoints = GetPatrolPoints(OwnerComp);
	float Progress = 0.0f;
	if (PatrolPoints.Num() > 0)
	{
		Progress = static_cast<float>(Memory->CurrentPatrolIndex) / static_cast<float>(PatrolPoints.Num());
	}
	UpdateActionProgress(OwnerComp, Progress);
}

void UBTTask_ExecuteMove::ExecuteHold(UBehaviorTreeComponent& OwnerComp, FBTExecuteMoveMemory* Memory, float DeltaSeconds)
{
	APawn* ControlledPawn = OwnerComp.GetAIOwner()->GetPawn();
	if (!ControlledPawn)
	{
		return;
	}

	// Stop moving if at destination
	if (HasReachedDestination(OwnerComp, Memory->Destination))
	{
		Memory->bReachedDestination = true;

		// Hold position
		AAIController* AIController = OwnerComp.GetAIOwner();
		if (AIController)
		{
			AIController->StopMovement();
		}

		UpdateActionProgress(OwnerComp, 1.0f);
	}
	else
	{
		// Move to destination first
		MoveToLocation(OwnerComp, Memory->Destination, 1.0f, false, DeltaSeconds);

		float CurrentDistance = FVector::Dist(ControlledPawn->GetActorLocation(), Memory->Destination);
		float Progress = 1.0f - FMath::Clamp(CurrentDistance / Memory->InitialDistance, 0.0f, 1.0f);
		UpdateActionProgress(OwnerComp, Progress);
	}
}

FVector UBTTask_ExecuteMove::GetMoveDestination(UBehaviorTreeComponent& OwnerComp) const
{
	UBlackboardComponent* Blackboard = OwnerComp.GetBlackboardComponent();
	if (Blackboard)
	{
		FVector Destination = Blackboard->GetValueAsVector(MoveDestinationKey.SelectedKeyName);
		if (!Destination.IsNearlyZero())
		{
			return Destination;
		}
	}

	return FVector::ZeroVector;
}

TArray<FVector> UBTTask_ExecuteMove::GetPatrolPoints(UBehaviorTreeComponent& OwnerComp) const
{
	TArray<FVector> PatrolPoints;

	// TODO: Implement patrol points retrieval from blackboard
	// For now, return empty array

	return PatrolPoints;
}

bool UBTTask_ExecuteMove::HasReachedDestination(UBehaviorTreeComponent& OwnerComp, const FVector& Destination) const
{
	APawn* ControlledPawn = OwnerComp.GetAIOwner()->GetPawn();
	if (!ControlledPawn)
	{
		return false;
	}

	float Distance = FVector::Dist(ControlledPawn->GetActorLocation(), Destination);
	return Distance <= AcceptanceRadius;
}

AActor* UBTTask_ExecuteMove::ScanForEnemies(UBehaviorTreeComponent& OwnerComp) const
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

	// Find enemies by tag
	TArray<AActor*> EnemyActors;
	UGameplayStatics::GetAllActorsWithTag(World, FName("Enemy"), EnemyActors);

	// Find nearest enemy within detection range
	AActor* NearestEnemy = nullptr;
	float NearestDistance = EnemyDetectionRange;

	for (AActor* Enemy : EnemyActors)
	{
		if (Enemy)
		{
			float Distance = FVector::Dist(ControlledPawn->GetActorLocation(), Enemy->GetActorLocation());
			if (Distance < NearestDistance)
			{
				NearestDistance = Distance;
				NearestEnemy = Enemy;
			}
		}
	}

	return NearestEnemy;
}

bool UBTTask_ExecuteMove::IsStuck(UBehaviorTreeComponent& OwnerComp, FBTExecuteMoveMemory* Memory) const
{
	APawn* ControlledPawn = OwnerComp.GetAIOwner()->GetPawn();
	if (!ControlledPawn)
	{
		return false;
	}

	// Check if moved very little in last tick
	float DistanceMoved = FVector::Dist(ControlledPawn->GetActorLocation(), Memory->LastPosition);
	return DistanceMoved < 10.0f && !Memory->bReachedDestination;
}

float UBTTask_ExecuteMove::CalculateTacticalReward(UBehaviorTreeComponent& OwnerComp, const FBTExecuteMoveMemory* Memory) const
{
	float Reward = 0.0f;

	// Reward for reaching destination
	if (Memory->bReachedDestination)
	{
		Reward += EfficientMovementReward;

		// Bonus for fast movement
		if (Memory->TimeInCurrentTactic < 10.0f)
		{
			Reward += 2.0f;
		}
	}

	// Reward for safe movement
	if (Memory->DamageTaken == 0)
	{
		Reward += SafeMovementReward;
	}

	// Reward for detecting enemies
	if (Memory->bEnemyDetected)
	{
		Reward += EnemyDetectionReward;
	}

	// Penalty for getting stuck
	if (Memory->TimeStuck > 2.0f)
	{
		Reward += StuckPenalty;
	}

	// Efficiency reward based on distance covered
	float DistanceCovered = Memory->InitialDistance - Memory->LastDistanceToDestination;
	if (DistanceCovered > 0.0f && Memory->InitialDistance > 0.0f)
	{
		float EfficiencyRatio = DistanceCovered / Memory->InitialDistance;
		Reward += EfficiencyRatio * 3.0f;
	}

	return Reward;
}

void UBTTask_ExecuteMove::UpdateActionProgress(UBehaviorTreeComponent& OwnerComp, float Progress) const
{
	UBlackboardComponent* Blackboard = OwnerComp.GetBlackboardComponent();
	if (Blackboard)
	{
		Blackboard->SetValueAsFloat(ActionProgressKey.SelectedKeyName, Progress);
	}
}

void UBTTask_ExecuteMove::DrawDebugInfo(UBehaviorTreeComponent& OwnerComp, const FBTExecuteMoveMemory* Memory) const
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
	FString TacticText = FString::Printf(TEXT("Move Tactic:\n%s\nTime: %.1fs\nDist: %.0f"),
		*URLPolicyNetwork::GetActionName(Memory->CurrentTactic),
		Memory->TimeInCurrentTactic,
		Memory->LastDistanceToDestination);
	DrawDebugString(World, PawnLocation + FVector(0, 0, 150), TacticText, nullptr, FColor::Cyan, 0.0f, true);

	// Draw path to destination
	DrawDebugLine(World, PawnLocation, Memory->Destination, FColor::Cyan, false, 0.0f, 0, 2.0f);
	DrawDebugSphere(World, Memory->Destination, AcceptanceRadius, 12, FColor::Cyan, false, 0.0f, 0, 2.0f);

	// Draw start location
	DrawDebugSphere(World, Memory->StartLocation, 50.0f, 12, FColor::Yellow, false, 0.0f, 0, 1.0f);
}

bool UBTTask_ExecuteMove::ShouldCompleteMovement(UBehaviorTreeComponent& OwnerComp, const FBTExecuteMoveMemory* Memory) const
{
	// Check if destination reached
	if (Memory->bReachedDestination)
	{
		return true;
	}

	// Check if command has changed
	UFollowerAgentComponent* FollowerComp = GetFollowerComponent(OwnerComp);
	if (FollowerComp)
	{
		EFollowerState CurrentState = FollowerComp->GetCurrentState();
		if (CurrentState != EFollowerState::Move)
		{
			return true;
		}
	}

	// For patrol, check if patrol is complete
	if (Memory->bIsPatrolling)
	{
		TArray<FVector> PatrolPoints = GetPatrolPoints(OwnerComp);
		if (Memory->CurrentPatrolIndex >= PatrolPoints.Num())
		{
			return true;
		}
	}

	return false;
}

void UBTTask_ExecuteMove::MoveToLocation(UBehaviorTreeComponent& OwnerComp, const FVector& Destination, float SpeedMultiplier, bool bCrouched, float DeltaSeconds)
{
	AAIController* AIController = OwnerComp.GetAIOwner();
	if (!AIController)
	{
		return;
	}

	// Use AI movement
	AIController->MoveToLocation(Destination, AcceptanceRadius, true, true, false, true, 0, true);

	// Adjust movement speed and stance
	ACharacter* Character = Cast<ACharacter>(AIController->GetPawn());
	if (Character && Character->GetCharacterMovement())
	{
		UCharacterMovementComponent* Movement = Character->GetCharacterMovement();
		float BaseSpeed = Movement->GetMaxSpeed();
		Movement->MaxWalkSpeed = BaseSpeed * SpeedMultiplier;

		// Set crouched state
		if (bCrouched && Character->CanCrouch())
		{
			Character->Crouch();
		}
		else if (!bCrouched && Character->bIsCrouched)
		{
			Character->UnCrouch();
		}
	}
}

FVector UBTTask_ExecuteMove::GetNextPatrolPoint(UBehaviorTreeComponent& OwnerComp, FBTExecuteMoveMemory* Memory) const
{
	TArray<FVector> PatrolPoints = GetPatrolPoints(OwnerComp);
	if (PatrolPoints.Num() == 0)
	{
		return Memory->Destination;
	}

	// Cycle through patrol points
	Memory->CurrentPatrolIndex = (Memory->CurrentPatrolIndex + 1) % PatrolPoints.Num();
	return PatrolPoints[Memory->CurrentPatrolIndex];
}
