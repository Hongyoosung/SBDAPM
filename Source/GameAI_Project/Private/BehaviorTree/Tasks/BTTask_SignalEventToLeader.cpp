// Copyright Epic Games, Inc. All Rights Reserved.

#include "BehaviorTree/Tasks/BTTask_SignalEventToLeader.h"
#include "BehaviorTree/BlackboardComponent.h"
#include "AIController.h"
#include "Team/FollowerAgentComponent.h"
#include "Team/TeamLeaderComponent.h"
#include "GameFramework/Pawn.h"

UBTTask_SignalEventToLeader::UBTTask_SignalEventToLeader()
{
	NodeName = "Signal Event to Leader";
	bNotifyTick = false;
}

EBTNodeResult::Type UBTTask_SignalEventToLeader::ExecuteTask(UBehaviorTreeComponent& OwnerComp, uint8* NodeMemory)
{
	// Get AI controller and pawn
	AAIController* AIController = OwnerComp.GetAIOwner();
	if (!AIController)
	{
		UE_LOG(LogTemp, Error, TEXT("BTTask_SignalEventToLeader: No AIController found"));
		return EBTNodeResult::Failed;
	}

	APawn* ControlledPawn = AIController->GetPawn();
	if (!ControlledPawn)
	{
		UE_LOG(LogTemp, Error, TEXT("BTTask_SignalEventToLeader: No controlled pawn"));
		return EBTNodeResult::Failed;
	}

	// Get FollowerAgentComponent
	UFollowerAgentComponent* FollowerComp = ControlledPawn->FindComponentByClass<UFollowerAgentComponent>();
	if (!FollowerComp)
	{
		UE_LOG(LogTemp, Error, TEXT("BTTask_SignalEventToLeader: No FollowerAgentComponent on pawn %s"),
			*ControlledPawn->GetName());
		return EBTNodeResult::Failed;
	}

	// Check if follower has a team leader
	if (!FollowerComp->GetTeamLeader())
	{
		if (bLogSignals)
		{
			UE_LOG(LogTemp, Warning, TEXT("BTTask_SignalEventToLeader: Follower %s has no team leader assigned"),
				*ControlledPawn->GetName());
		}
		return EBTNodeResult::Failed;
	}

	// Check if leader is busy (if bOnlySignalIfLeaderIdle is true)
	if (bOnlySignalIfLeaderIdle)
	{
		UTeamLeaderComponent* Leader = FollowerComp->GetTeamLeader();
		if (Leader && Leader->IsRunningMCTS())
		{
			if (bLogSignals)
			{
				UE_LOG(LogTemp, Verbose, TEXT("BTTask_SignalEventToLeader: Leader is busy, skipping signal"));
			}
			return EBTNodeResult::Succeeded; // Not a failure, just skip
		}
	}

	// Check minimum signal interval
	double CurrentTime = FPlatformTime::Seconds();
	if (LastSignalTimes.Contains(FollowerComp))
	{
		double TimeSinceLastSignal = CurrentTime - LastSignalTimes[FollowerComp];
		if (TimeSinceLastSignal < MinSignalInterval)
		{
			if (bLogSignals)
			{
				UE_LOG(LogTemp, Verbose, TEXT("BTTask_SignalEventToLeader: Signal interval not met (%.2fs < %.2fs)"),
					TimeSinceLastSignal, MinSignalInterval);
			}
			return EBTNodeResult::Succeeded; // Not a failure, just throttled
		}
	}

	// Get target actor from blackboard (optional)
	AActor* TargetActor = nullptr;
	if (TargetActorKey.SelectedKeyName != NAME_None)
	{
		UBlackboardComponent* BlackboardComp = OwnerComp.GetBlackboardComponent();
		if (BlackboardComp)
		{
			TargetActor = Cast<AActor>(BlackboardComp->GetValueAsObject(TargetActorKey.SelectedKeyName));
		}
	}

	// Signal event to leader
	FollowerComp->SignalEventToLeader(EventType, TargetActor);

	// Update last signal time
	LastSignalTimes.Add(FollowerComp, CurrentTime);

	// Log signal
	if (bLogSignals)
	{
		FString EventName = UEnum::GetValueAsString(EventType);
		FString TargetName = TargetActor ? TargetActor->GetName() : TEXT("None");
		UE_LOG(LogTemp, Log, TEXT("BTTask_SignalEventToLeader: Follower %s signaled %s (Target: %s)"),
			*ControlledPawn->GetName(), *EventName, *TargetName);
	}

	return EBTNodeResult::Succeeded;
}

FString UBTTask_SignalEventToLeader::GetStaticDescription() const
{
	FString EventName = UEnum::GetValueAsString(EventType);
	FString TargetKeyName = TargetActorKey.SelectedKeyName != NAME_None
		? TargetActorKey.SelectedKeyName.ToString()
		: TEXT("None");

	return FString::Printf(TEXT("Signal '%s' to leader (Target: %s)"),
		*EventName, *TargetKeyName);
}
