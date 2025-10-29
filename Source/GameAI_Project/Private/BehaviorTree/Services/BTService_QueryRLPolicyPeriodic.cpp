// Copyright Epic Games, Inc. All Rights Reserved.

#include "BehaviorTree/Services/BTService_QueryRLPolicyPeriodic.h"
#include "BehaviorTree/BlackboardComponent.h"
#include "AIController.h"
#include "GameFramework/Pawn.h"
#include "Team/FollowerAgentComponent.h"
#include "Observation/ObservationElement.h"
#include "RL/RLTypes.h"

UBTService_QueryRLPolicyPeriodic::UBTService_QueryRLPolicyPeriodic()
{
	NodeName = "Query RL Policy (Periodic)";
	Interval = 1.0f;  // Query every 1 second
	RandomDeviation = 0.2f;
	bNotifyBecomeRelevant = true;
}

void UBTService_QueryRLPolicyPeriodic::TickNode(UBehaviorTreeComponent& OwnerComp, uint8* NodeMemory, float DeltaSeconds)
{
	Super::TickNode(OwnerComp, NodeMemory, DeltaSeconds);

	// Get AI controller and pawn
	AAIController* AIController = OwnerComp.GetAIOwner();
	if (!AIController)
	{
		return;
	}

	APawn* ControlledPawn = AIController->GetPawn();
	if (!ControlledPawn)
	{
		return;
	}

	// Get blackboard component
	UBlackboardComponent* BlackboardComp = OwnerComp.GetBlackboardComponent();
	if (!BlackboardComp)
	{
		return;
	}

	// Get FollowerAgentComponent
	UFollowerAgentComponent* FollowerComp = ControlledPawn->FindComponentByClass<UFollowerAgentComponent>();
	if (!FollowerComp)
	{
		if (bLogQueries)
		{
			UE_LOG(LogTemp, Warning, TEXT("BTService_QueryRLPolicyPeriodic: No FollowerAgentComponent found on %s"),
				*ControlledPawn->GetName());
		}
		return;
	}

	// Check if policy is ready
	bool bPolicyReady = FollowerComp->IsTacticalPolicyReady();
	if (IsPolicyReadyKey.SelectedKeyName != NAME_None)
	{
		BlackboardComp->SetValueAsBool(IsPolicyReadyKey.SelectedKeyName, bPolicyReady);
	}

	if (!bPolicyReady)
	{
		if (bLogQueries)
		{
			UE_LOG(LogTemp, Warning, TEXT("BTService_QueryRLPolicyPeriodic: RL policy not ready"));
		}
		return;
	}

	// Check if active command required
	if (bRequireActiveCommand && !FollowerComp->HasActiveCommand())
	{
		if (bLogQueries)
		{
			UE_LOG(LogTemp, Verbose, TEXT("BTService_QueryRLPolicyPeriodic: No active command, skipping query"));
		}
		return;
	}

	// Get current observation
	FObservationElement CurrentObservation = FollowerComp->GetLocalObservation();

	// Check if observation changed significantly (if enabled)
	if (bQueryOnlyWhenObservationChanged)
	{
		if (LastObservations.Contains(FollowerComp))
		{
			FObservationElement LastObservation = LastObservations[FollowerComp];
			float Similarity = FObservationElement::CalculateSimilarity(CurrentObservation, LastObservation);

			if (Similarity >= ObservationSimilarityThreshold)
			{
				if (bLogQueries)
				{
					UE_LOG(LogTemp, Verbose, TEXT("BTService_QueryRLPolicyPeriodic: Observation unchanged (%.2f%%), skipping query"),
						Similarity * 100.0f);
				}
				return;
			}
		}

		// Store current observation for next comparison
		LastObservations.Add(FollowerComp, CurrentObservation);
	}

	// Query RL policy
	ETacticalAction SelectedAction = FollowerComp->QueryRLPolicy();

	// Update blackboard with selected action
	if (TacticalActionKey.SelectedKeyName != NAME_None)
	{
		uint8 ActionByte = static_cast<uint8>(SelectedAction);
		BlackboardComp->SetValueAsEnum(TacticalActionKey.SelectedKeyName, ActionByte);
	}

	// Get action probability (if policy supports it)
	if (ActionProbabilityKey.SelectedKeyName != NAME_None)
	{
		// Get policy from follower component
		if (FollowerComp->GetTacticalPolicy())
		{
			float ActionValue = FollowerComp->GetTacticalPolicy()->GetActionValue(CurrentObservation, SelectedAction);
			BlackboardComp->SetValueAsFloat(ActionProbabilityKey.SelectedKeyName, ActionValue);
		}
	}

	// Log query
	if (bLogQueries)
	{
		FString ActionName = UEnum::GetValueAsString(SelectedAction);
		UE_LOG(LogTemp, Log, TEXT("BTService_QueryRLPolicyPeriodic: %s selected action: %s"),
			*ControlledPawn->GetName(), *ActionName);
	}
}

FString UBTService_QueryRLPolicyPeriodic::GetStaticDescription() const
{
	FString ActionKeyName = TacticalActionKey.SelectedKeyName != NAME_None
		? TacticalActionKey.SelectedKeyName.ToString()
		: TEXT("None");

	FString ConfigStr;
	if (bQueryOnlyWhenObservationChanged)
	{
		ConfigStr += FString::Printf(TEXT(", OnChange(%.0f%%)"), ObservationSimilarityThreshold * 100.0f);
	}
	if (bRequireActiveCommand)
	{
		ConfigStr += TEXT(", RequireCmd");
	}
	if (!bEnableExploration)
	{
		ConfigStr += TEXT(", NoExplore");
	}

	return FString::Printf(TEXT("Query RL policy -> %s%s"),
		*ActionKeyName, *ConfigStr);
}
