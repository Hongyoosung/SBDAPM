// Copyright Epic Games, Inc. All Rights Reserved.

#include "AI/SBDAPMController.h"
#include "Core/StateMachine.h"
#include "BehaviorTree/BehaviorTree.h"
#include "BehaviorTree/BlackboardComponent.h"
#include "GameFramework/Pawn.h"

ASBDAPMController::ASBDAPMController()
{
	// Enable ticking for this controller
	PrimaryActorTick.bCanEverTick = true;

	// Create and initialize the Blackboard component
	// Note: AAIController already has a BlackboardComponent, we just need to ensure it's properly set up
}

void ASBDAPMController::BeginPlay()
{
	Super::BeginPlay();

	// Additional initialization can go here if needed
}

void ASBDAPMController::OnPossess(APawn* InPawn)
{
	Super::OnPossess(InPawn);

	if (!InPawn)
	{
		UE_LOG(LogTemp, Warning, TEXT("ASBDAPMController::OnPossess - InPawn is nullptr"));
		return;
	}

	// Cache the StateMachine component
	CachedStateMachine = InPawn->FindComponentByClass<UStateMachine>();

	if (!CachedStateMachine)
	{
		UE_LOG(LogTemp, Warning, TEXT("ASBDAPMController::OnPossess - No StateMachine component found on pawn %s"), *InPawn->GetName());
	}
	else
	{
		UE_LOG(LogTemp, Log, TEXT("ASBDAPMController::OnPossess - Successfully found StateMachine on pawn %s"), *InPawn->GetName());
	}

	// Start the Behavior Tree if configured to do so
	if (bAutoStartBehaviorTree)
	{
		if (!StartBehaviorTree())
		{
			UE_LOG(LogTemp, Warning, TEXT("ASBDAPMController::OnPossess - Failed to start Behavior Tree on pawn %s"), *InPawn->GetName());
		}
	}
}

void ASBDAPMController::OnUnPossess()
{
	// Stop the Behavior Tree before unpossessing
	StopBehaviorTree();

	// Clear cached references
	CachedStateMachine = nullptr;

	Super::OnUnPossess();
}

UStateMachine* ASBDAPMController::GetStateMachine() const
{
	return CachedStateMachine;
}

bool ASBDAPMController::StartBehaviorTree()
{
	// Check if we have a Behavior Tree asset assigned
	if (!BehaviorTreeAsset)
	{
		UE_LOG(LogTemp, Error, TEXT("ASBDAPMController::StartBehaviorTree - No BehaviorTreeAsset assigned to controller"));
		return false;
	}

	// Run the Behavior Tree
	bool bSuccess = RunBehaviorTree(BehaviorTreeAsset);

	if (bSuccess)
	{
		UE_LOG(LogTemp, Log, TEXT("ASBDAPMController::StartBehaviorTree - Successfully started Behavior Tree"));

		// Initialize default Blackboard values
		if (Blackboard)
		{
			// Set default strategy to "MoveTo"
			SetBlackboardValueAsString(FName("CurrentStrategy"), TEXT("MoveTo"));

			// Initialize other default values
			SetBlackboardValueAsFloat(FName("ThreatLevel"), 0.0f);
			SetBlackboardValueAsBool(FName("bCanSeeEnemy"), false);
			SetBlackboardValueAsFloat(FName("LastObservationUpdate"), 0.0f);
		}
	}
	else
	{
		UE_LOG(LogTemp, Error, TEXT("ASBDAPMController::StartBehaviorTree - Failed to run Behavior Tree"));
	}

	return bSuccess;
}

void ASBDAPMController::StopBehaviorTree()
{
	if (BrainComponent)
	{
		BrainComponent->StopLogic(TEXT("Manually stopped"));
		UE_LOG(LogTemp, Log, TEXT("ASBDAPMController::StopBehaviorTree - Behavior Tree stopped"));
	}
}

bool ASBDAPMController::IsBehaviorTreeRunning() const
{
	return BrainComponent && BrainComponent->IsRunning();
}

// Blackboard convenience methods

void ASBDAPMController::SetBlackboardValueAsString(FName KeyName, const FString& Value)
{
	if (Blackboard)
	{
		Blackboard->SetValueAsString(KeyName, Value);
	}
	else
	{
		UE_LOG(LogTemp, Warning, TEXT("ASBDAPMController::SetBlackboardValueAsString - Blackboard is nullptr"));
	}
}

void ASBDAPMController::SetBlackboardValueAsVector(FName KeyName, FVector Value)
{
	if (Blackboard)
	{
		Blackboard->SetValueAsVector(KeyName, Value);
	}
	else
	{
		UE_LOG(LogTemp, Warning, TEXT("ASBDAPMController::SetBlackboardValueAsVector - Blackboard is nullptr"));
	}
}

void ASBDAPMController::SetBlackboardValueAsObject(FName KeyName, UObject* Value)
{
	if (Blackboard)
	{
		Blackboard->SetValueAsObject(KeyName, Value);
	}
	else
	{
		UE_LOG(LogTemp, Warning, TEXT("ASBDAPMController::SetBlackboardValueAsObject - Blackboard is nullptr"));
	}
}

void ASBDAPMController::SetBlackboardValueAsFloat(FName KeyName, float Value)
{
	if (Blackboard)
	{
		Blackboard->SetValueAsFloat(KeyName, Value);
	}
	else
	{
		UE_LOG(LogTemp, Warning, TEXT("ASBDAPMController::SetBlackboardValueAsFloat - Blackboard is nullptr"));
	}
}

void ASBDAPMController::SetBlackboardValueAsBool(FName KeyName, bool Value)
{
	if (Blackboard)
	{
		Blackboard->SetValueAsBool(KeyName, Value);
	}
	else
	{
		UE_LOG(LogTemp, Warning, TEXT("ASBDAPMController::SetBlackboardValueAsBool - Blackboard is nullptr"));
	}
}

FString ASBDAPMController::GetBlackboardValueAsString(FName KeyName) const
{
	if (Blackboard)
	{
		return Blackboard->GetValueAsString(KeyName);
	}

	UE_LOG(LogTemp, Warning, TEXT("ASBDAPMController::GetBlackboardValueAsString - Blackboard is nullptr"));
	return FString();
}

FVector ASBDAPMController::GetBlackboardValueAsVector(FName KeyName) const
{
	if (Blackboard)
	{
		return Blackboard->GetValueAsVector(KeyName);
	}

	UE_LOG(LogTemp, Warning, TEXT("ASBDAPMController::GetBlackboardValueAsVector - Blackboard is nullptr"));
	return FVector::ZeroVector;
}

UObject* ASBDAPMController::GetBlackboardValueAsObject(FName KeyName) const
{
	if (Blackboard)
	{
		return Blackboard->GetValueAsObject(KeyName);
	}

	UE_LOG(LogTemp, Warning, TEXT("ASBDAPMController::GetBlackboardValueAsObject - Blackboard is nullptr"));
	return nullptr;
}

float ASBDAPMController::GetBlackboardValueAsFloat(FName KeyName) const
{
	if (Blackboard)
	{
		return Blackboard->GetValueAsFloat(KeyName);
	}

	UE_LOG(LogTemp, Warning, TEXT("ASBDAPMController::GetBlackboardValueAsFloat - Blackboard is nullptr"));
	return 0.0f;
}

bool ASBDAPMController::GetBlackboardValueAsBool(FName KeyName) const
{
	if (Blackboard)
	{
		return Blackboard->GetValueAsBool(KeyName);
	}

	UE_LOG(LogTemp, Warning, TEXT("ASBDAPMController::GetBlackboardValueAsBool - Blackboard is nullptr"));
	return false;
}
