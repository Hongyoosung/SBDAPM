// Copyright Epic Games, Inc. All Rights Reserved.

#include "StateTree/Tasks/STTask_Dead.h"
#include "StateTree/FollowerStateTreeContext.h"
#include "Team/FollowerAgentComponent.h"
#include "AIController.h"
#include "GameFramework/Character.h"
#include "GameFramework/CharacterMovementComponent.h"
#include "Components/CapsuleComponent.h"
#include "Animation/AnimInstance.h"

EStateTreeRunStatus FSTTask_Dead::EnterState(FStateTreeExecutionContext& Context, const FStateTreeTransitionResult& Transition) const
{
	FInstanceDataType& InstanceData = Context.GetInstanceData(*this);

	// Validate inputs
	if (!InstanceData.Context.AIController)
	{
		UE_LOG(LogTemp, Error, TEXT("STTask_Dead: Invalid AIController"));
		return EStateTreeRunStatus::Failed;
	}

	APawn* Pawn = InstanceData.Context.AIController->GetPawn();
	if (!Pawn)
	{
		UE_LOG(LogTemp, Error, TEXT("STTask_Dead: No pawn found"));
		return EStateTreeRunStatus::Failed;
	}

	// Reset state
	InstanceData.TimeSinceDeath = 0.0f;
	InstanceData.bAnimationStarted = false;
	InstanceData.bMarkedForDestruction = false;

	// Stop all movement
	InstanceData.Context.AIController->StopMovement();

	// Clear focus
	InstanceData.Context.AIController->ClearFocus(EAIFocusPriority::Gameplay);

	// Disable collision if configured
	if (InstanceData.bDisableCollision)
	{
		if (ACharacter* Character = Cast<ACharacter>(Pawn))
		{
			if (UCapsuleComponent* Capsule = Character->GetCapsuleComponent())
			{
				Capsule->SetCollisionEnabled(ECollisionEnabled::NoCollision);
			}
		}
		else
		{
			Pawn->SetActorEnableCollision(false);
		}
	}

	// Play death animation
	PlayDeathAnimation(Context);

	// Enable ragdoll if configured
	if (InstanceData.bEnableRagdoll)
	{
		EnableRagdoll(Context);
	}

	UE_LOG(LogTemp, Log, TEXT("STTask_Dead: Agent %s entered death state (destroy in %.1fs)"),
		*Pawn->GetName(), InstanceData.DestroyDelay);

	return EStateTreeRunStatus::Running;
}

EStateTreeRunStatus FSTTask_Dead::Tick(FStateTreeExecutionContext& Context, float DeltaTime) const
{
	FInstanceDataType& InstanceData = Context.GetInstanceData(*this);

	// Update timer
	InstanceData.TimeSinceDeath += DeltaTime;

	// Check if it's time to destroy
	if (InstanceData.TimeSinceDeath >= InstanceData.DestroyDelay && !InstanceData.bMarkedForDestruction)
	{
		DestroyActor(Context);
		InstanceData.bMarkedForDestruction = true;

		// Return succeeded to signal completion (though actor will be destroyed)
		return EStateTreeRunStatus::Succeeded;
	}

	return EStateTreeRunStatus::Running;
}

void FSTTask_Dead::ExitState(FStateTreeExecutionContext& Context, const FStateTreeTransitionResult& Transition) const
{
	FInstanceDataType& InstanceData = Context.GetInstanceData(*this);

	UE_LOG(LogTemp, Log, TEXT("STTask_Dead: Exiting death state (time: %.1fs)"),
		InstanceData.TimeSinceDeath);
}

void FSTTask_Dead::PlayDeathAnimation(FStateTreeExecutionContext& Context) const
{
	FInstanceDataType& InstanceData = Context.GetInstanceData(*this);

	if (!InstanceData.DeathMontage)
	{
		UE_LOG(LogTemp, Warning, TEXT("STTask_Dead: No death montage assigned"));
		return;
	}

	APawn* Pawn = InstanceData.Context.AIController ? InstanceData.Context.AIController->GetPawn() : nullptr;
	if (!Pawn)
	{
		return;
	}

	ACharacter* Character = Cast<ACharacter>(Pawn);
	if (!Character)
	{
		UE_LOG(LogTemp, Warning, TEXT("STTask_Dead: Pawn is not a Character, cannot play montage"));
		return;
	}

	UAnimInstance* AnimInstance = Character->GetMesh()->GetAnimInstance();
	if (!AnimInstance)
	{
		UE_LOG(LogTemp, Warning, TEXT("STTask_Dead: No AnimInstance found"));
		return;
	}

	// Play the death montage
	float MontageLength = AnimInstance->Montage_Play(InstanceData.DeathMontage);
	if (MontageLength > 0.0f)
	{
		InstanceData.bAnimationStarted = true;
		UE_LOG(LogTemp, Log, TEXT("STTask_Dead: Playing death montage (length: %.1fs)"), MontageLength);
	}
	else
	{
		UE_LOG(LogTemp, Warning, TEXT("STTask_Dead: Failed to play death montage"));
	}
}

void FSTTask_Dead::EnableRagdoll(FStateTreeExecutionContext& Context) const
{
	FInstanceDataType& InstanceData = Context.GetInstanceData(*this);

	APawn* Pawn = InstanceData.Context.AIController ? InstanceData.Context.AIController->GetPawn() : nullptr;
	if (!Pawn)
	{
		return;
	}

	ACharacter* Character = Cast<ACharacter>(Pawn);
	if (!Character)
	{
		return;
	}

	// Get skeletal mesh
	USkeletalMeshComponent* MeshComp = Character->GetMesh();
	if (!MeshComp)
	{
		return;
	}

	// Disable character movement
	if (UCharacterMovementComponent* MovementComp = Character->GetCharacterMovement())
	{
		MovementComp->DisableMovement();
		MovementComp->StopMovementImmediately();
	}

	// Enable physics simulation on mesh
	MeshComp->SetSimulatePhysics(true);
	MeshComp->SetCollisionEnabled(ECollisionEnabled::QueryAndPhysics);
	MeshComp->SetCollisionResponseToAllChannels(ECR_Block);

	UE_LOG(LogTemp, Log, TEXT("STTask_Dead: Ragdoll enabled"));
}

void FSTTask_Dead::DestroyActor(FStateTreeExecutionContext& Context) const
{
	FInstanceDataType& InstanceData = Context.GetInstanceData(*this);

	APawn* Pawn = InstanceData.Context.AIController ? InstanceData.Context.AIController->GetPawn() : nullptr;
	if (!Pawn)
	{
		return;
	}

	UE_LOG(LogTemp, Log, TEXT("STTask_Dead: Destroying actor %s"), *Pawn->GetName());

	// Unpossess first
	if (InstanceData.Context.AIController)
	{
		InstanceData.Context.AIController->UnPossess();
	}

	// Destroy the pawn
	Pawn->Destroy();
}
