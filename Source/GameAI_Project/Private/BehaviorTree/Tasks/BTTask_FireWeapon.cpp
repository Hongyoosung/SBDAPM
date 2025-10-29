// Copyright Epic Games, Inc. All Rights Reserved.

#include "BehaviorTree/Tasks/BTTask_FireWeapon.h"
#include "BehaviorTree/BlackboardComponent.h"
#include "AIController.h"
#include "GameFramework/Pawn.h"
#include "GameFramework/Actor.h"
#include "Engine/World.h"
#include "DrawDebugHelpers.h"
#include "Interfaces/CombatStatsInterface.h"
#include "Team/FollowerAgentComponent.h"

UBTTask_FireWeapon::UBTTask_FireWeapon()
{
	NodeName = "Fire Weapon";
	bNotifyTick = false;
}

EBTNodeResult::Type UBTTask_FireWeapon::ExecuteTask(UBehaviorTreeComponent& OwnerComp, uint8* NodeMemory)
{
	// Get AI controller and pawn
	AAIController* AIController = OwnerComp.GetAIOwner();
	if (!AIController)
	{
		UE_LOG(LogTemp, Error, TEXT("BTTask_FireWeapon: No AIController found"));
		return EBTNodeResult::Failed;
	}

	APawn* ControlledPawn = AIController->GetPawn();
	if (!ControlledPawn)
	{
		UE_LOG(LogTemp, Error, TEXT("BTTask_FireWeapon: No controlled pawn"));
		return EBTNodeResult::Failed;
	}

	// Get target from blackboard
	UBlackboardComponent* BlackboardComp = OwnerComp.GetBlackboardComponent();
	if (!BlackboardComp)
	{
		UE_LOG(LogTemp, Error, TEXT("BTTask_FireWeapon: No blackboard component"));
		return EBTNodeResult::Failed;
	}

	AActor* TargetActor = Cast<AActor>(BlackboardComp->GetValueAsObject(TargetActorKey.SelectedKeyName));
	if (!TargetActor)
	{
		if (bLogFiring)
		{
			UE_LOG(LogTemp, Warning, TEXT("BTTask_FireWeapon: No target actor in blackboard key '%s'"),
				*TargetActorKey.SelectedKeyName.ToString());
		}
		return EBTNodeResult::Failed;
	}

	// Check if target is valid and alive
	if (!IsValid(TargetActor) || TargetActor->IsPendingKill())
	{
		if (bLogFiring)
		{
			UE_LOG(LogTemp, Warning, TEXT("BTTask_FireWeapon: Target actor is invalid or destroyed"));
		}
		return EBTNodeResult::Failed;
	}

	// Check range
	float DistanceToTarget = FVector::Dist(ControlledPawn->GetActorLocation(), TargetActor->GetActorLocation());
	if (DistanceToTarget > MaxRange)
	{
		if (bLogFiring)
		{
			UE_LOG(LogTemp, Warning, TEXT("BTTask_FireWeapon: Target out of range (%.1f > %.1f)"),
				DistanceToTarget, MaxRange);
		}
		return EBTNodeResult::Failed;
	}

	// Check if weapon can fire (using ICombatStatsInterface)
	ICombatStatsInterface* CombatStats = Cast<ICombatStatsInterface>(ControlledPawn);
	if (CombatStats)
	{
		bool bCanFire = CombatStats->Execute_CanFireWeapon(ControlledPawn);
		if (!bCanFire)
		{
			if (bLogFiring)
			{
				UE_LOG(LogTemp, Verbose, TEXT("BTTask_FireWeapon: Weapon cannot fire (cooldown or no ammo)"));
			}
			return EBTNodeResult::Failed;
		}
	}

	// Check line of sight
	if (bRequireLineOfSight && !HasLineOfSight(ControlledPawn, TargetActor))
	{
		if (bLogFiring)
		{
			UE_LOG(LogTemp, Verbose, TEXT("BTTask_FireWeapon: No line of sight to target"));
		}
		return EBTNodeResult::Failed;
	}

	// Fire weapon
	float DamageDealt = 0.0f;
	bool bKilled = false;
	bool bFired = FireWeapon(ControlledPawn, TargetActor, DamageDealt, bKilled);

	if (!bFired)
	{
		return EBTNodeResult::Failed;
	}

	// Provide reward to RL policy
	if (bProvideReward)
	{
		float Reward = 0.0f;
		if (bKilled)
		{
			Reward = RewardForKill;
		}
		else if (DamageDealt > 0.0f)
		{
			Reward = RewardForHit;
		}
		else
		{
			Reward = RewardForMiss;
		}

		ProvideRewardToRLPolicy(OwnerComp, Reward);
	}

	// Log result
	if (bLogFiring)
	{
		if (bKilled)
		{
			UE_LOG(LogTemp, Log, TEXT("BTTask_FireWeapon: KILLED target %s (damage: %.1f, reward: %.1f)"),
				*TargetActor->GetName(), DamageDealt, RewardForKill);
		}
		else if (DamageDealt > 0.0f)
		{
			UE_LOG(LogTemp, Log, TEXT("BTTask_FireWeapon: HIT target %s (damage: %.1f, reward: %.1f)"),
				*TargetActor->GetName(), DamageDealt, RewardForHit);
		}
		else
		{
			UE_LOG(LogTemp, Log, TEXT("BTTask_FireWeapon: MISSED target %s (reward: %.1f)"),
				*TargetActor->GetName(), RewardForMiss);
		}
	}

	return EBTNodeResult::Succeeded;
}

FString UBTTask_FireWeapon::GetStaticDescription() const
{
	FString TargetKeyName = TargetActorKey.SelectedKeyName != NAME_None
		? TargetActorKey.SelectedKeyName.ToString()
		: TEXT("None");

	return FString::Printf(TEXT("Fire at '%s' (Range: %.0f, Damage: %.1f, Accuracy: %.1f%%)"),
		*TargetKeyName, MaxRange, BaseDamage, BaseAccuracy * 100.0f);
}

float UBTTask_FireWeapon::CalculateAccuracy(float Distance, const FVector& TargetVelocity) const
{
	// Start with base accuracy
	float FinalAccuracy = BaseAccuracy;

	// Apply distance penalty
	float DistanceInKilounits = Distance / 1000.0f;
	FinalAccuracy -= (DistanceInKilounits * AccuracyDistancePenalty);

	// Apply movement penalty (faster targets harder to hit)
	float TargetSpeed = TargetVelocity.Size();
	float MovementPenalty = FMath::Clamp(TargetSpeed / 600.0f, 0.0f, 0.3f); // Max 30% penalty
	FinalAccuracy -= MovementPenalty;

	// Clamp to valid range
	return FMath::Clamp(FinalAccuracy, 0.0f, 1.0f);
}

bool UBTTask_FireWeapon::HasLineOfSight(APawn* Shooter, AActor* Target) const
{
	if (!Shooter || !Target)
	{
		return false;
	}

	UWorld* World = Shooter->GetWorld();
	if (!World)
	{
		return false;
	}

	// Get fire start location (from pawn's eye level or weapon socket)
	FVector FireStart = Shooter->GetActorLocation() + FVector(0, 0, 80.0f); // Approximate eye level

	// Get target location
	FVector TargetLocation = Target->GetActorLocation();

	// Perform line trace
	FHitResult HitResult;
	FCollisionQueryParams QueryParams;
	QueryParams.AddIgnoredActor(Shooter);
	QueryParams.bTraceComplex = false;

	bool bHit = World->LineTraceSingleByChannel(
		HitResult,
		FireStart,
		TargetLocation,
		ECC_Visibility,
		QueryParams
	);

	// If we hit the target or nothing, we have line of sight
	return !bHit || HitResult.GetActor() == Target;
}

bool UBTTask_FireWeapon::FireWeapon(APawn* Shooter, AActor* Target, float& OutDamageDealt, bool& OutKilled)
{
	OutDamageDealt = 0.0f;
	OutKilled = false;

	if (!Shooter || !Target)
	{
		return false;
	}

	UWorld* World = Shooter->GetWorld();
	if (!World)
	{
		return false;
	}

	// Get fire start and target locations
	FVector FireStart = Shooter->GetActorLocation() + FVector(0, 0, 80.0f);
	FVector TargetLocation = Target->GetActorLocation();
	float Distance = FVector::Dist(FireStart, TargetLocation);

	// Get target velocity
	FVector TargetVelocity = FVector::ZeroVector;
	if (APawn* TargetPawn = Cast<APawn>(Target))
	{
		TargetVelocity = TargetPawn->GetVelocity();
	}

	// Calculate accuracy
	float Accuracy = CalculateAccuracy(Distance, TargetVelocity);

	// Determine if shot hits
	bool bHit = FMath::FRand() <= Accuracy;

	// Calculate final fire end point (with spread if miss)
	FVector FireEnd = TargetLocation;
	if (!bHit)
	{
		// Add random spread for miss
		float SpreadRadius = Distance * 0.2f; // 20% of distance
		FVector RandomOffset = FVector(
			FMath::RandRange(-SpreadRadius, SpreadRadius),
			FMath::RandRange(-SpreadRadius, SpreadRadius),
			FMath::RandRange(-SpreadRadius * 0.5f, SpreadRadius * 0.5f)
		);
		FireEnd += RandomOffset;
	}

	// Spawn visual effects
	if (bSpawnMuzzleFlash || bSpawnTracer)
	{
		SpawnFireEffects(Shooter, FireStart, FireEnd, bHit);
	}

	// Draw debug line
	if (bDrawDebugLines)
	{
		FColor LineColor = bHit ? FColor::Red : FColor::Yellow;
		DrawDebugLine(World, FireStart, FireEnd, LineColor, false, 1.0f, 0, 2.0f);
	}

	// Apply damage if hit
	if (bHit)
	{
		OutDamageDealt = BaseDamage;
		OutKilled = ApplyDamageToTarget(Target, BaseDamage, Shooter, OutKilled);
	}

	return true;
}

bool UBTTask_FireWeapon::ApplyDamageToTarget(AActor* Target, float Damage, APawn* Instigator, bool& OutKilled)
{
	OutKilled = false;

	if (!Target || !Instigator)
	{
		return false;
	}

	// Try to apply damage via ICombatStatsInterface if target implements it
	ICombatStatsInterface* TargetCombatStats = Cast<ICombatStatsInterface>(Target);
	if (TargetCombatStats)
	{
		// Check if target is alive before damage
		bool bWasAlive = TargetCombatStats->Execute_IsAlive(Target);

		// Apply damage (would need a TakeDamage function in the interface or use Unreal's damage system)
		// For now, we'll use Unreal's built-in damage system
		Target->TakeDamage(Damage, FDamageEvent(), Instigator->GetController(), Instigator);

		// Check if target died
		bool bIsAliveNow = TargetCombatStats->Execute_IsAlive(Target);
		OutKilled = bWasAlive && !bIsAliveNow;

		return true;
	}
	else
	{
		// Fallback to Unreal's damage system
		Target->TakeDamage(Damage, FDamageEvent(), Instigator->GetController(), Instigator);
		return true;
	}
}

void UBTTask_FireWeapon::ProvideRewardToRLPolicy(UBehaviorTreeComponent& OwnerComp, float Reward)
{
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

	// Get FollowerAgentComponent
	UFollowerAgentComponent* FollowerComp = ControlledPawn->FindComponentByClass<UFollowerAgentComponent>();
	if (!FollowerComp)
	{
		return;
	}

	// Accumulate reward
	FollowerComp->AccumulateReward(Reward);

	if (bLogFiring)
	{
		UE_LOG(LogTemp, Verbose, TEXT("BTTask_FireWeapon: Provided reward %.2f to RL policy (Total: %.2f)"),
			Reward, FollowerComp->GetAccumulatedReward());
	}
}

void UBTTask_FireWeapon::SpawnFireEffects(APawn* Shooter, const FVector& FireStart, const FVector& FireEnd, bool bHit)
{
	if (!Shooter)
	{
		return;
	}

	UWorld* World = Shooter->GetWorld();
	if (!World)
	{
		return;
	}

	// TODO: Spawn muzzle flash particle effect at FireStart
	// TODO: Spawn tracer effect from FireStart to FireEnd
	// TODO: Spawn impact effect at FireEnd if bHit
	// TODO: Apply camera shake to player if nearby

	// For now, just log
	if (bLogFiring)
	{
		UE_LOG(LogTemp, Verbose, TEXT("BTTask_FireWeapon: Spawned fire effects (muzzle flash: %s, tracer: %s)"),
			bSpawnMuzzleFlash ? TEXT("yes") : TEXT("no"),
			bSpawnTracer ? TEXT("yes") : TEXT("no"));
	}
}
