// Copyright Epic Games, Inc. All Rights Reserved.

#include "BehaviorTree/BTService_UpdateObservation.h"
#include "Core/StateMachine.h"
#include "Core/ObservationElement.h"
#include "AIController.h"
#include "BehaviorTree/BlackboardComponent.h"
#include "GameFramework/Character.h"
#include "GameFramework/CharacterMovementComponent.h"
#include "Kismet/GameplayStatics.h"
#include "DrawDebugHelpers.h"

UBTService_UpdateObservation::UBTService_UpdateObservation()
{
	NodeName = "Update Observation";

	// Set the interval - this controls how often TickNode is called
	// We'll use our own time accumulator for more precise control
	Interval = UpdateInterval;
	RandomDeviation = 0.0f; // No random deviation for consistent updates
}

void UBTService_UpdateObservation::OnBecomeRelevant(UBehaviorTreeComponent& OwnerComp, uint8* NodeMemory)
{
	Super::OnBecomeRelevant(OwnerComp, NodeMemory);

	// Reset time accumulator
	TimeAccumulator = 0.0f;

	if (bEnableDebugLog)
	{
		UE_LOG(LogTemp, Log, TEXT("BTService_UpdateObservation: Service became active"));
	}
}

void UBTService_UpdateObservation::TickNode(UBehaviorTreeComponent& OwnerComp, uint8* NodeMemory, float DeltaSeconds)
{
	Super::TickNode(OwnerComp, NodeMemory, DeltaSeconds);

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

	// Update time accumulator
	TimeAccumulator += DeltaSeconds;

	// Only update at the specified interval
	if (TimeAccumulator >= UpdateInterval)
	{
		TimeAccumulator = 0.0f;

		// Gather observation data
		FObservationElement NewObservation = GatherObservationData(AIController, ControlledPawn);

		// Get the StateMachine component
		UStateMachine* StateMachine = ControlledPawn->FindComponentByClass<UStateMachine>();
		if (StateMachine)
		{
			// Update the StateMachine's observation
			StateMachine->UpdateObservation(NewObservation);

			if (bEnableDebugLog)
			{
				UE_LOG(LogTemp, Log, TEXT("BTService_UpdateObservation: Updated StateMachine observation - Health: %.1f, Enemies: %d"),
					NewObservation.Health, NewObservation.VisibleEnemyCount);
			}
		}
		else if (bEnableDebugLog)
		{
			UE_LOG(LogTemp, Warning, TEXT("BTService_UpdateObservation: No StateMachine component found on pawn"));
		}

		// Sync to Blackboard
		UBlackboardComponent* BlackboardComp = OwnerComp.GetBlackboardComponent();
		if (BlackboardComp)
		{
			SyncToBlackboard(BlackboardComp, NewObservation);
		}
	}
}

void UBTService_UpdateObservation::OnCeaseRelevant(UBehaviorTreeComponent& OwnerComp, uint8* NodeMemory)
{
	Super::OnCeaseRelevant(OwnerComp, NodeMemory);

	if (bEnableDebugLog)
	{
		UE_LOG(LogTemp, Log, TEXT("BTService_UpdateObservation: Service became inactive"));
	}
}

FObservationElement UBTService_UpdateObservation::GatherObservationData(AAIController* OwnerController, APawn* ControlledPawn)
{
	FObservationElement Observation;

	// Update all observation components
	UpdateAgentState(Observation, ControlledPawn);
	PerformRaycastPerception(Observation, ControlledPawn);
	ScanForEnemies(Observation, ControlledPawn);
	DetectCover(Observation, ControlledPawn);
	UpdateCombatState(Observation, ControlledPawn);

	return Observation;
}

void UBTService_UpdateObservation::UpdateAgentState(FObservationElement& Observation, APawn* ControlledPawn)
{
	// Position
	Observation.Position = ControlledPawn->GetActorLocation();

	// Velocity
	Observation.Velocity = ControlledPawn->GetVelocity();

	// Rotation
	Observation.Rotation = ControlledPawn->GetActorRotation();

	// Health (try to get from Character, otherwise use default)
	ACharacter* Character = Cast<ACharacter>(ControlledPawn);
	if (Character)
	{
		// Note: This assumes your character has a Health property accessible via Blueprint
		// You may need to adjust this based on your actual health system
		// For now, we'll use a placeholder that can be overridden in Blueprint
		Observation.Health = 100.0f; // TODO: Get actual health from character
	}
	else
	{
		Observation.Health = 100.0f;
	}

	// Stamina (placeholder)
	Observation.Stamina = 100.0f; // TODO: Get actual stamina

	// Shield (placeholder)
	Observation.Shield = 0.0f; // TODO: Get actual shield value
}

void UBTService_UpdateObservation::PerformRaycastPerception(FObservationElement& Observation, APawn* ControlledPawn)
{
	UWorld* World = ControlledPawn->GetWorld();
	if (!World)
	{
		return;
	}

	// Clear existing raycast data
	Observation.RaycastDistances.Empty();
	Observation.RaycastHitTypes.Empty();

	FVector StartLocation = ControlledPawn->GetActorLocation();
	FRotator BaseRotation = ControlledPawn->GetActorRotation();

	// Perform raycasts in a circle around the agent
	float AngleStep = 360.0f / RaycastCount;

	for (int32 i = 0; i < RaycastCount; i++)
	{
		float Angle = i * AngleStep;
		FRotator RayRotation = BaseRotation + FRotator(0.0f, Angle, 0.0f);
		FVector RayDirection = RayRotation.Vector();
		FVector EndLocation = StartLocation + (RayDirection * RaycastMaxDistance);

		FHitResult HitResult;
		FCollisionQueryParams QueryParams;
		QueryParams.AddIgnoredActor(ControlledPawn);

		bool bHit = World->LineTraceSingleByChannel(
			HitResult,
			StartLocation,
			EndLocation,
			ECC_Visibility,
			QueryParams
		);

		if (bHit)
		{
			// Normalize distance to [0, 1] range
			float NormalizedDistance = HitResult.Distance / RaycastMaxDistance;
			Observation.RaycastDistances.Add(NormalizedDistance);
			Observation.RaycastHitTypes.Add(ClassifyHitType(HitResult));

			if (bDrawDebugInfo)
			{
				DrawDebugLine(World, StartLocation, HitResult.Location, FColor::Green, false, UpdateInterval, 0, 1.0f);
				DrawDebugPoint(World, HitResult.Location, 5.0f, FColor::Red, false, UpdateInterval);
			}
		}
		else
		{
			// No hit - max distance
			Observation.RaycastDistances.Add(1.0f);
			Observation.RaycastHitTypes.Add(ERaycastHitType::None);

			if (bDrawDebugInfo)
			{
				DrawDebugLine(World, StartLocation, EndLocation, FColor::Blue, false, UpdateInterval, 0, 0.5f);
			}
		}
	}
}

void UBTService_UpdateObservation::ScanForEnemies(FObservationElement& Observation, APawn* ControlledPawn)
{
	UWorld* World = ControlledPawn->GetWorld();
	if (!World)
	{
		return;
	}

	// Find all actors with the Enemy tag
	TArray<AActor*> FoundEnemies;
	UGameplayStatics::GetAllActorsWithTag(World, EnemyTag, FoundEnemies);

	// Filter enemies by distance and visibility
	TArray<FEnemyObservation> EnemyObservations;
	FVector AgentLocation = ControlledPawn->GetActorLocation();
	FVector AgentForward = ControlledPawn->GetActorForwardVector();

	for (AActor* EnemyActor : FoundEnemies)
	{
		if (!EnemyActor || EnemyActor == ControlledPawn)
		{
			continue;
		}

		float Distance = FVector::Dist(AgentLocation, EnemyActor->GetActorLocation());

		// Only consider enemies within detection range
		if (Distance <= MaxEnemyDetectionDistance)
		{
			FEnemyObservation EnemyObs;
			EnemyObs.Distance = Distance;

			// Calculate relative angle
			FVector ToEnemy = (EnemyActor->GetActorLocation() - AgentLocation).GetSafeNormal();
			float DotProduct = FVector::DotProduct(AgentForward, ToEnemy);
			float AngleRadians = FMath::Acos(DotProduct);
			EnemyObs.RelativeAngle = FMath::RadiansToDegrees(AngleRadians);

			// Determine if enemy is to the left or right
			FVector CrossProduct = FVector::CrossProduct(AgentForward, ToEnemy);
			if (CrossProduct.Z < 0)
			{
				EnemyObs.RelativeAngle = -EnemyObs.RelativeAngle;
			}

			// Get enemy health (placeholder - adjust based on your health system)
			EnemyObs.Health = 100.0f; // TODO: Get actual enemy health

			EnemyObservations.Add(EnemyObs);

			if (bDrawDebugInfo)
			{
				DrawDebugLine(World, AgentLocation, EnemyActor->GetActorLocation(), FColor::Red, false, UpdateInterval, 0, 2.0f);
				DrawDebugSphere(World, EnemyActor->GetActorLocation(), 50.0f, 8, FColor::Orange, false, UpdateInterval);
			}
		}
	}

	// Sort by distance (closest first)
	EnemyObservations.Sort([](const FEnemyObservation& A, const FEnemyObservation& B) {
		return A.Distance < B.Distance;
	});

	// Take top 5 closest enemies
	Observation.NearbyEnemies.Empty();
	int32 Count = FMath::Min(5, EnemyObservations.Num());
	for (int32 i = 0; i < Count; i++)
	{
		Observation.NearbyEnemies.Add(EnemyObservations[i]);
	}

	// Fill remaining slots with default values
	while (Observation.NearbyEnemies.Num() < 5)
	{
		FEnemyObservation EmptyObs;
		EmptyObs.Distance = MaxEnemyDetectionDistance;
		EmptyObs.Health = 0.0f;
		EmptyObs.RelativeAngle = 0.0f;
		Observation.NearbyEnemies.Add(EmptyObs);
	}

	// Set visible enemy count
	Observation.VisibleEnemyCount = EnemyObservations.Num();
}

void UBTService_UpdateObservation::DetectCover(FObservationElement& Observation, APawn* ControlledPawn)
{
	UWorld* World = ControlledPawn->GetWorld();
	if (!World)
	{
		return;
	}

	// Find all actors with the Cover tag
	TArray<AActor*> CoverActors;
	UGameplayStatics::GetAllActorsWithTag(World, CoverTag, CoverActors);

	FVector AgentLocation = ControlledPawn->GetActorLocation();
	float NearestCoverDistance = CoverDetectionDistance;
	FVector NearestCoverLocation = FVector::ZeroVector;
	bool bFoundCover = false;

	for (AActor* CoverActor : CoverActors)
	{
		if (!CoverActor)
		{
			continue;
		}

		float Distance = FVector::Dist(AgentLocation, CoverActor->GetActorLocation());

		if (Distance < NearestCoverDistance)
		{
			NearestCoverDistance = Distance;
			NearestCoverLocation = CoverActor->GetActorLocation();
			bFoundCover = true;

			if (bDrawDebugInfo)
			{
				DrawDebugSphere(World, CoverActor->GetActorLocation(), 100.0f, 12, FColor::Cyan, false, UpdateInterval);
			}
		}
	}

	Observation.bHasCover = bFoundCover;
	Observation.NearestCoverDistance = NearestCoverDistance;

	if (bFoundCover)
	{
		FVector ToCover = (NearestCoverLocation - AgentLocation).GetSafeNormal2D();
		Observation.CoverDirection = FVector2D(ToCover.X, ToCover.Y);
	}
	else
	{
		Observation.CoverDirection = FVector2D::ZeroVector;
	}
}

void UBTService_UpdateObservation::UpdateCombatState(FObservationElement& Observation, APawn* ControlledPawn)
{
	// Placeholder combat state
	// TODO: Integrate with your weapon/combat system
	Observation.WeaponCooldown = 0.0f;
	Observation.Ammunition = 30;
	Observation.CurrentWeaponType = 0;
}

void UBTService_UpdateObservation::SyncToBlackboard(UBlackboardComponent* BlackboardComp, const FObservationElement& Observation)
{
	if (!BlackboardComp)
	{
		return;
	}

	// Sync key tactical information to Blackboard
	BlackboardComp->SetValueAsFloat(FName("ThreatLevel"), Observation.VisibleEnemyCount / 10.0f);
	BlackboardComp->SetValueAsBool(FName("bCanSeeEnemy"), Observation.VisibleEnemyCount > 0);
	BlackboardComp->SetValueAsFloat(FName("LastObservationUpdate"), BlackboardComp->GetWorld()->GetTimeSeconds());

	// If there's at least one enemy, set the nearest enemy as target
	if (Observation.NearbyEnemies.Num() > 0 && Observation.NearbyEnemies[0].Distance < MaxEnemyDetectionDistance)
	{
		// Note: We don't have the actual enemy actor reference here
		// This would need to be stored in the observation or retrieved differently
		// For now, we just update the boolean
	}

	// Update cover location if available
	if (Observation.bHasCover)
	{
		FVector CoverLocation = Observation.Position + FVector(
			Observation.CoverDirection.X,
			Observation.CoverDirection.Y,
			0.0f
		) * Observation.NearestCoverDistance;

		BlackboardComp->SetValueAsVector(FName("CoverLocation"), CoverLocation);
	}
}

ERaycastHitType UBTService_UpdateObservation::ClassifyHitType(const FHitResult& HitResult)
{
	if (!HitResult.GetActor())
	{
		return ERaycastHitType::Wall;
	}

	AActor* HitActor = HitResult.GetActor();

	// Check tags to classify the hit
	if (HitActor->ActorHasTag(EnemyTag))
	{
		return ERaycastHitType::Enemy;
	}

	if (HitActor->ActorHasTag(CoverTag))
	{
		return ERaycastHitType::Cover;
	}

	if (HitActor->ActorHasTag(FName("HealthPack")))
	{
		return ERaycastHitType::HealthPack;
	}

	if (HitActor->ActorHasTag(FName("Weapon")))
	{
		return ERaycastHitType::Weapon;
	}

	// Default to wall for static geometry
	return ERaycastHitType::Wall;
}
