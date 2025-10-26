// Fill out your copyright notice in the Description page of Project Settings.

#include "Actions/Movement/MoveToAllyAction.h"
#include "Core/StateMachine.h"
#include "GameFramework/Character.h"
#include "Kismet/GameplayStatics.h"

void UMoveToAllyAction::ExecuteAction(UStateMachine* StateMachine)
{
    UE_LOG(LogTemp, Warning, TEXT("------------MoveToAllyAction Executed"));

    if (!StateMachine)
    {
        UE_LOG(LogTemp, Error, TEXT("MoveToAllyAction: StateMachine is null"));
        return;
    }

    // Calculate optimal ally destination
    FVector AllyDestination = CalculateAllyDestination(StateMachine);

    if (AllyDestination.IsNearlyZero())
    {
        UE_LOG(LogTemp, Warning, TEXT("MoveToAllyAction: No valid ally destination found"));
        // Fallback: stay in place or move to a default location
        return;
    }

    // Set the destination in the StateMachine (which updates Blackboard)
    StateMachine->SetDestination(AllyDestination);

    // Get current observation for context
    FObservationElement CurrentObs = StateMachine->GetCurrentObservation();
    float DistanceToAlly = FVector::Dist(CurrentObs.Position, AllyDestination);

    UE_LOG(LogTemp, Display, TEXT("MoveToAllyAction: Moving to ally at distance %.2f"), DistanceToAlly);

    // Trigger Blueprint event for additional behavior (optional)
    // This allows designers to customize ally movement in Blueprint
    StateMachine->TriggerBlueprintEvent("MoveToAlly");

    // Note: Actual pathfinding and movement execution is handled by the Behavior Tree
    // This action only sets the strategic destination
}

FVector UMoveToAllyAction::CalculateAllyDestination(UStateMachine* StateMachine) const
{
    if (!StateMachine)
    {
        return FVector::ZeroVector;
    }

    ACharacter* OwnerCharacter = Cast<ACharacter>(StateMachine->GetOwner());
    if (!OwnerCharacter)
    {
        return FVector::ZeroVector;
    }

    UWorld* World = OwnerCharacter->GetWorld();
    if (!World)
    {
        return FVector::ZeroVector;
    }

    FObservationElement CurrentObs = StateMachine->GetCurrentObservation();
    FVector MyPosition = CurrentObs.Position;

    // Find all potential allies (characters with specific tag or team ID)
    // In a real game, you'd filter by team ID, faction, etc.
    TArray<AActor*> AllCharacters;
    UGameplayStatics::GetAllActorsOfClass(World, ACharacter::StaticClass(), AllCharacters);

    TArray<FVector> AllyPositions;
    for (AActor* Actor : AllCharacters)
    {
        if (Actor == OwnerCharacter)
        {
            continue;  // Skip self
        }

        ACharacter* PotentialAlly = Cast<ACharacter>(Actor);
        if (!PotentialAlly)
        {
            continue;
        }

        // Check if this character is an ally (you'd implement proper team checking here)
        // For now, we'll use a simple tag check
        if (PotentialAlly->ActorHasTag(FName("Ally")) || PotentialAlly->ActorHasTag(FName("Friendly")))
        {
            FVector AllyLocation = PotentialAlly->GetActorLocation();
            float Distance = FVector::Dist(MyPosition, AllyLocation);

            // Only consider allies within search radius
            if (Distance <= MaxAllySearchRadius)
            {
                AllyPositions.Add(AllyLocation);
            }
        }
    }

    if (AllyPositions.Num() == 0)
    {
        UE_LOG(LogTemp, Warning, TEXT("MoveToAllyAction: No allies found within range"));
        return FVector::ZeroVector;
    }

    // Strategy: Move toward the centroid of nearby allies
    // This encourages grouping behavior
    FVector Centroid = FVector::ZeroVector;
    for (const FVector& AllyPos : AllyPositions)
    {
        Centroid += AllyPos;
    }
    Centroid /= AllyPositions.Num();

    // Find a position near the centroid (maintain minimum distance)
    FVector DirectionToCentroid = (Centroid - MyPosition).GetSafeNormal();
    float DistanceToCentroid = FVector::Dist(MyPosition, Centroid);

    // If already close to allies, stay in place
    if (DistanceToCentroid < MinAllyDistance)
    {
        return MyPosition;
    }

    // Move toward centroid but stop at minimum distance
    float TargetDistance = FMath::Max(DistanceToCentroid - MinAllyDistance, 0.0f);
    FVector Destination = MyPosition + (DirectionToCentroid * TargetDistance);

    return Destination;
}

float UMoveToAllyAction::EvaluateAllyMovementPriority(UStateMachine* StateMachine) const
{
    if (!StateMachine)
    {
        return 0.0f;
    }

    FObservationElement CurrentObs = StateMachine->GetCurrentObservation();

    float Priority = 0.5f;  // Base priority

    // Higher priority if:
    // 1. Low health (want support from allies)
    if (CurrentObs.Health < 50.0f)
    {
        Priority += 0.2f;
    }

    // 2. Many enemies nearby (safety in numbers)
    if (CurrentObs.VisibleEnemyCount >= 3)
    {
        Priority += 0.2f;
    }

    // 3. Low stamina (need to regroup and recover)
    if (CurrentObs.Stamina < 30.0f)
    {
        Priority += 0.1f;
    }

    return FMath::Clamp(Priority, 0.0f, 1.0f);
}
