// Fill out your copyright notice in the Description page of Project Settings.

#include "Actions/Movement/MoveToCoverAction.h"
#include "Core/StateMachine.h"
#include "GameFramework/Character.h"

void UMoveToCoverAction::ExecuteAction(UStateMachine* StateMachine)
{
    UE_LOG(LogTemp, Warning, TEXT("------------MoveToCoverAction Executed"));

    if (!StateMachine)
    {
        UE_LOG(LogTemp, Error, TEXT("MoveToCoverAction: StateMachine is null"));
        return;
    }

    // Calculate cover destination
    FVector CoverDestination = CalculateCoverDestination(StateMachine);

    if (CoverDestination.IsNearlyZero())
    {
        UE_LOG(LogTemp, Warning, TEXT("MoveToCoverAction: No cover available"));
        return;
    }

    // Set destination
    StateMachine->SetDestination(CoverDestination);

    FObservationElement CurrentObs = StateMachine->GetCurrentObservation();
    float DistanceToCover = FVector::Dist(CurrentObs.Position, CoverDestination);
    float Urgency = EvaluateCoverUrgency(StateMachine);

    UE_LOG(LogTemp, Display, TEXT("MoveToCoverAction: Distance=%.2f, Urgency=%.2f"),
           DistanceToCover, Urgency);

    // Trigger Blueprint event
    StateMachine->TriggerBlueprintEvent("MoveToCover");

    // Note: Actual pathfinding is handled by Behavior Tree
}

FVector UMoveToCoverAction::CalculateCoverDestination(UStateMachine* StateMachine) const
{
    if (!StateMachine)
    {
        return FVector::ZeroVector;
    }

    FObservationElement CurrentObs = StateMachine->GetCurrentObservation();
    FVector MyPosition = CurrentObs.Position;

    // Check if cover is available in observation
    if (!CurrentObs.bHasCover || CurrentObs.NearestCoverDistance <= 0.0f)
    {
        UE_LOG(LogTemp, Warning, TEXT("MoveToCoverAction: No cover detected in observation"));

        // Fallback: Try to find cover using raycasts
        return FindCoverUsingRaycasts(StateMachine);
    }

    // Calculate cover position from observation data
    FVector CoverDirection3D = FVector(CurrentObs.CoverDirection.X, CurrentObs.CoverDirection.Y, 0.0f).GetSafeNormal();
    FVector CoverPosition = MyPosition + (CoverDirection3D * CurrentObs.NearestCoverDistance);

    // Validate cover is within acceptable range
    if (CurrentObs.NearestCoverDistance > MaxCoverSearchRadius)
    {
        UE_LOG(LogTemp, Warning, TEXT("MoveToCoverAction: Cover too far away (%.2f)"),
               CurrentObs.NearestCoverDistance);
        return FVector::ZeroVector;
    }

    if (CurrentObs.NearestCoverDistance < MinCoverDistance)
    {
        UE_LOG(LogTemp, Display, TEXT("MoveToCoverAction: Already at cover"));
        return MyPosition;  // Already at cover
    }

    return CoverPosition;
}

FVector UMoveToCoverAction::FindCoverUsingRaycasts(UStateMachine* StateMachine) const
{
    if (!StateMachine)
    {
        return FVector::ZeroVector;
    }

    FObservationElement CurrentObs = StateMachine->GetCurrentObservation();
    FVector MyPosition = CurrentObs.Position;

    // Analyze raycast data to find cover
    if (CurrentObs.RaycastHitTypes.Num() != 16 || CurrentObs.RaycastDistances.Num() != 16)
    {
        UE_LOG(LogTemp, Warning, TEXT("MoveToCoverAction: Raycast data incomplete"));
        return FVector::ZeroVector;
    }

    // Find directions with cover objects
    TArray<int32> CoverDirections;
    for (int32 i = 0; i < CurrentObs.RaycastHitTypes.Num(); ++i)
    {
        if (CurrentObs.RaycastHitTypes[i] == ERaycastHitType::Cover)
        {
            CoverDirections.Add(i);
        }
    }

    if (CoverDirections.Num() == 0)
    {
        UE_LOG(LogTemp, Warning, TEXT("MoveToCoverAction: No cover found in raycasts"));
        return FVector::ZeroVector;
    }

    // Select the best cover direction
    // Strategy: Find cover that is:
    // 1. Not too close (> MinCoverDistance)
    // 2. Not too far (< MaxCoverSearchRadius)
    // 3. Preferably away from enemies

    int32 BestCoverIndex = -1;
    float BestCoverScore = -1.0f;

    for (int32 CoverDirIndex : CoverDirections)
    {
        // Calculate angle for this raycast (22.5° intervals, 16 rays)
        float Angle = CoverDirIndex * 22.5f;

        // Get distance (normalized 0-1, multiply by max raycast range)
        float MaxRaycastRange = 1000.0f;  // This should match your raycast implementation
        float CoverDistance = CurrentObs.RaycastDistances[CoverDirIndex] * MaxRaycastRange;

        // Skip if too close or too far
        if (CoverDistance < MinCoverDistance || CoverDistance > MaxCoverSearchRadius)
        {
            continue;
        }

        // Calculate score based on:
        // - Distance (prefer PreferredCoverDistance)
        // - Angle relative to enemies
        float DistanceScore = 1.0f - FMath::Abs(CoverDistance - PreferredCoverDistance) / PreferredCoverDistance;

        // Penalty if cover direction aligns with enemy direction
        float EnemyPenalty = 0.0f;
        if (CurrentObs.NearbyEnemies.Num() > 0)
        {
            float NearestEnemyAngle = CurrentObs.NearbyEnemies[0].RelativeAngle;

            // Normalize angles to 0-360
            float CoverAngle = Angle;
            float EnemyAngle = NearestEnemyAngle + 180.0f;  // Convert relative to absolute
            if (EnemyAngle < 0.0f) EnemyAngle += 360.0f;

            float AngleDiff = FMath::Abs(CoverAngle - EnemyAngle);
            if (AngleDiff > 180.0f) AngleDiff = 360.0f - AngleDiff;

            // Prefer cover opposite to enemy direction
            EnemyPenalty = (180.0f - AngleDiff) / 180.0f * 0.5f;  // Max penalty 0.5
        }

        float TotalScore = DistanceScore - EnemyPenalty;

        if (TotalScore > BestCoverScore)
        {
            BestCoverScore = TotalScore;
            BestCoverIndex = CoverDirIndex;
        }
    }

    if (BestCoverIndex == -1)
    {
        return FVector::ZeroVector;
    }

    // Calculate cover position
    float BestCoverAngle = BestCoverIndex * 22.5f;
    float MaxRaycastRange = 1000.0f;
    float BestCoverDistance = CurrentObs.RaycastDistances[BestCoverIndex] * MaxRaycastRange;

    FRotator MyRotation = CurrentObs.Rotation;
    float AbsoluteAngle = MyRotation.Yaw + BestCoverAngle;
    FVector DirectionToCover = FRotator(0.0f, AbsoluteAngle, 0.0f).Vector();
    FVector CoverPosition = MyPosition + (DirectionToCover * BestCoverDistance);

    return CoverPosition;
}

float UMoveToCoverAction::EvaluateCoverUrgency(UStateMachine* StateMachine) const
{
    if (!StateMachine)
    {
        return 0.0f;
    }

    FObservationElement CurrentObs = StateMachine->GetCurrentObservation();

    float Urgency = 0.0f;

    // Factor 1: Low health = high urgency
    if (CurrentObs.AgentHealth < 30.0f)
    {
        Urgency += 0.5f;
    }
    else if (CurrentObs.AgentHealth < 60.0f)
    {
        Urgency += 0.3f;
    }

    // Factor 2: Multiple enemies = higher urgency
    if (CurrentObs.VisibleEnemyCount >= 3)
    {
        Urgency += 0.3f;
    }
    else if (CurrentObs.VisibleEnemyCount >= 2)
    {
        Urgency += 0.2f;
    }

    // Factor 3: Close enemies = higher urgency
    if (CurrentObs.NearbyEnemies.Num() > 0)
    {
        float NearestEnemyDistance = CurrentObs.NearbyEnemies[0].Distance;
        if (NearestEnemyDistance < 300.0f)
        {
            Urgency += 0.3f;
        }
        else if (NearestEnemyDistance < 600.0f)
        {
            Urgency += 0.15f;
        }
    }

    // Factor 4: Low shield = higher urgency
    if (CurrentObs.Shield <= 0.0f && CurrentObs.AgentHealth < 80.0f)
    {
        Urgency += 0.2f;
    }

    // Factor 5: Weapon on cooldown = need cover to wait
    if (CurrentObs.WeaponCooldown > 2.0f)
    {
        Urgency += 0.1f;
    }

    return FMath::Clamp(Urgency, 0.0f, 1.0f);
}

bool UMoveToCoverAction::IsCoverCompromised(UStateMachine* StateMachine) const
{
    if (!StateMachine)
    {
        return false;
    }

    FObservationElement CurrentObs = StateMachine->GetCurrentObservation();

    // Cover is compromised if:
    // 1. Enemies are too close (flanking)
    if (CurrentObs.NearbyEnemies.Num() > 0)
    {
        for (const FEnemyObservation& Enemy : CurrentObs.NearbyEnemies)
        {
            if (Enemy.Distance > 0.0f && Enemy.Distance < 200.0f)
            {
                // Enemy is very close, cover is compromised
                return true;
            }

            // Check if enemy is flanking (angle > 90 degrees from forward)
            if (FMath::Abs(Enemy.RelativeAngle) > 90.0f && Enemy.Distance < 500.0f)
            {
                return true;
            }
        }
    }

    // 2. No cover detected in observation (cover destroyed or moved away)
    if (!CurrentObs.bHasCover)
    {
        return true;
    }

    // 3. Surrounded by enemies
    if (CurrentObs.VisibleEnemyCount >= 4)
    {
        return true;
    }

    return false;
}
