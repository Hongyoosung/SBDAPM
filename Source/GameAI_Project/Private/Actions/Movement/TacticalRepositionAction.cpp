// Fill out your copyright notice in the Description page of Project Settings.

#include "Actions/Movement/TacticalRepositionAction.h"
#include "Core/StateMachine.h"
#include "GameFramework/Character.h"

void UTacticalRepositionAction::ExecuteAction(UStateMachine* StateMachine)
{
    UE_LOG(LogTemp, Warning, TEXT("------------TacticalRepositionAction Executed"));

    if (!StateMachine)
    {
        UE_LOG(LogTemp, Error, TEXT("TacticalRepositionAction: StateMachine is null"));
        return;
    }

    // Check if repositioning is necessary
    if (!ShouldReposition(StateMachine))
    {
        UE_LOG(LogTemp, Display, TEXT("TacticalRepositionAction: Current position is good, no reposition needed"));
        return;
    }

    // Calculate reposition destination
    FVector RepositionDestination = CalculateRepositionDestination(StateMachine);

    if (RepositionDestination.IsNearlyZero())
    {
        UE_LOG(LogTemp, Warning, TEXT("TacticalRepositionAction: No valid reposition destination found"));
        return;
    }

    // Set destination
    StateMachine->SetDestination(RepositionDestination);

    FObservationElement CurrentObs = StateMachine->GetCurrentObservation();
    float DistanceToReposition = FVector::Dist(CurrentObs.Position, RepositionDestination);
    float CurrentPositionQuality = EvaluateCurrentPosition(StateMachine);

    UE_LOG(LogTemp, Display, TEXT("TacticalRepositionAction: Distance=%.2f, CurrentQuality=%.2f"),
           DistanceToReposition, CurrentPositionQuality);

    // Trigger Blueprint event
    StateMachine->TriggerBlueprintEvent("TacticalReposition");
}

float UTacticalRepositionAction::EvaluateCurrentPosition(UStateMachine* StateMachine) const
{
    if (!StateMachine)
    {
        return 0.0f;
    }

    FObservationElement CurrentObs = StateMachine->GetCurrentObservation();
    float Quality = 0.5f;  // Base quality

    // Factor 1: Cover availability (weighted by CoverWeight)
    if (CurrentObs.bHasCover)
    {
        // Closer cover is better
        float CoverQuality = 1.0f;
        if (CurrentObs.NearestCoverDistance > 0.0f)
        {
            CoverQuality = FMath::Clamp(1.0f - (CurrentObs.NearestCoverDistance / 500.0f), 0.0f, 1.0f);
        }
        Quality += CoverQuality * CoverWeight;
    }
    else
    {
        // No cover is bad
        Quality -= CoverWeight * 0.5f;
    }

    // Factor 2: Enemy proximity (weighted by EnemyAvoidanceWeight)
    if (CurrentObs.NearbyEnemies.Num() > 0)
    {
        float NearestEnemyDistance = CurrentObs.NearbyEnemies[0].Distance;

        // Too close to enemies is bad
        if (NearestEnemyDistance < 300.0f)
        {
            Quality -= EnemyAvoidanceWeight;
        }
        else if (NearestEnemyDistance < 600.0f)
        {
            Quality -= EnemyAvoidanceWeight * 0.5f;
        }

        // Being flanked (enemies at wide angles) is very bad
        int32 FlankingEnemies = 0;
        for (const FEnemyObservation& Enemy : CurrentObs.NearbyEnemies)
        {
            if (Enemy.Distance > 0.0f && FMath::Abs(Enemy.RelativeAngle) > 90.0f)
            {
                FlankingEnemies++;
            }
        }

        if (FlankingEnemies >= 2)
        {
            Quality -= 0.3f;  // Heavily penalize being flanked
        }
    }

    // Factor 3: Visibility (based on raycasts - weighted by VisibilityWeight)
    if (CurrentObs.RaycastDistances.Num() > 0)
    {
        // Calculate average raycast distance (good visibility = higher average)
        float TotalDistance = 0.0f;
        for (float Distance : CurrentObs.RaycastDistances)
        {
            TotalDistance += Distance;
        }
        float AverageDistance = TotalDistance / CurrentObs.RaycastDistances.Num();

        // Higher average distance = better visibility
        Quality += AverageDistance * VisibilityWeight;
    }

    // Factor 4: Health/Shield status (lower health = lower position quality)
    if (CurrentObs.Health < 50.0f && !CurrentObs.bHasCover)
    {
        Quality -= 0.2f;  // Low health without cover is very bad
    }

    // Factor 5: Surrounded (multiple enemies from different directions)
    if (CurrentObs.VisibleEnemyCount >= 3)
    {
        Quality -= 0.25f;
    }

    return FMath::Clamp(Quality, 0.0f, 1.0f);
}

FVector UTacticalRepositionAction::CalculateRepositionDestination(UStateMachine* StateMachine) const
{
    if (!StateMachine)
    {
        return FVector::ZeroVector;
    }

    FObservationElement CurrentObs = StateMachine->GetCurrentObservation();

    // Strategy 1: If under heavy fire (low health, many enemies), find safe position
    if (CurrentObs.Health < 40.0f || CurrentObs.VisibleEnemyCount >= 3)
    {
        return FindSafePosition(StateMachine);
    }

    // Strategy 2: If flanked, move to better position
    bool bIsFlanked = false;
    for (const FEnemyObservation& Enemy : CurrentObs.NearbyEnemies)
    {
        if (Enemy.Distance > 0.0f && FMath::Abs(Enemy.RelativeAngle) > 120.0f && Enemy.Distance < 500.0f)
        {
            bIsFlanked = true;
            break;
        }
    }

    if (bIsFlanked)
    {
        // Move perpendicular to threat
        return FindSafePosition(StateMachine);
    }

    // Strategy 3: General tactical optimization
    return CalculateOptimalTacticalPosition(StateMachine);
}

bool UTacticalRepositionAction::ShouldReposition(UStateMachine* StateMachine) const
{
    if (!StateMachine)
    {
        return false;
    }

    float PositionQuality = EvaluateCurrentPosition(StateMachine);

    // Reposition if position quality is below threshold
    if (PositionQuality < RepositionThreshold)
    {
        return true;
    }

    FObservationElement CurrentObs = StateMachine->GetCurrentObservation();

    // Force reposition if in immediate danger
    if (CurrentObs.Health < 30.0f && !CurrentObs.bHasCover)
    {
        return true;
    }

    // Force reposition if flanked by multiple enemies
    int32 FlankingEnemies = 0;
    for (const FEnemyObservation& Enemy : CurrentObs.NearbyEnemies)
    {
        if (Enemy.Distance > 0.0f && FMath::Abs(Enemy.RelativeAngle) > 90.0f && Enemy.Distance < 400.0f)
        {
            FlankingEnemies++;
        }
    }

    if (FlankingEnemies >= 2)
    {
        return true;
    }

    return false;
}

FVector UTacticalRepositionAction::FindSafePosition(UStateMachine* StateMachine) const
{
    if (!StateMachine)
    {
        return FVector::ZeroVector;
    }

    FObservationElement CurrentObs = StateMachine->GetCurrentObservation();
    FVector MyPosition = CurrentObs.Position;

    // Analyze threat vectors
    TArray<FVector> ThreatVectors = AnalyzeThreatVectors(CurrentObs);

    if (ThreatVectors.Num() == 0)
    {
        // No threats, stay in place or move toward cover
        if (CurrentObs.bHasCover && CurrentObs.NearestCoverDistance > 100.0f)
        {
            FVector CoverDirection3D = FVector(CurrentObs.CoverDirection.X, CurrentObs.CoverDirection.Y, 0.0f).GetSafeNormal();
            return MyPosition + (CoverDirection3D * CurrentObs.NearestCoverDistance);
        }
        return MyPosition;
    }

    // Calculate average threat direction
    FVector AverageThreat = FVector::ZeroVector;
    for (const FVector& Threat : ThreatVectors)
    {
        AverageThreat += Threat;
    }
    AverageThreat /= ThreatVectors.Num();
    AverageThreat.Normalize();

    // Move away from average threat direction
    FVector SafeDirection = -AverageThreat;

    // If cover is available, bias toward cover
    if (CurrentObs.bHasCover)
    {
        FVector CoverDirection3D = FVector(CurrentObs.CoverDirection.X, CurrentObs.CoverDirection.Y, 0.0f).GetSafeNormal();

        // Blend safe direction with cover direction
        SafeDirection = (SafeDirection + CoverDirection3D * 2.0f).GetSafeNormal();
    }

    // Calculate safe position
    float RepositionDistance = FMath::Clamp(
        (1.0f - CurrentObs.Health / 100.0f) * MaxRepositionDistance,
        MinRepositionDistance,
        MaxRepositionDistance
    );

    FVector SafePosition = MyPosition + (SafeDirection * RepositionDistance);

    return SafePosition;
}

FVector UTacticalRepositionAction::CalculateOptimalTacticalPosition(UStateMachine* StateMachine) const
{
    if (!StateMachine)
    {
        return FVector::ZeroVector;
    }

    FObservationElement CurrentObs = StateMachine->GetCurrentObservation();
    FVector MyPosition = CurrentObs.Position;

    // Sample multiple candidate positions in a circle around current position
    TArray<FVector> CandidatePositions;
    int32 NumSamples = 12;
    float SampleRadius = (MinRepositionDistance + MaxRepositionDistance) / 2.0f;

    for (int32 i = 0; i < NumSamples; ++i)
    {
        float Angle = (360.0f / NumSamples) * i;
        FVector Direction = FRotator(0.0f, Angle, 0.0f).Vector();
        FVector CandidatePosition = MyPosition + (Direction * SampleRadius);

        CandidatePositions.Add(CandidatePosition);
    }

    // Evaluate each candidate position
    FVector BestPosition = MyPosition;
    float BestScore = -1.0f;

    for (const FVector& Candidate : CandidatePositions)
    {
        float Score = 0.0f;

        // Factor 1: Distance from enemies (prefer farther, but not too far)
        if (CurrentObs.NearbyEnemies.Num() > 0)
        {
            float NearestEnemyAngle = CurrentObs.NearbyEnemies[0].RelativeAngle;
            float NearestEnemyDistance = CurrentObs.NearbyEnemies[0].Distance;

            FRotator MyRotation = CurrentObs.Rotation;
            float AbsoluteAngle = MyRotation.Yaw + NearestEnemyAngle;
            FVector DirectionToEnemy = FRotator(0.0f, AbsoluteAngle, 0.0f).Vector();
            FVector EnemyPosition = MyPosition + (DirectionToEnemy * NearestEnemyDistance);

            float DistanceToEnemy = FVector::Dist(Candidate, EnemyPosition);

            // Prefer positions at medium range (400-800 units)
            if (DistanceToEnemy >= 400.0f && DistanceToEnemy <= 800.0f)
            {
                Score += EnemyAvoidanceWeight;
            }
            else if (DistanceToEnemy > 800.0f)
            {
                Score += EnemyAvoidanceWeight * 0.5f;
            }
        }

        // Factor 2: Cover proximity (simulated - real implementation would use nav queries)
        if (CurrentObs.bHasCover)
        {
            FVector CoverDirection3D = FVector(CurrentObs.CoverDirection.X, CurrentObs.CoverDirection.Y, 0.0f).GetSafeNormal();
            FVector CoverPosition = MyPosition + (CoverDirection3D * CurrentObs.NearestCoverDistance);

            float DistanceToCover = FVector::Dist(Candidate, CoverPosition);
            if (DistanceToCover < 300.0f)
            {
                Score += CoverWeight;
            }
        }

        // Factor 3: Not moving into worse position (not toward enemies)
        if (CurrentObs.NearbyEnemies.Num() > 0)
        {
            FVector MoveDirection = (Candidate - MyPosition).GetSafeNormal();

            float NearestEnemyAngle = CurrentObs.NearbyEnemies[0].RelativeAngle;
            FRotator MyRotation = CurrentObs.Rotation;
            float AbsoluteAngle = MyRotation.Yaw + NearestEnemyAngle;
            FVector DirectionToEnemy = FRotator(0.0f, AbsoluteAngle, 0.0f).Vector();

            float DotProduct = FVector::DotProduct(MoveDirection, DirectionToEnemy);

            // Penalize moving toward enemy
            if (DotProduct > 0.0f)
            {
                Score -= 0.3f;
            }
        }

        if (Score > BestScore)
        {
            BestScore = Score;
            BestPosition = Candidate;
        }
    }

    // If no good position found, stay in place
    if (BestScore < 0.0f)
    {
        return MyPosition;
    }

    return BestPosition;
}

TArray<FVector> UTacticalRepositionAction::AnalyzeThreatVectors(const FObservationElement& Observation) const
{
    TArray<FVector> ThreatVectors;

    // Add threat vector for each nearby enemy
    for (const FEnemyObservation& Enemy : Observation.NearbyEnemies)
    {
        if (Enemy.Distance > 0.0f)
        {
            // Calculate direction to enemy
            float AbsoluteAngle = Observation.Rotation.Yaw + Enemy.RelativeAngle;
            FVector DirectionToEnemy = FRotator(0.0f, AbsoluteAngle, 0.0f).Vector();

            // Weight by proximity (closer enemies are bigger threats)
            float ThreatWeight = 1.0f - FMath::Clamp(Enemy.Distance / 1000.0f, 0.0f, 1.0f);

            ThreatVectors.Add(DirectionToEnemy * ThreatWeight);
        }
    }

    return ThreatVectors;
}
