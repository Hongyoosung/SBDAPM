// Fill out your copyright notice in the Description page of Project Settings.

#include "Actions/Movement/MoveToAttackPositionAction.h"
#include "Core/StateMachine.h"
#include "GameFramework/Character.h"
#include "Kismet/GameplayStatics.h"
#include "DrawDebugHelpers.h"

void UMoveToAttackPositionAction::ExecuteAction(UStateMachine* StateMachine)
{
    UE_LOG(LogTemp, Warning, TEXT("------------MoveToAttackPositionAction Executed"));

    if (!StateMachine)
    {
        UE_LOG(LogTemp, Error, TEXT("MoveToAttackPositionAction: StateMachine is null"));
        return;
    }

    // Select best strategy for current situation
    EAttackPositionStrategy Strategy = SelectBestStrategy(StateMachine);

    // Calculate optimal attack position based on strategy
    FVector AttackPosition = CalculateAttackPosition(StateMachine, Strategy);

    if (AttackPosition.IsNearlyZero())
    {
        UE_LOG(LogTemp, Warning, TEXT("MoveToAttackPositionAction: No valid attack position found"));
        return;
    }

    // Set the destination in the StateMachine
    StateMachine->SetDestination(AttackPosition);

    FObservationElement CurrentObs = StateMachine->GetCurrentObservation();
    float DistanceToPosition = FVector::Dist(CurrentObs.Position, AttackPosition);

    UE_LOG(LogTemp, Display, TEXT("MoveToAttackPositionAction: Strategy=%d, Distance=%.2f"),
           static_cast<int32>(Strategy), DistanceToPosition);

    // Trigger Blueprint event
    StateMachine->TriggerBlueprintEvent("MoveToAttackPosition");

    // Note: Actual pathfinding is handled by Behavior Tree
}

FVector UMoveToAttackPositionAction::CalculateAttackPosition(UStateMachine* StateMachine,
                                                              EAttackPositionStrategy Strategy) const
{
    if (!StateMachine)
    {
        return FVector::ZeroVector;
    }

    FObservationElement CurrentObs = StateMachine->GetCurrentObservation();

    // Find nearest enemy to target
    if (CurrentObs.NearbyEnemies.Num() == 0 || CurrentObs.NearbyEnemies[0].Distance <= 0.0f)
    {
        UE_LOG(LogTemp, Warning, TEXT("MoveToAttackPositionAction: No enemies detected"));
        return FVector::ZeroVector;
    }

    // Get target enemy position (use nearest enemy)
    FEnemyObservation NearestEnemy = CurrentObs.NearbyEnemies[0];
    float EnemyAngle = NearestEnemy.RelativeAngle;
    float EnemyDistance = NearestEnemy.Distance;

    // Calculate enemy world position from relative angle and distance
    FVector MyPosition = CurrentObs.Position;
    FRotator MyRotation = CurrentObs.Rotation;

    float AbsoluteAngle = MyRotation.Yaw + EnemyAngle;
    FVector DirectionToEnemy = FRotator(0.0f, AbsoluteAngle, 0.0f).Vector();
    FVector EnemyPosition = MyPosition + (DirectionToEnemy * EnemyDistance);

    // Enemy's estimated rotation (facing toward us)
    FRotator EnemyRotation = (MyPosition - EnemyPosition).Rotation();

    // Calculate position based on strategy
    FVector Destination;
    switch (Strategy)
    {
        case EAttackPositionStrategy::Flanking:
            Destination = CalculateFlankingPosition(MyPosition, EnemyPosition, EnemyRotation);
            break;

        case EAttackPositionStrategy::HighGround:
            Destination = CalculateHighGroundPosition(StateMachine, EnemyPosition);
            break;

        case EAttackPositionStrategy::CoverBased:
            Destination = CalculateCoverBasedPosition(StateMachine, EnemyPosition);
            break;

        case EAttackPositionStrategy::DirectAssault:
            {
                // Move directly toward enemy at optimal range
                FVector DirectionToTarget = (EnemyPosition - MyPosition).GetSafeNormal();
                Destination = EnemyPosition - (DirectionToTarget * OptimalAttackRange);
            }
            break;

        case EAttackPositionStrategy::Encircle:
            {
                // Move to encircle enemy (perpendicular to current position)
                FVector ToEnemy = EnemyPosition - MyPosition;
                FVector Perpendicular = FVector(-ToEnemy.Y, ToEnemy.X, 0.0f).GetSafeNormal();
                Destination = EnemyPosition + (Perpendicular * OptimalAttackRange);
            }
            break;

        default:
            Destination = FVector::ZeroVector;
            break;
    }

    return Destination;
}

EAttackPositionStrategy UMoveToAttackPositionAction::SelectBestStrategy(UStateMachine* StateMachine) const
{
    if (!StateMachine)
    {
        return EAttackPositionStrategy::DirectAssault;
    }

    FObservationElement CurrentObs = StateMachine->GetCurrentObservation();

    // Strategy selection based on observations

    // If low health, prioritize cover
    if (CurrentObs.AgentHealth < 40.0f && CurrentObs.bHasCover)
    {
        return EAttackPositionStrategy::CoverBased;
    }

    // If multiple enemies, try to flank
    if (CurrentObs.VisibleEnemyCount >= 2)
    {
        return EAttackPositionStrategy::Flanking;
    }

    // If enemy is close, maintain distance or flank
    if (CurrentObs.NearbyEnemies.Num() > 0 && CurrentObs.NearbyEnemies[0].Distance < MinAttackRange)
    {
        return EAttackPositionStrategy::Flanking;
    }

    // If high stamina and full health, go for high ground
    if (CurrentObs.AgentHealth > 80.0f && CurrentObs.Stamina > 70.0f)
    {
        return EAttackPositionStrategy::HighGround;
    }

    // Default: direct assault
    return EAttackPositionStrategy::DirectAssault;
}

float UMoveToAttackPositionAction::EvaluateAttackPosition(const FVector& Position, const FVector& EnemyPosition,
                                                           const FVector& MyPosition, const FObservationElement& CurrentObs) const
{
    float Score = 0.5f;  // Base score

    // Factor 1: Distance to enemy (prefer optimal range)
    float DistanceToEnemy = FVector::Dist(Position, EnemyPosition);
    float DistanceScore = 1.0f - FMath::Abs(DistanceToEnemy - OptimalAttackRange) / OptimalAttackRange;
    Score += DistanceScore * 0.3f;

    // Factor 2: Height advantage
    float HeightDiff = Position.Z - EnemyPosition.Z;
    if (HeightDiff > 0.0f)
    {
        Score += FMath::Min(HeightDiff / HighGroundHeightBonus, 1.0f) * 0.2f;
    }

    // Factor 3: Cover availability (simplified - would check actual cover in real implementation)
    if (CurrentObs.bHasCover && FVector::Dist(Position, MyPosition + CurrentObs.Rotation.Vector() * CurrentObs.NearestCoverDistance) < 200.0f)
    {
        Score += 0.2f;
    }

    // Factor 4: Not too close to enemy (safety)
    if (DistanceToEnemy < MinAttackRange)
    {
        Score -= 0.3f;
    }

    return FMath::Clamp(Score, 0.0f, 1.0f);
}

FVector UMoveToAttackPositionAction::CalculateFlankingPosition(const FVector& MyPosition, const FVector& EnemyPosition,
                                                                const FRotator& EnemyRotation) const
{
    // Calculate position to the side or rear of the enemy

    FVector EnemyForward = EnemyRotation.Vector();
    FVector ToMe = (MyPosition - EnemyPosition).GetSafeNormal();

    // Calculate angle between enemy's forward and direction to me
    float DotProduct = FVector::DotProduct(EnemyForward, ToMe);
    float AngleDegrees = FMath::RadiansToDegrees(FMath::Acos(DotProduct));

    // If already flanking, maintain position
    if (AngleDegrees >= FlankingAngleMin)
    {
        return MyPosition;
    }

    // Move to enemy's side (perpendicular to their forward direction)
    FVector EnemyRight = FVector::CrossProduct(EnemyForward, FVector::UpVector).GetSafeNormal();

    // Choose left or right side based on current position
    float CrossZ = FVector::CrossProduct(EnemyForward, ToMe).Z;
    FVector FlankDirection = (CrossZ > 0.0f) ? EnemyRight : -EnemyRight;

    // Position at optimal range on the flank
    FVector FlankPosition = EnemyPosition + (FlankDirection * OptimalAttackRange);

    return FlankPosition;
}

FVector UMoveToAttackPositionAction::CalculateHighGroundPosition(UStateMachine* StateMachine, const FVector& EnemyPosition) const
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

    FObservationElement CurrentObs = StateMachine->GetCurrentObservation();
    FVector MyPosition = CurrentObs.Position;

    // Search for elevated positions using raycasts
    // This is a simplified version - real implementation would use navigation queries

    TArray<FVector> CandidatePositions;
    int32 NumSamples = 8;  // Sample positions around enemy in a circle

    for (int32 i = 0; i < NumSamples; ++i)
    {
        float Angle = (360.0f / NumSamples) * i;
        FVector Direction = FRotator(0.0f, Angle, 0.0f).Vector();
        FVector SamplePosition = EnemyPosition + (Direction * OptimalAttackRange);

        // Add height bonus
        SamplePosition.Z += HighGroundHeightBonus;

        CandidatePositions.Add(SamplePosition);
    }

    // Find the highest valid position
    FVector BestPosition = MyPosition;
    float MaxHeight = MyPosition.Z;

    for (const FVector& Candidate : CandidatePositions)
    {
        if (Candidate.Z > MaxHeight)
        {
            // In real implementation, would check if position is reachable via navigation
            MaxHeight = Candidate.Z;
            BestPosition = Candidate;
        }
    }

    // If no higher position found, return position at optimal range
    if (BestPosition.Equals(MyPosition))
    {
        FVector ToEnemy = (EnemyPosition - MyPosition).GetSafeNormal();
        BestPosition = EnemyPosition - (ToEnemy * OptimalAttackRange);
    }

    return BestPosition;
}

FVector UMoveToAttackPositionAction::CalculateCoverBasedPosition(UStateMachine* StateMachine, const FVector& EnemyPosition) const
{
    if (!StateMachine)
    {
        return FVector::ZeroVector;
    }

    FObservationElement CurrentObs = StateMachine->GetCurrentObservation();

    // If cover is available, move toward it
    if (CurrentObs.bHasCover && CurrentObs.NearestCoverDistance > 0.0f)
    {
        FVector MyPosition = CurrentObs.Position;

        // Direction to cover (from CoverDirection 2D vector)
        FVector CoverDirection3D = FVector(CurrentObs.CoverDirection.X, CurrentObs.CoverDirection.Y, 0.0f).GetSafeNormal();
        FVector CoverPosition = MyPosition + (CoverDirection3D * CurrentObs.NearestCoverDistance);

        // Ensure cover position has line of sight to enemy
        // In real implementation, would perform line trace to verify
        FVector ToCover = (CoverPosition - MyPosition).GetSafeNormal();
        FVector ToEnemy = (EnemyPosition - CoverPosition).GetSafeNormal();

        // Adjust cover position to maintain optimal attack range
        float DistanceFromCoverToEnemy = FVector::Dist(CoverPosition, EnemyPosition);
        if (DistanceFromCoverToEnemy > OptimalAttackRange * 1.5f)
        {
            // Cover is too far from enemy, move closer
            FVector AdjustedPosition = CoverPosition + (ToEnemy * (DistanceFromCoverToEnemy - OptimalAttackRange));
            return AdjustedPosition;
        }

        return CoverPosition;
    }

    // No cover available, fall back to direct assault position
    FVector MyPosition = CurrentObs.Position;
    FVector ToEnemy = (EnemyPosition - MyPosition).GetSafeNormal();
    return EnemyPosition - (ToEnemy * OptimalAttackRange);
}
