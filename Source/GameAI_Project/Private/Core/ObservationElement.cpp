// Fill out your copyright notice in the Description page of Project Settings.

#include "Core/ObservationElement.h"

TArray<float> FObservationElement::ToFeatureVector() const
{
    TArray<float> Features;
    Features.Reserve(71);

    // ========================================
    // AGENT STATE (12 features)
    // ========================================

    // Position (3)
    Features.Add(Position.X / 10000.0f);  // Normalize by typical map size
    Features.Add(Position.Y / 10000.0f);
    Features.Add(Position.Z / 10000.0f);

    // Velocity (3)
    Features.Add(Velocity.X / 1000.0f);  // Normalize by max expected velocity
    Features.Add(Velocity.Y / 1000.0f);
    Features.Add(Velocity.Z / 1000.0f);

    // Rotation (3) - Convert to normalized values
    Features.Add(Rotation.Pitch / 180.0f);  // -1 to 1
    Features.Add(Rotation.Yaw / 180.0f);
    Features.Add(Rotation.Roll / 180.0f);

    // Health, Stamina, Shield (3) - Already 0-100
    Features.Add(Health / 100.0f);     // Normalize to 0-1
    Features.Add(Stamina / 100.0f);
    Features.Add(Shield / 100.0f);

    // ========================================
    // COMBAT STATE (3 features)
    // ========================================

    Features.Add(FMath::Clamp(WeaponCooldown / 10.0f, 0.0f, 1.0f));  // Normalize by max cooldown
    Features.Add(FMath::Clamp(Ammunition / 100.0f, 0.0f, 1.0f));     // Normalize by max ammo
    Features.Add(CurrentWeaponType / 10.0f);  // Normalize weapon type (assuming max 10 weapon types)

    // ========================================
    // ENVIRONMENT PERCEPTION (32 features)
    // ========================================

    // Raycast Distances (16) - Already normalized 0-1
    for (int32 i = 0; i < 16; ++i)
    {
        Features.Add(i < RaycastDistances.Num() ? RaycastDistances[i] : 1.0f);
    }

    // Raycast Hit Types (16) - One-hot encode or normalize enum values
    for (int32 i = 0; i < 16; ++i)
    {
        if (i < RaycastHitTypes.Num())
        {
            Features.Add(static_cast<float>(RaycastHitTypes[i]) / 6.0f);  // Normalize enum (0-6)
        }
        else
        {
            Features.Add(0.0f);
        }
    }

    // ========================================
    // ENEMY INFORMATION (16 features)
    // ========================================

    // Visible Enemy Count (1)
    Features.Add(FMath::Clamp(VisibleEnemyCount / 20.0f, 0.0f, 1.0f));  // Normalize by max expected enemies

    // Nearby Enemies (5 × 3 = 15)
    for (int32 i = 0; i < 5; ++i)
    {
        if (i < NearbyEnemies.Num())
        {
            Features.Add(FMath::Clamp(NearbyEnemies[i].Distance / 5000.0f, 0.0f, 1.0f));  // Normalize distance
            Features.Add(NearbyEnemies[i].Health / 100.0f);  // Normalize health
            Features.Add((NearbyEnemies[i].RelativeAngle + 180.0f) / 360.0f);  // Normalize angle to 0-1
        }
        else
        {
            // No enemy data available - use neutral values
            Features.Add(1.0f);  // Max distance (no enemy)
            Features.Add(0.0f);  // No health
            Features.Add(0.5f);  // Neutral angle
        }
    }

    // ========================================
    // TACTICAL CONTEXT (5 features)
    // ========================================

    Features.Add(bHasCover ? 1.0f : 0.0f);  // Boolean to float
    Features.Add(FMath::Clamp(NearestCoverDistance / 5000.0f, 0.0f, 1.0f));  // Normalize distance
    Features.Add(CoverDirection.X);  // Already normalized -1 to 1
    Features.Add(CoverDirection.Y);
    Features.Add(static_cast<float>(CurrentTerrain) / 4.0f);  // Normalize terrain enum (0-4)

    // ========================================
    // TEMPORAL FEATURES (2 features)
    // ========================================

    Features.Add(FMath::Clamp(TimeSinceLastAction / 10.0f, 0.0f, 1.0f));  // Normalize by max expected time
    Features.Add(LastActionType / 20.0f);  // Normalize action type (assuming max 20 action types)

    // ========================================
    // LEGACY FIELDS (1 feature)
    // ========================================

    Features.Add(FMath::Clamp(DistanceToDestination / 5000.0f, 0.0f, 1.0f));  // Normalize distance

    // Verify we have exactly 71 features
    check(Features.Num() == 71);

    return Features;
}

void FObservationElement::Reset()
{
    // Agent State
    Position = FVector::ZeroVector;
    Velocity = FVector::ZeroVector;
    Rotation = FRotator::ZeroRotator;
    Health = 100.0f;
    Stamina = 100.0f;
    Shield = 0.0f;

    // Combat State
    WeaponCooldown = 0.0f;
    Ammunition = 0;
    CurrentWeaponType = 0;

    // Environment Perception
    RaycastDistances.Init(1.0f, 16);
    RaycastHitTypes.Init(ERaycastHitType::None, 16);

    // Enemy Information
    VisibleEnemyCount = 0;
    NearbyEnemies.Init(FEnemyObservation(), 5);

    // Tactical Context
    bHasCover = false;
    NearestCoverDistance = 0.0f;
    CoverDirection = FVector2D::ZeroVector;
    CurrentTerrain = ETerrainType::Unknown;

    // Temporal Features
    TimeSinceLastAction = 0.0f;
    LastActionType = 0;

    // Legacy
    DistanceToDestination = 0.0f;
}
