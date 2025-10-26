// Fill out your copyright notice in the Description page of Project Settings.

#pragma once

#include "CoreMinimal.h"
#include "UObject/NoExportTypes.h"
#include "ObservationElement.generated.h"

/**
 * Represents a single nearby enemy's information
 */
USTRUCT(BlueprintType)
struct FEnemyObservation
{
    GENERATED_BODY()

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Enemy Info")
    float Distance = 0.0f;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Enemy Info")
    float Health = 0.0f;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Enemy Info")
    float RelativeAngle = 0.0f;  // Angle relative to agent's forward direction (-180 to 180)
};

/**
 * Enum for different object types detected by raycasts
 */
UENUM(BlueprintType)
enum class ERaycastHitType : uint8
{
    None = 0,
    Wall = 1,
    Enemy = 2,
    Cover = 3,
    HealthPack = 4,
    Weapon = 5,
    Other = 6
};

/**
 * Enum for terrain types
 */
UENUM(BlueprintType)
enum class ETerrainType : uint8
{
    Flat = 0,
    Inclined = 1,
    Rough = 2,
    Water = 3,
    Unknown = 4
};

/**
 * Enhanced observation structure with ~70 features for deep RL
 * Replaces the original 3-feature observation space
 */
USTRUCT(BlueprintType)
struct FObservationElement
{
    GENERATED_BODY()

    // ========================================
    // AGENT STATE (12 features)
    // ========================================

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Agent State|Position")
    FVector Position = FVector::ZeroVector;  // 3 features: X, Y, Z

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Agent State|Movement")
    FVector Velocity = FVector::ZeroVector;  // 3 features: VX, VY, VZ

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Agent State|Orientation")
    FRotator Rotation = FRotator::ZeroRotator;  // 3 features: Pitch, Yaw, Roll

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Agent State|Health")
    float Health = 100.0f;  // 1 feature: 0-100

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Agent State|Stamina")
    float Stamina = 100.0f;  // 1 feature: 0-100

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Agent State|Defense")
    float Shield = 0.0f;  // 1 feature: 0-100

    // ========================================
    // COMBAT STATE (3 features)
    // ========================================

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Combat|Weapon")
    float WeaponCooldown = 0.0f;  // 1 feature: seconds remaining

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Combat|Ammo")
    int32 Ammunition = 0;  // 1 feature: bullets/charges remaining

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Combat|Weapon")
    int32 CurrentWeaponType = 0;  // 1 feature: weapon ID (0=unarmed, 1=pistol, 2=rifle, etc.)

    // ========================================
    // ENVIRONMENT PERCEPTION (32 features)
    // ========================================

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Perception|Raycasts")
    TArray<float> RaycastDistances;  // 16 features: normalized distances (0-1) for 16 rays at 22.5° intervals

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Perception|Raycasts")
    TArray<ERaycastHitType> RaycastHitTypes;  // 16 features: object type at each raycast direction

    // ========================================
    // ENEMY INFORMATION (16 features)
    // ========================================

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Enemies|Count")
    int32 VisibleEnemyCount = 0;  // 1 feature: total visible enemies

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Enemies|Nearby")
    TArray<FEnemyObservation> NearbyEnemies;  // 5 × 3 = 15 features (top 5 closest enemies)

    // ========================================
    // TACTICAL CONTEXT (5 features)
    // ========================================

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Tactical|Cover")
    bool bHasCover = false;  // 1 feature: is cover available nearby?

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Tactical|Cover")
    float NearestCoverDistance = 0.0f;  // 1 feature: distance to nearest cover

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Tactical|Cover")
    FVector2D CoverDirection = FVector2D::ZeroVector;  // 2 features: normalized X,Y direction to cover

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Tactical|Terrain")
    ETerrainType CurrentTerrain = ETerrainType::Unknown;  // 1 feature: terrain type enum

    // ========================================
    // TEMPORAL FEATURES (2 features)
    // ========================================

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Temporal")
    float TimeSinceLastAction = 0.0f;  // 1 feature: seconds since last action

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Temporal")
    int32 LastActionType = 0;  // 1 feature: ID of last action taken

    // ========================================
    // LEGACY FIELDS (1 feature - for backward compatibility)
    // ========================================

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Legacy")
    float DistanceToDestination = 0.0f;  // 1 feature: distance to goal

    // Total: 12 + 3 + 32 + 16 + 5 + 2 + 1 = 71 features

    /**
     * Constructor with default initialization
     */
    FObservationElement()
    {
        // Initialize raycast arrays with default values
        RaycastDistances.Init(1.0f, 16);  // 16 rays, max distance = 1.0 (normalized)
        RaycastHitTypes.Init(ERaycastHitType::None, 16);

        // Initialize enemy array with empty observations
        NearbyEnemies.Init(FEnemyObservation(), 5);  // Track top 5 closest enemies
    }

    /**
     * Get total number of features for neural network input
     */
    int32 GetFeatureCount() const
    {
        return 71;
    }

    /**
     * Flatten observation into a single array for ML input
     * @return Array of 71 normalized float values
     */
    TArray<float> ToFeatureVector() const;

    /**
     * Reset observation to default values
     */
    void Reset();
};
