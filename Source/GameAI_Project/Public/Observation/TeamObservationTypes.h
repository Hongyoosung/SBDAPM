#pragma once

#include "CoreMinimal.h"
#include "TeamObservationTypes.generated.h"

/**
 * Engagement range classification
 */
UENUM(BlueprintType)
enum class EEngagementRange : uint8
{
    VeryClose   UMETA(DisplayName = "Very Close (< 5m)"),
    Close       UMETA(DisplayName = "Close (5-15m)"),
    Medium      UMETA(DisplayName = "Medium (15-30m)"),
    Long        UMETA(DisplayName = "Long (30-50m)"),
    VeryLong    UMETA(DisplayName = "Very Long (> 50m)")
};

/**
 * Objective type classification
 */
UENUM(BlueprintType)
enum class EObjectiveTypeLegacy : uint8
{
    None            UMETA(DisplayName = "No Objective"),
    Eliminate       UMETA(DisplayName = "Eliminate Enemies"),
    Defend          UMETA(DisplayName = "Defend Position"),
    Capture         UMETA(DisplayName = "Capture Point"),
    Escort          UMETA(DisplayName = "Escort Target"),
    Retrieve        UMETA(DisplayName = "Retrieve Item"),
    Survive         UMETA(DisplayName = "Survive")
};

/**
 * Mission phase classification
 */
UENUM(BlueprintType)
enum class EMissionPhase : uint8
{
    Preparation     UMETA(DisplayName = "Preparation"),
    Approach        UMETA(DisplayName = "Approach"),
    Engagement      UMETA(DisplayName = "Engagement"),
    Retreat         UMETA(DisplayName = "Retreat"),
    Complete        UMETA(DisplayName = "Complete"),
    Failed          UMETA(DisplayName = "Failed")
};
