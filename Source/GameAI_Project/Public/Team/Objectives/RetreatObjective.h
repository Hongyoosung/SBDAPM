// RetreatObjective.h - Retreat to safe location

#pragma once

#include "CoreMinimal.h"
#include "Team/Objective.h"
#include "RetreatObjective.generated.h"

/**
 * Objective: Fall back to a safe location away from enemies
 * Success: All assigned agents reach safe zone
 * Failure: Timeout, all agents die
 */
UCLASS(Blueprintable)
class GAMEAI_PROJECT_API URetreatObjective : public UObjective
{
    GENERATED_BODY()

public:
    URetreatObjective();

    // Retreat parameters
    UPROPERTY(BlueprintReadWrite, Category = "Retreat")
    float SafeRadius = 400.0f;  // Distance to be "safe"

    UPROPERTY(BlueprintReadWrite, Category = "Retreat")
    float MinDistanceFromEnemy = 2000.0f;  // Minimum safe distance from enemies

    // Tracking
    UPROPERTY(BlueprintReadOnly, Category = "Retreat")
    int32 AgentsAtSafeZone = 0;

    UPROPERTY(BlueprintReadOnly, Category = "Retreat")
    bool bIsSafe = false;  // No enemies nearby

    // Override base methods
    virtual bool CheckCompletion() override;
    virtual bool CheckFailure() override;
    virtual void UpdateProgress(float DeltaTime) override;
    virtual float CalculateStrategicReward() const override;

private:
    int32 CountAgentsAtSafeZone() const;
    bool CheckIfSafe() const;
};
