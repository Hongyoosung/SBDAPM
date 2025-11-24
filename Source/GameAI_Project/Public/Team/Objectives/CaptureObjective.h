// CaptureObjective.h - Capture zone/flag objective

#pragma once

#include "CoreMinimal.h"
#include "Team/Objective.h"
#include "CaptureObjective.generated.h"

/**
 * Objective: Capture a zone or flag
 * Success: All assigned agents reach and hold the zone
 * Failure: Timeout, all agents die
 */
UCLASS(Blueprintable)
class GAMEAI_PROJECT_API UCaptureObjective : public UObjective
{
    GENERATED_BODY()

public:
    UCaptureObjective();

    // Capture parameters
    UPROPERTY(BlueprintReadWrite, Category = "Capture")
    float CaptureRadius = 500.0f;  // Distance to be "in zone"

    UPROPERTY(BlueprintReadWrite, Category = "Capture")
    float CaptureTime = 10.0f;  // Time agents must stay in zone

    UPROPERTY(BlueprintReadWrite, Category = "Capture")
    int32 MinAgentsRequired = 1;  // Minimum agents to capture

    // Tracking
    UPROPERTY(BlueprintReadOnly, Category = "Capture")
    float TimeInZone = 0.0f;

    UPROPERTY(BlueprintReadOnly, Category = "Capture")
    int32 AgentsInZone = 0;

    UPROPERTY(BlueprintReadOnly, Category = "Capture")
    bool bIsContested = false;  // Enemies also in zone

    // Override base methods
    virtual bool CheckCompletion() override;
    virtual bool CheckFailure() override;
    virtual void UpdateProgress(float DeltaTime) override;
    virtual float CalculateStrategicReward() const override;

private:
    int32 CountAgentsInZone() const;
    bool CheckIfContested() const;
};
