// SupportAllyObjective.h - Support ally teammate

#pragma once

#include "CoreMinimal.h"
#include "Team/Objective.h"
#include "SupportAllyObjective.generated.h"

/**
 * Objective: Provide covering fire and support to ally
 * Success: Keep ally alive, assist in eliminating threats
 * Failure: Ally dies, timeout
 */
UCLASS(Blueprintable)
class GAMEAI_PROJECT_API USupportAllyObjective : public UObjective
{
    GENERATED_BODY()

public:
    USupportAllyObjective();

    // Support parameters
    UPROPERTY(BlueprintReadWrite, Category = "Support")
    float SupportRange = 1000.0f;  // Max distance to provide support

    UPROPERTY(BlueprintReadWrite, Category = "Support")
    float IdealDistance = 500.0f;  // Optimal support distance

    // Tracking
    UPROPERTY(BlueprintReadOnly, Category = "Support")
    float TimeSupportingAlly = 0.0f;

    UPROPERTY(BlueprintReadOnly, Category = "Support")
    int32 ThreatsNeutralized = 0;

    UPROPERTY(BlueprintReadOnly, Category = "Support")
    float AllyHealthPercent = 1.0f;

    // Override base methods
    virtual void Activate() override;
    virtual bool CheckCompletion() override;
    virtual bool CheckFailure() override;
    virtual void UpdateProgress(float DeltaTime) override;
    virtual float CalculateStrategicReward() const override;

private:
    bool IsAllyAlive() const;
    float GetAllyHealthPercent() const;
    bool IsInSupportRange() const;
};
