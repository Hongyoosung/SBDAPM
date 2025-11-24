// EliminateObjective.h - Kill specific enemy target

#pragma once

#include "CoreMinimal.h"
#include "Team/Objective.h"
#include "EliminateObjective.generated.h"

/**
 * Objective: Eliminate a specific enemy target
 * Success: Target is killed
 * Failure: Target escapes, timeout
 */
UCLASS(Blueprintable)
class GAMEAI_PROJECT_API UEliminateObjective : public UObjective
{
    GENERATED_BODY()

public:
    UEliminateObjective();

    // Tracking for partial rewards
    UPROPERTY(BlueprintReadOnly, Category = "Eliminate")
    float InitialTargetHealth = 100.0f;

    UPROPERTY(BlueprintReadOnly, Category = "Eliminate")
    float CurrentTargetHealth = 100.0f;

    UPROPERTY(BlueprintReadOnly, Category = "Eliminate")
    float DamageDealt = 0.0f;

    // Override base methods
    virtual void Activate() override;
    virtual bool CheckCompletion() override;
    virtual bool CheckFailure() override;
    virtual void UpdateProgress(float DeltaTime) override;
    virtual float CalculateStrategicReward() const override;

private:
    void CacheTargetHealth();
    float GetTargetCurrentHealth() const;
};
