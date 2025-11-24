// DefendObjective.h - Defend zone/flag objective

#pragma once

#include "CoreMinimal.h"
#include "Team/Objective.h"
#include "DefendObjective.generated.h"

/**
 * Objective: Defend a zone or actor
 * Success: Hold zone for required time, prevent enemies from entering
 * Failure: Zone is breached for too long, timeout, all agents die
 */
UCLASS(Blueprintable)
class GAMEAI_PROJECT_API UDefendObjective : public UObjective
{
    GENERATED_BODY()

public:
    UDefendObjective();

    // Defense parameters
    UPROPERTY(BlueprintReadWrite, Category = "Defend")
    float DefenseRadius = 800.0f;  // Zone to defend

    UPROPERTY(BlueprintReadWrite, Category = "Defend")
    float DefenseTime = 30.0f;  // Time to successfully defend

    UPROPERTY(BlueprintReadWrite, Category = "Defend")
    float MaxBreachTime = 5.0f;  // Max time zone can be breached

    // Tracking
    UPROPERTY(BlueprintReadOnly, Category = "Defend")
    float TimeDefended = 0.0f;

    UPROPERTY(BlueprintReadOnly, Category = "Defend")
    float TimeBreached = 0.0f;

    UPROPERTY(BlueprintReadOnly, Category = "Defend")
    int32 FriendliesInZone = 0;

    UPROPERTY(BlueprintReadOnly, Category = "Defend")
    int32 EnemiesInZone = 0;

    // Override base methods
    virtual bool CheckCompletion() override;
    virtual bool CheckFailure() override;
    virtual void UpdateProgress(float DeltaTime) override;
    virtual float CalculateStrategicReward() const override;

private:
    int32 CountFriendliesInZone() const;
    int32 CountEnemiesInZone() const;
};
