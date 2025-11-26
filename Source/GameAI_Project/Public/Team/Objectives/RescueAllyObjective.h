// RescueAllyObjective.h - Rescue wounded teammate

#pragma once

#include "CoreMinimal.h"
#include "Team/Objective.h"
#include "RescueAllyObjective.generated.h"

/**
 * Objective: Rescue a wounded teammate
 * Success: Reach wounded ally and provide support
 * Failure: Timeout, ally dies, all rescuers die
 */
UCLASS(Blueprintable)
class GAMEAI_PROJECT_API URescueAllyObjective : public UObjective
{
    GENERATED_BODY()

public:
    URescueAllyObjective();

    // Rescue parameters
    UPROPERTY(BlueprintReadWrite, Category = "Rescue")
    float RescueRadius = 300.0f;  // Distance to be "in rescue range"

    UPROPERTY(BlueprintReadWrite, Category = "Rescue")
    float RescueTime = 5.0f;  // Time to complete rescue

    UPROPERTY(BlueprintReadWrite, Category = "Rescue")
    int32 MinRescuers = 1;  // Minimum agents needed to rescue

    // Tracking
    UPROPERTY(BlueprintReadOnly, Category = "Rescue")
    float TimeRescuing = 0.0f;

    UPROPERTY(BlueprintReadOnly, Category = "Rescue")
    int32 RescuersInRange = 0;

    UPROPERTY(BlueprintReadOnly, Category = "Rescue")
    bool bAllyStillAlive = true;

    // Override base methods
    virtual bool CheckCompletion() override;
    virtual bool CheckFailure() override;
    virtual void UpdateProgress(float DeltaTime) override;
    virtual float CalculateStrategicReward() const override;

private:
    int32 CountRescuersInRange() const;
    bool IsAllyAlive() const;
};
