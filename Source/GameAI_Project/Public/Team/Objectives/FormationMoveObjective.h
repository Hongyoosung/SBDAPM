// FormationMoveObjective.h - Coordinated movement objective

#pragma once

#include "CoreMinimal.h"
#include "Team/Objective.h"
#include "FormationMoveObjective.generated.h"

/**
 * Objective: Move to target location while maintaining formation
 * Success: All assigned agents reach destination with good formation
 * Failure: Timeout, formation breaks critically, all agents die
 */
UCLASS(Blueprintable)
class GAMEAI_PROJECT_API UFormationMoveObjective : public UObjective
{
    GENERATED_BODY()

public:
    UFormationMoveObjective();

    // Movement parameters
    UPROPERTY(BlueprintReadWrite, Category = "Formation")
    float ArrivalRadius = 300.0f;  // Distance to be "at destination"

    UPROPERTY(BlueprintReadWrite, Category = "Formation")
    float FormationThreshold = 0.6f;  // Minimum formation coherence

    UPROPERTY(BlueprintReadWrite, Category = "Formation")
    float OptimalFormationDistance = 500.0f;  // Target spacing between agents

    // Tracking
    UPROPERTY(BlueprintReadOnly, Category = "Formation")
    int32 AgentsAtDestination = 0;

    UPROPERTY(BlueprintReadOnly, Category = "Formation")
    float FormationCoherence = 0.0f;

    // Override base methods
    virtual bool CheckCompletion() override;
    virtual bool CheckFailure() override;
    virtual void UpdateProgress(float DeltaTime) override;
    virtual float CalculateStrategicReward() const override;

private:
    int32 CountAgentsAtDestination() const;
    float CalculateFormationCoherence() const;
};
