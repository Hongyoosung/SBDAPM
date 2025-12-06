// FormationMoveObjective.cpp - Coordinated movement objective

#include "Team/Objectives/FormationMoveObjective.h"
#include "GameFramework/Actor.h"
#include "Combat/HealthComponent.h"

UFormationMoveObjective::UFormationMoveObjective()
{
    Type = EObjectiveType::FormationMove;
    Priority = 5;
    ArrivalRadius = 300.0f;
    FormationThreshold = 0.6f;
    OptimalFormationDistance = 500.0f;
    AgentsAtDestination = 0;
    FormationCoherence = 0.0f;
}

bool UFormationMoveObjective::CheckCompletion()
{
    // SAFEGUARD: Prevent completion if no agents assigned (dummy objective edge case)
    if (AssignedAgents.Num() == 0)
    {
        return false;
    }

    // Complete when all agents reach destination with good formation
    return AgentsAtDestination >= AssignedAgents.Num() &&
           FormationCoherence >= FormationThreshold;
}

bool UFormationMoveObjective::CheckFailure()
{
    // Fail if timed out
    if (HasTimedOut())
    {
        return true;
    }

    // Fail if all assigned agents are dead
    int32 AliveAgents = 0;
    for (AActor* Agent : AssignedAgents)
    {
        if (Agent && IsValid(Agent))
        {
            if (UHealthComponent* HealthComp = Agent->FindComponentByClass<UHealthComponent>())
            {
                if (!HealthComp->IsDead())
                {
                    AliveAgents++;
                }
            }
        }
    }

    return AliveAgents == 0;
}

void UFormationMoveObjective::UpdateProgress(float DeltaTime)
{
    // Count agents at destination
    AgentsAtDestination = CountAgentsAtDestination();
    FormationCoherence = CalculateFormationCoherence();

    // Progress based on agents at destination
    int32 TotalAgents = AssignedAgents.Num();
    if (TotalAgents > 0)
    {
        float ArrivalProgress = static_cast<float>(AgentsAtDestination) / TotalAgents;
        float FormationProgress = FormationCoherence;
        Progress = FMath::Clamp((ArrivalProgress + FormationProgress) / 2.0f, 0.0f, 1.0f);
    }
}

float UFormationMoveObjective::CalculateStrategicReward() const
{
    if (Status == EObjectiveStatus::Completed)
    {
        // Reward for successful coordinated movement
        float BaseReward = 30.0f;

        // Bonus for maintaining excellent formation
        if (FormationCoherence >= 0.8f)
        {
            BaseReward += 15.0f;
        }

        return BaseReward;
    }

    if (Status == EObjectiveStatus::Failed)
    {
        return -20.0f;
    }

    // Partial credit for progress
    return Progress * 15.0f;
}

int32 UFormationMoveObjective::CountAgentsAtDestination() const
{
    int32 Count = 0;

    for (AActor* Agent : AssignedAgents)
    {
        if (Agent && IsValid(Agent))
        {
            // Check if agent is alive
            if (UHealthComponent* HealthComp = Agent->FindComponentByClass<UHealthComponent>())
            {
                if (HealthComp->IsDead())
                {
                    continue;
                }
            }

            // Check distance to destination
            float Distance = FVector::Distance(Agent->GetActorLocation(), TargetLocation);
            if (Distance <= ArrivalRadius)
            {
                Count++;
            }
        }
    }

    return Count;
}

float UFormationMoveObjective::CalculateFormationCoherence() const
{
    TArray<AActor*> AliveAgents;

    // Collect alive agents
    for (AActor* Agent : AssignedAgents)
    {
        if (Agent && IsValid(Agent))
        {
            if (UHealthComponent* HealthComp = Agent->FindComponentByClass<UHealthComponent>())
            {
                if (!HealthComp->IsDead())
                {
                    AliveAgents.Add(Agent);
                }
            }
        }
    }

    if (AliveAgents.Num() < 2)
    {
        return 1.0f;  // Single agent is always "in formation"
    }

    // Calculate average spacing between agents
    float TotalScore = 0.0f;
    int32 PairCount = 0;

    for (int32 i = 0; i < AliveAgents.Num(); ++i)
    {
        for (int32 j = i + 1; j < AliveAgents.Num(); ++j)
        {
            float Distance = FVector::Distance(
                AliveAgents[i]->GetActorLocation(),
                AliveAgents[j]->GetActorLocation()
            );

            // Score based on how close to optimal distance
            float DistanceRatio = Distance / OptimalFormationDistance;
            float Score = FMath::Exp(-FMath::Square(DistanceRatio - 1.0f));  // Gaussian scoring
            TotalScore += Score;
            PairCount++;
        }
    }

    return PairCount > 0 ? TotalScore / PairCount : 0.0f;
}
