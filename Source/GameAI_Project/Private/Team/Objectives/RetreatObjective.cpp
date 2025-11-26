// RetreatObjective.cpp - Retreat to safe location

#include "Team/Objectives/RetreatObjective.h"
#include "GameFramework/Actor.h"
#include "Combat/HealthComponent.h"
#include "Core/SimulationManagerGameMode.h"

URetreatObjective::URetreatObjective()
{
    Type = EObjectiveType::Retreat;
    Priority = 6;  // Medium-high priority
    SafeRadius = 400.0f;
    MinDistanceFromEnemy = 2000.0f;
    AgentsAtSafeZone = 0;
    bIsSafe = false;
}

bool URetreatObjective::CheckCompletion()
{
    // Complete when all agents reach safe zone and are safe from enemies
    return AgentsAtSafeZone >= AssignedAgents.Num() && bIsSafe;
}

bool URetreatObjective::CheckFailure()
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

void URetreatObjective::UpdateProgress(float DeltaTime)
{
    // Count agents at safe zone
    AgentsAtSafeZone = CountAgentsAtSafeZone();
    bIsSafe = CheckIfSafe();

    // Progress based on agents reaching safe zone
    int32 TotalAgents = AssignedAgents.Num();
    if (TotalAgents > 0)
    {
        float ArrivalProgress = static_cast<float>(AgentsAtSafeZone) / TotalAgents;
        float SafetyBonus = bIsSafe ? 0.2f : 0.0f;
        Progress = FMath::Clamp(ArrivalProgress + SafetyBonus, 0.0f, 1.0f);
    }
}

float URetreatObjective::CalculateStrategicReward() const
{
    if (Status == EObjectiveStatus::Completed)
    {
        // Reward for successful retreat
        float BaseReward = 40.0f;

        // Bonus for fast retreat
        if (TimeActive < 10.0f)
        {
            BaseReward += 10.0f;
        }

        return BaseReward;
    }

    if (Status == EObjectiveStatus::Failed)
    {
        return -30.0f;
    }

    // Partial credit for agents reaching safe zone
    return Progress * 20.0f;
}

int32 URetreatObjective::CountAgentsAtSafeZone() const
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

            // Check distance to safe zone
            float Distance = FVector::Distance(Agent->GetActorLocation(), TargetLocation);
            if (Distance <= SafeRadius)
            {
                Count++;
            }
        }
    }

    return Count;
}

bool URetreatObjective::CheckIfSafe() const
{
    // Get the SimulationManager to query enemy teams
    UWorld* World = GetWorld();
    if (!World) return false;

    ASimulationManagerGameMode* GameMode = World->GetAuthGameMode<ASimulationManagerGameMode>();
    if (!GameMode) return false;

    // Find team ID of our agents
    int32 OurTeamID = -1;
    if (AssignedAgents.Num() > 0 && AssignedAgents[0])
    {
        OurTeamID = GameMode->GetTeamIDForActor(AssignedAgents[0]);
    }

    if (OurTeamID == -1) return false;

    // Get all enemy actors
    TArray<AActor*> EnemyActors = GameMode->GetEnemyActors(OurTeamID);

    // Check if any enemy is too close to safe zone
    for (AActor* Enemy : EnemyActors)
    {
        if (Enemy && IsValid(Enemy))
        {
            // Check if enemy is alive
            if (UHealthComponent* HealthComp = Enemy->FindComponentByClass<UHealthComponent>())
            {
                if (HealthComp->IsDead())
                {
                    continue;
                }
            }

            // Check distance to safe zone
            float Distance = FVector::Distance(Enemy->GetActorLocation(), TargetLocation);
            if (Distance < MinDistanceFromEnemy)
            {
                return false;  // Not safe, enemy too close
            }
        }
    }

    return true;  // Safe, no enemies nearby
}
