// CaptureObjective.cpp - Capture zone/flag objective

#include "Team/Objectives/CaptureObjective.h"
#include "GameFramework/Actor.h"
#include "Combat/HealthComponent.h"
#include "Core/SimulationManagerGameMode.h"

UCaptureObjective::UCaptureObjective()
{
    Type = EObjectiveType::CaptureObjective;
    Priority = 8;  // High priority
    CaptureRadius = 500.0f;
    CaptureTime = 10.0f;
    MinAgentsRequired = 1;
    TimeInZone = 0.0f;
    AgentsInZone = 0;
    bIsContested = false;
}

bool UCaptureObjective::CheckCompletion()
{
    // Complete when agents have held zone for required time
    return TimeInZone >= CaptureTime && !bIsContested;
}

bool UCaptureObjective::CheckFailure()
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

void UCaptureObjective::UpdateProgress(float DeltaTime)
{
    // Count agents currently in zone
    AgentsInZone = CountAgentsInZone();
    bIsContested = CheckIfContested();

    // Update capture timer
    if (AgentsInZone >= MinAgentsRequired && !bIsContested)
    {
        TimeInZone += DeltaTime;
    }
    else
    {
        // Lose progress if not enough agents or contested
        TimeInZone = FMath::Max(0.0f, TimeInZone - DeltaTime * 0.5f);
    }

    // Progress based on capture time
    if (CaptureTime > 0.0f)
    {
        Progress = FMath::Clamp(TimeInZone / CaptureTime, 0.0f, 1.0f);
    }
}

float UCaptureObjective::CalculateStrategicReward() const
{
    if (Status == EObjectiveStatus::Completed)
    {
        // Major reward for capturing objective
        float BaseReward = 100.0f;

        // Bonus for fast capture
        if (TimeActive < CaptureTime * 1.5f)
        {
            BaseReward += 25.0f;
        }

        return BaseReward;
    }

    if (Status == EObjectiveStatus::Failed)
    {
        return -50.0f;
    }

    // Partial credit for time in zone
    return Progress * 50.0f;
}

int32 UCaptureObjective::CountAgentsInZone() const
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

            // Check distance to zone
            float Distance = FVector::Distance(Agent->GetActorLocation(), TargetLocation);
            if (Distance <= CaptureRadius)
            {
                Count++;
            }
        }
    }

    return Count;
}

bool UCaptureObjective::CheckIfContested() const
{
    // Get the SimulationManager to query enemy teams
    UWorld* World = GetWorld();
    if (!World) return false;

    ASimulationManagerGameMode* GameMode = World->GetAuthGameMode<ASimulationManagerGameMode>();
    if (!GameMode) return false;

    // Find team ID of our agents (assume all assigned agents are on same team)
    int32 OurTeamID = -1;
    if (AssignedAgents.Num() > 0 && AssignedAgents[0])
    {
        OurTeamID = GameMode->GetTeamIDForActor(AssignedAgents[0]);
    }

    if (OurTeamID == -1) return false;

    // Get all enemy actors for our team
    TArray<AActor*> EnemyActors = GameMode->GetEnemyActors(OurTeamID);

    // Check if any enemy is within capture radius
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

            // Check distance to capture zone
            float Distance = FVector::Distance(Enemy->GetActorLocation(), TargetLocation);
            if (Distance <= CaptureRadius)
            {
                return true;  // Zone is contested
            }
        }
    }

    return false;  // No enemies in zone
}
