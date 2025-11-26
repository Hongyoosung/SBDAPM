// DefendObjective.cpp - Defend zone/flag objective

#include "Team/Objectives/DefendObjective.h"
#include "GameFramework/Actor.h"
#include "Combat/HealthComponent.h"
#include "Core/SimulationManagerGameMode.h"

UDefendObjective::UDefendObjective()
{
    Type = EObjectiveType::DefendObjective;
    Priority = 7;
    DefenseRadius = 800.0f;
    DefenseTime = 30.0f;
    MaxBreachTime = 5.0f;
    TimeDefended = 0.0f;
    TimeBreached = 0.0f;
    FriendliesInZone = 0;
    EnemiesInZone = 0;
}

bool UDefendObjective::CheckCompletion()
{
    // Complete when defended for required time
    return TimeDefended >= DefenseTime;
}

bool UDefendObjective::CheckFailure()
{
    // Fail if timed out
    if (HasTimedOut())
    {
        return true;
    }

    // Fail if zone breached for too long
    if (TimeBreached >= MaxBreachTime)
    {
        return true;
    }

    // Fail if all agents are dead
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

void UDefendObjective::UpdateProgress(float DeltaTime)
{
    // Count friendlies and enemies in zone
    FriendliesInZone = CountFriendliesInZone();
    EnemiesInZone = CountEnemiesInZone();

    // Update timers based on zone status
    if (EnemiesInZone == 0)
    {
        // Zone is secure
        TimeDefended += DeltaTime;
        TimeBreached = 0.0f;  // Reset breach timer
    }
    else
    {
        // Zone is breached
        TimeBreached += DeltaTime;
    }

    // Progress based on defense time
    if (DefenseTime > 0.0f)
    {
        Progress = FMath::Clamp(TimeDefended / DefenseTime, 0.0f, 1.0f);
    }

    // Penalty for breach time
    Progress -= (TimeBreached / MaxBreachTime) * 0.2f;
    Progress = FMath::Clamp(Progress, 0.0f, 1.0f);
}

float UDefendObjective::CalculateStrategicReward() const
{
    if (Status == EObjectiveStatus::Completed)
    {
        // Major reward for successful defense
        float BaseReward = 80.0f;

        // Bonus for perfect defense (no breaches)
        if (TimeBreached <= 0.1f)
        {
            BaseReward += 30.0f;
        }

        return BaseReward;
    }

    if (Status == EObjectiveStatus::Failed)
    {
        return -40.0f;
    }

    // Partial credit for time defended
    return Progress * 40.0f;
}

int32 UDefendObjective::CountFriendliesInZone() const
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
            if (Distance <= DefenseRadius)
            {
                Count++;
            }
        }
    }

    return Count;
}

int32 UDefendObjective::CountEnemiesInZone() const
{
    int32 Count = 0;

    // Get the SimulationManager to query enemy teams
    UWorld* World = GetWorld();
    if (!World) return 0;

    ASimulationManagerGameMode* GameMode = World->GetAuthGameMode<ASimulationManagerGameMode>();
    if (!GameMode) return 0;

    // Find team ID of our agents (assume all assigned agents are on same team)
    int32 OurTeamID = -1;
    if (AssignedAgents.Num() > 0 && AssignedAgents[0])
    {
        OurTeamID = GameMode->GetTeamIDForActor(AssignedAgents[0]);
    }

    if (OurTeamID == -1) return 0;

    // Get all enemy actors for our team
    TArray<AActor*> EnemyActors = GameMode->GetEnemyActors(OurTeamID);

    // Count enemies within defense radius
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

            // Check distance to defense zone
            float Distance = FVector::Distance(Enemy->GetActorLocation(), TargetLocation);
            if (Distance <= DefenseRadius)
            {
                Count++;
            }
        }
    }

    return Count;
}
