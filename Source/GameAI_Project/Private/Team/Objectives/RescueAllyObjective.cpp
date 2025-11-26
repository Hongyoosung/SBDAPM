// RescueAllyObjective.cpp - Rescue wounded teammate

#include "Team/Objectives/RescueAllyObjective.h"
#include "GameFramework/Actor.h"
#include "Combat/HealthComponent.h"

URescueAllyObjective::URescueAllyObjective()
{
    Type = EObjectiveType::RescueAlly;
    Priority = 7;  // High priority
    RescueRadius = 300.0f;
    RescueTime = 5.0f;
    MinRescuers = 1;
    TimeRescuing = 0.0f;
    RescuersInRange = 0;
    bAllyStillAlive = true;
}

bool URescueAllyObjective::CheckCompletion()
{
    // Complete when rescue time is met
    return TimeRescuing >= RescueTime && bAllyStillAlive;
}

bool URescueAllyObjective::CheckFailure()
{
    // Fail if ally dies
    if (!bAllyStillAlive)
    {
        return true;
    }

    // Fail if timed out
    if (HasTimedOut())
    {
        return true;
    }

    // Fail if all rescuers are dead
    int32 AliveRescuers = 0;
    for (AActor* Agent : AssignedAgents)
    {
        if (Agent && IsValid(Agent))
        {
            if (UHealthComponent* HealthComp = Agent->FindComponentByClass<UHealthComponent>())
            {
                if (!HealthComp->IsDead())
                {
                    AliveRescuers++;
                }
            }
        }
    }

    return AliveRescuers == 0;
}

void URescueAllyObjective::UpdateProgress(float DeltaTime)
{
    // Update ally status
    bAllyStillAlive = IsAllyAlive();

    if (!bAllyStillAlive)
    {
        return;  // Can't rescue a dead ally
    }

    // Count rescuers in range
    RescuersInRange = CountRescuersInRange();

    // Update rescue timer
    if (RescuersInRange >= MinRescuers)
    {
        TimeRescuing += DeltaTime;
    }
    else
    {
        // Lose progress if not enough rescuers
        TimeRescuing = FMath::Max(0.0f, TimeRescuing - DeltaTime * 0.5f);
    }

    // Progress based on rescue time
    if (RescueTime > 0.0f)
    {
        Progress = FMath::Clamp(TimeRescuing / RescueTime, 0.0f, 1.0f);
    }
}

float URescueAllyObjective::CalculateStrategicReward() const
{
    if (Status == EObjectiveStatus::Completed)
    {
        // Major reward for saving a teammate
        float BaseReward = 80.0f;

        // Bonus for fast rescue
        if (TimeActive < RescueTime * 1.5f)
        {
            BaseReward += 20.0f;
        }

        return BaseReward;
    }

    if (Status == EObjectiveStatus::Failed)
    {
        return -40.0f;
    }

    // Partial credit for time rescuing
    return Progress * 40.0f;
}

int32 URescueAllyObjective::CountRescuersInRange() const
{
    int32 Count = 0;

    if (!TargetActor || !IsValid(TargetActor))
    {
        return 0;
    }

    FVector AllyLocation = TargetActor->GetActorLocation();

    for (AActor* Agent : AssignedAgents)
    {
        if (Agent && IsValid(Agent))
        {
            // Check if rescuer is alive
            if (UHealthComponent* HealthComp = Agent->FindComponentByClass<UHealthComponent>())
            {
                if (HealthComp->IsDead())
                {
                    continue;
                }
            }

            // Check distance to ally
            float Distance = FVector::Distance(Agent->GetActorLocation(), AllyLocation);
            if (Distance <= RescueRadius)
            {
                Count++;
            }
        }
    }

    return Count;
}

bool URescueAllyObjective::IsAllyAlive() const
{
    if (!TargetActor || !IsValid(TargetActor))
    {
        return false;
    }

    if (UHealthComponent* HealthComp = TargetActor->FindComponentByClass<UHealthComponent>())
    {
        return !HealthComp->IsDead();
    }

    return true;  // Assume alive if no health component
}
