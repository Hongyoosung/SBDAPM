// SupportAllyObjective.cpp - Support ally teammate

#include "Team/Objectives/SupportAllyObjective.h"
#include "GameFramework/Actor.h"
#include "Combat/HealthComponent.h"

USupportAllyObjective::USupportAllyObjective()
{
    Type = EObjectiveType::SupportAlly;
    Priority = 6;
    SupportRange = 1000.0f;
    IdealDistance = 500.0f;
    TimeSupportingAlly = 0.0f;
    ThreatsNeutralized = 0;
    AllyHealthPercent = 1.0f;
}

void USupportAllyObjective::Activate()
{
    Super::Activate();

    // Cache initial ally health
    AllyHealthPercent = GetAllyHealthPercent();
}

bool USupportAllyObjective::CheckCompletion()
{
    // For now, consider completed after supporting for certain time
    // Could be: ally completed their objective, threats eliminated, etc.
    return TimeSupportingAlly >= 30.0f && IsAllyAlive();
}

bool USupportAllyObjective::CheckFailure()
{
    // Fail if timed out
    if (HasTimedOut())
    {
        return true;
    }

    // Fail if ally dies
    if (!IsAllyAlive())
    {
        return true;
    }

    // Fail if all support agents are dead
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

void USupportAllyObjective::UpdateProgress(float DeltaTime)
{
    // Update ally health
    AllyHealthPercent = GetAllyHealthPercent();

    // Count time providing support
    if (IsInSupportRange() && IsAllyAlive())
    {
        TimeSupportingAlly += DeltaTime;
    }

    // Progress based on support time and ally survival
    float TimeProgress = FMath::Clamp(TimeSupportingAlly / 30.0f, 0.0f, 1.0f);
    float SurvivalProgress = AllyHealthPercent;

    Progress = (TimeProgress * 0.6f) + (SurvivalProgress * 0.4f);
}

float USupportAllyObjective::CalculateStrategicReward() const
{
    if (Status == EObjectiveStatus::Completed)
    {
        // Reward for successful support
        float BaseReward = 60.0f;

        // Bonus if ally stayed healthy
        if (AllyHealthPercent > 0.8f)
        {
            BaseReward += 20.0f;
        }

        // Bonus for threats neutralized
        BaseReward += ThreatsNeutralized * 10.0f;

        return BaseReward;
    }

    if (Status == EObjectiveStatus::Failed)
    {
        // Penalty if ally died
        return -30.0f;
    }

    // Partial credit for support time
    return Progress * 30.0f;
}

bool USupportAllyObjective::IsAllyAlive() const
{
    if (!IsTargetValid())
    {
        return false;
    }

    if (UHealthComponent* HealthComp = TargetActor->FindComponentByClass<UHealthComponent>())
    {
        return !HealthComp->IsDead();
    }

    return true;
}

float USupportAllyObjective::GetAllyHealthPercent() const
{
    if (!IsTargetValid())
    {
        return 0.0f;
    }

    if (UHealthComponent* HealthComp = TargetActor->FindComponentByClass<UHealthComponent>())
    {
        return HealthComp->GetHealthPercentage();
    }

    return 1.0f;
}

bool USupportAllyObjective::IsInSupportRange() const
{
    if (!IsTargetValid() || AssignedAgents.Num() == 0)
    {
        return false;
    }

    // Check if at least one assigned agent is in support range
    for (AActor* Agent : AssignedAgents)
    {
        if (Agent && IsValid(Agent))
        {
            float Distance = FVector::Distance(Agent->GetActorLocation(), TargetActor->GetActorLocation());
            if (Distance <= SupportRange)
            {
                return true;
            }
        }
    }

    return false;
}
