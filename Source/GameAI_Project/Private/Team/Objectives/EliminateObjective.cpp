// EliminateObjective.cpp - Kill specific enemy target

#include "Team/Objectives/EliminateObjective.h"
#include "Combat/HealthComponent.h"
#include "GameFramework/Actor.h"

UEliminateObjective::UEliminateObjective()
{
    Type = EObjectiveType::Eliminate;
    Priority = 7;  // High priority for eliminations
}

void UEliminateObjective::Activate()
{
    Super::Activate();

    // Cache initial health
    CacheTargetHealth();
}

bool UEliminateObjective::CheckCompletion()
{
    // Target must be dead or invalid
    if (!IsTargetValid())
    {
        return true;
    }

    if (UHealthComponent* HealthComp = TargetActor->FindComponentByClass<UHealthComponent>())
    {
        return HealthComp->IsDead();
    }

    return false;
}

bool UEliminateObjective::CheckFailure()
{
    // Fail if timed out
    if (HasTimedOut())
    {
        return true;
    }

    // Could add: Target escaped (too far away for too long)
    // For now, just timeout

    return false;
}

void UEliminateObjective::UpdateProgress(float DeltaTime)
{
    if (!IsTargetValid())
    {
        Progress = 0.0f;
        return;
    }

    // Update current health
    CurrentTargetHealth = GetTargetCurrentHealth();
    DamageDealt = InitialTargetHealth - CurrentTargetHealth;

    // Progress based on damage dealt
    if (InitialTargetHealth > 0.0f)
    {
        Progress = FMath::Clamp(DamageDealt / InitialTargetHealth, 0.0f, 1.0f);
    }
}

float UEliminateObjective::CalculateStrategicReward() const
{
    if (Status == EObjectiveStatus::Completed)
    {
        // Full reward for kill
        float BaseReward = 50.0f;

        // Bonus for quick elimination
        if (TimeLimit > 0.0f && TimeActive < TimeLimit * 0.5f)
        {
            BaseReward += 15.0f;  // Efficiency bonus
        }

        return BaseReward;
    }

    if (Status == EObjectiveStatus::Failed)
    {
        return -25.0f;
    }

    // Partial credit for damage dealt
    return Progress * 25.0f;
}

void UEliminateObjective::CacheTargetHealth()
{
    if (IsTargetValid())
    {
        if (UHealthComponent* HealthComp = TargetActor->FindComponentByClass<UHealthComponent>())
        {
            InitialTargetHealth = HealthComp->GetCurrentHealth();
            CurrentTargetHealth = InitialTargetHealth;
            DamageDealt = 0.0f;
        }
    }
}

float UEliminateObjective::GetTargetCurrentHealth() const
{
    if (IsTargetValid())
    {
        if (UHealthComponent* HealthComp = TargetActor->FindComponentByClass<UHealthComponent>())
        {
            return HealthComp->GetCurrentHealth();
        }
    }
    return 0.0f;
}
