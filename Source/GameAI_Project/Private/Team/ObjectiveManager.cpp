// ObjectiveManager.cpp - Manages strategic objectives for team

#include "Team/ObjectiveManager.h"
#include "Team/Objectives/EliminateObjective.h"
#include "Team/Objectives/CaptureObjective.h"
#include "Team/Objectives/DefendObjective.h"
#include "Team/Objectives/SupportAllyObjective.h"

UObjectiveManager::UObjectiveManager()
{
    PrimaryComponentTick.bCanEverTick = true;
    PrimaryComponentTick.TickInterval = 0.1f;  // Tick objectives every 0.1s
}

void UObjectiveManager::BeginPlay()
{
    Super::BeginPlay();
}

void UObjectiveManager::TickComponent(float DeltaTime, ELevelTick TickType, FActorComponentTickFunction* ThisTickFunction)
{
    Super::TickComponent(DeltaTime, TickType, ThisTickFunction);

    TickObjectives(DeltaTime);
}

UObjective* UObjectiveManager::CreateObjective(EObjectiveType Type, AActor* Target, const FVector& Location, int32 Priority)
{
    UObjective* NewObjective = nullptr;

    switch (Type)
    {
        case EObjectiveType::Eliminate:
            NewObjective = CreateObjectiveOfType<UEliminateObjective>(Type);
            break;
        case EObjectiveType::CaptureObjective:
            NewObjective = CreateObjectiveOfType<UCaptureObjective>(Type);
            break;
        case EObjectiveType::DefendObjective:
            NewObjective = CreateObjectiveOfType<UDefendObjective>(Type);
            break;
        case EObjectiveType::SupportAlly:
            NewObjective = CreateObjectiveOfType<USupportAllyObjective>(Type);
            break;
        default:
            UE_LOG(LogTemp, Warning, TEXT("ObjectiveManager: Unsupported objective type"));
            return nullptr;
    }

    if (NewObjective)
    {
        NewObjective->TargetActor = Target;
        NewObjective->TargetLocation = Location;
        NewObjective->Priority = Priority;
        Objectives.Add(NewObjective);
    }

    return NewObjective;
}

UObjective* UObjectiveManager::CreateEliminateObjective(AActor* Target, int32 Priority)
{
    return CreateObjective(EObjectiveType::Eliminate, Target, FVector::ZeroVector, Priority);
}

UObjective* UObjectiveManager::CreateCaptureObjective(const FVector& Location, int32 Priority)
{
    return CreateObjective(EObjectiveType::CaptureObjective, nullptr, Location, Priority);
}

UObjective* UObjectiveManager::CreateDefendObjective(const FVector& Location, int32 Priority)
{
    return CreateObjective(EObjectiveType::DefendObjective, nullptr, Location, Priority);
}

UObjective* UObjectiveManager::CreateSupportObjective(AActor* AllyTarget, int32 Priority)
{
    return CreateObjective(EObjectiveType::SupportAlly, AllyTarget, FVector::ZeroVector, Priority);
}

void UObjectiveManager::AssignAgentsToObjective(UObjective* Objective, const TArray<AActor*>& Agents)
{
    if (!Objective)
    {
        return;
    }

    // Remove agents from previous objectives
    for (AActor* Agent : Agents)
    {
        UnassignAgentFromObjective(Agent);
    }

    // Assign to new objective
    Objective->AssignedAgents = Agents;

    // Update mapping
    for (AActor* Agent : Agents)
    {
        if (Agent)
        {
            AgentObjectiveMap.Add(Agent, Objective);
        }
    }
}

void UObjectiveManager::UnassignAgentFromObjective(AActor* Agent)
{
    if (!Agent)
    {
        return;
    }

    // Find agent's current objective
    if (TObjectPtr<UObjective>* ObjectivePtr = AgentObjectiveMap.Find(Agent))
    {
        UObjective* CurrentObjective = ObjectivePtr->Get();
        if (CurrentObjective)
        {
            CurrentObjective->AssignedAgents.Remove(Agent);
        }
    }

    AgentObjectiveMap.Remove(Agent);
}

void UObjectiveManager::ClearAllAssignments()
{
    for (UObjective* Objective : Objectives)
    {
        if (Objective)
        {
            Objective->AssignedAgents.Empty();
        }
    }
    AgentObjectiveMap.Empty();
}

void UObjectiveManager::ActivateObjective(UObjective* Objective)
{
    if (Objective)
    {
        Objective->Activate();
    }
}

void UObjectiveManager::DeactivateObjective(UObjective* Objective)
{
    if (Objective)
    {
        Objective->Deactivate();
    }
}

void UObjectiveManager::CancelObjective(UObjective* Objective)
{
    if (Objective)
    {
        Objective->Cancel();

        // Unassign all agents
        for (AActor* Agent : Objective->AssignedAgents)
        {
            AgentObjectiveMap.Remove(Agent);
        }
        Objective->AssignedAgents.Empty();
    }
}

void UObjectiveManager::RemoveObjective(UObjective* Objective)
{
    if (Objective)
    {
        // Unassign agents
        CancelObjective(Objective);

        // Remove from list
        Objectives.Remove(Objective);
    }
}

void UObjectiveManager::ClearCompletedObjectives()
{
    TArray<UObjective*> ToRemove;

    for (UObjective* Objective : Objectives)
    {
        if (Objective && (Objective->IsCompleted() || Objective->IsFailed()))
        {
            ToRemove.Add(Objective);
        }
    }

    for (UObjective* Objective : ToRemove)
    {
        RemoveObjective(Objective);
    }
}

TArray<UObjective*> UObjectiveManager::GetActiveObjectives() const
{
    TArray<UObjective*> ActiveObjectives;

    for (UObjective* Objective : Objectives)
    {
        if (Objective && Objective->IsActive())
        {
            ActiveObjectives.Add(Objective);
        }
    }

    return ActiveObjectives;
}

TArray<UObjective*> UObjectiveManager::GetAllObjectives() const
{
    return Objectives;
}

UObjective* UObjectiveManager::GetAgentObjective(AActor* Agent) const
{
    if (const TObjectPtr<UObjective>* ObjectivePtr = AgentObjectiveMap.Find(Agent))
    {
        return *ObjectivePtr;
    }
    return nullptr;
}

TArray<AActor*> UObjectiveManager::GetObjectiveAgents(UObjective* Objective) const
{
    if (Objective)
    {
        return Objective->AssignedAgents;
    }
    return TArray<AActor*>();
}

int32 UObjectiveManager::GetActiveObjectiveCount() const
{
    return GetActiveObjectives().Num();
}

UObjective* UObjectiveManager::GetHighestPriorityObjective() const
{
    UObjective* Highest = nullptr;
    int32 HighestPriority = -1;

    for (UObjective* Objective : GetActiveObjectives())
    {
        if (Objective && Objective->Priority > HighestPriority)
        {
            Highest = Objective;
            HighestPriority = Objective->Priority;
        }
    }

    return Highest;
}

float UObjectiveManager::CalculateTotalTeamReward() const
{
    float TotalReward = 0.0f;

    for (UObjective* Objective : Objectives)
    {
        if (Objective)
        {
            TotalReward += Objective->CalculateStrategicReward();
        }
    }

    return TotalReward;
}

template<typename T>
T* UObjectiveManager::CreateObjectiveOfType(EObjectiveType Type)
{
    T* NewObjective = NewObject<T>(this);
    if (NewObjective)
    {
        NewObjective->Type = Type;
    }
    return NewObjective;
}

void UObjectiveManager::TickObjectives(float DeltaTime)
{
    for (UObjective* Objective : Objectives)
    {
        if (Objective && Objective->IsActive())
        {
            Objective->Tick(DeltaTime);
        }
    }

    // Periodically clean up completed objectives
    static float CleanupTimer = 0.0f;
    CleanupTimer += DeltaTime;
    if (CleanupTimer >= 5.0f)
    {
        CleanupObjectives();
        CleanupTimer = 0.0f;
    }
}

void UObjectiveManager::CleanupObjectives()
{
    // Remove objectives that are completed/failed for more than 10 seconds
    TArray<UObjective*> ToRemove;

    for (UObjective* Objective : Objectives)
    {
        if (Objective && !Objective->IsActive())
        {
            // Could add time tracking for cleanup delay
            // For now, just keep them for reward calculation
        }
    }
}
