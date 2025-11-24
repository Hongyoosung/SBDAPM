// Objective.h - Base class for strategic objectives
// Part of Combat System Refactoring v3.0

#pragma once

#include "CoreMinimal.h"
#include "UObject/NoExportTypes.h"
#include "Objective.generated.h"

/**
 * Objective types for strategic planning
 */
UENUM(BlueprintType)
enum class EObjectiveType : uint8
{
    Eliminate        UMETA(DisplayName = "Eliminate Target"),      // Kill specific enemy
    CaptureObjective UMETA(DisplayName = "Capture Objective"),     // Capture zone/flag
    DefendObjective  UMETA(DisplayName = "Defend Objective"),      // Hold zone/flag
    SupportAlly      UMETA(DisplayName = "Support Ally"),          // Provide covering fire
    FormationMove    UMETA(DisplayName = "Formation Move"),        // Coordinated movement
    Retreat          UMETA(DisplayName = "Retreat"),               // Fall back
    RescueAlly       UMETA(DisplayName = "Rescue Ally")            // Rescue wounded teammate
};

/**
 * Objective status for tracking
 */
UENUM(BlueprintType)
enum class EObjectiveStatus : uint8
{
    Inactive    UMETA(DisplayName = "Inactive"),    // Not started
    Active      UMETA(DisplayName = "Active"),      // Currently executing
    Completed   UMETA(DisplayName = "Completed"),   // Successfully finished
    Failed      UMETA(DisplayName = "Failed"),      // Failed to complete
    Cancelled   UMETA(DisplayName = "Cancelled")    // Cancelled by leader
};

/**
 * Base class for strategic objectives
 * Objectives define WHAT agents should do (strategic layer)
 * Tactical layer (RL policy) determines HOW to execute
 */
UCLASS(Abstract, Blueprintable)
class GAMEAI_PROJECT_API UObjective : public UObject
{
    GENERATED_BODY()

public:
    UObjective();

    // Core properties
    UPROPERTY(BlueprintReadWrite, Category = "Objective")
    EObjectiveType Type;

    UPROPERTY(BlueprintReadWrite, Category = "Objective")
    TObjectPtr<AActor> TargetActor = nullptr;

    UPROPERTY(BlueprintReadWrite, Category = "Objective")
    FVector TargetLocation = FVector::ZeroVector;

    UPROPERTY(BlueprintReadWrite, Category = "Objective")
    int32 Priority = 5;  // 0-10, higher = more important

    UPROPERTY(BlueprintReadWrite, Category = "Objective")
    float TimeLimit = 0.0f;  // 0 = no limit (seconds)

    UPROPERTY(BlueprintReadWrite, Category = "Objective")
    TArray<TObjectPtr<AActor>> AssignedAgents;

    // State tracking
    UPROPERTY(BlueprintReadOnly, Category = "Objective")
    EObjectiveStatus Status = EObjectiveStatus::Inactive;

    UPROPERTY(BlueprintReadOnly, Category = "Objective")
    float Progress = 0.0f;  // 0.0-1.0

    UPROPERTY(BlueprintReadOnly, Category = "Objective")
    float TimeActive = 0.0f;  // Time since activation

    UPROPERTY(BlueprintReadOnly, Category = "Objective")
    float TimeRemaining = 0.0f;  // Time until timeout

    // Lifecycle methods
    UFUNCTION(BlueprintCallable, Category = "Objective")
    virtual void Activate();

    UFUNCTION(BlueprintCallable, Category = "Objective")
    virtual void Deactivate();

    UFUNCTION(BlueprintCallable, Category = "Objective")
    virtual void Cancel();

    UFUNCTION(BlueprintCallable, Category = "Objective")
    virtual void Tick(float DeltaTime);

    // Status queries
    UFUNCTION(BlueprintPure, Category = "Objective")
    virtual bool IsActive() const { return Status == EObjectiveStatus::Active; }

    UFUNCTION(BlueprintPure, Category = "Objective")
    virtual bool IsCompleted() const { return Status == EObjectiveStatus::Completed; }

    UFUNCTION(BlueprintPure, Category = "Objective")
    virtual bool IsFailed() const { return Status == EObjectiveStatus::Failed; }

    UFUNCTION(BlueprintPure, Category = "Objective")
    virtual float GetProgress() const { return Progress; }

    // Reward calculation for MCTS/RL
    UFUNCTION(BlueprintPure, Category = "Objective")
    virtual float CalculateStrategicReward() const;

    // Override these in subclasses
    virtual bool CheckCompletion();
    virtual bool CheckFailure();
    virtual void UpdateProgress(float DeltaTime);

protected:
    // Helper to check if target actor is still valid
    bool IsTargetValid() const;

    // Helper to check if objective timed out
    bool HasTimedOut() const;
};
