// Fill out your copyright notice in the Description page of Project Settings.

#pragma once

#include "CoreMinimal.h"
#include "Components/ActorComponent.h"
#include "Core/ObservationElement.h"
#include "StateMachine.generated.h"

class UState;
class AAIController;
class UBlackboardComponent;


UCLASS( ClassGroup=(Custom), meta=(BlueprintSpawnableComponent) )
class GAMEAI_PROJECT_API UStateMachine : public UActorComponent
{
	GENERATED_BODY()


public:
	UStateMachine();


protected:
	// Called when the game starts
	virtual void BeginPlay() override;


public:
	// Called every frame
	virtual void TickComponent(float DeltaTime, ELevelTick TickType, FActorComponentTickFunction* ThisTickFunction) override;

    UFUNCTION(BlueprintCallable, Category = "State Machine")
    void ChangeState(UState* NewState);

	// ========================================
	// OBSERVATION INTERFACE
	// ========================================

	/**
	 * Legacy observation update (backward compatibility)
	 * Prefer using UpdateObservation() with full FObservationElement instead
	 */
	UFUNCTION(BlueprintCallable, Category = "State Machine|Observation", meta = (DeprecatedFunction, DeprecationMessage = "Use UpdateObservation with FObservationElement instead"))
	void GetObservation(float Health, float Distance, int32 Num);

	/**
	 * Update the full observation structure
	 * This is the preferred method for setting observations
	 */
	UFUNCTION(BlueprintCallable, Category = "State Machine|Observation")
	void UpdateObservation(const FObservationElement& NewObservation);

	/**
	 * Get the current observation structure
	 */
	UFUNCTION(BlueprintCallable, BlueprintPure, Category = "State Machine|Observation")
	FObservationElement GetCurrentObservation() const { return CurrentObservation; }

	/**
	 * Update only specific observation fields (for Blueprint convenience)
	 */
	UFUNCTION(BlueprintCallable, Category = "State Machine|Observation")
	void UpdateAgentState(FVector Position, FVector Velocity, FRotator Rotation, float Health, float Stamina, float Shield);

	UFUNCTION(BlueprintCallable, Category = "State Machine|Observation")
	void UpdateCombatState(float WeaponCooldown, int32 Ammunition, int32 WeaponType);

	UFUNCTION(BlueprintCallable, Category = "State Machine|Observation")
	void UpdateEnemyInfo(int32 VisibleCount, const TArray<FEnemyObservation>& NearbyEnemies);

	UFUNCTION(BlueprintCallable, Category = "State Machine|Observation")
	void UpdateTacticalContext(bool bHasCover, float CoverDistance, FVector2D CoverDirection, ETerrainType Terrain);

	void TriggerBlueprintEvent(const FName&);

	APawn* OwnerPawn;

	UFUNCTION(BlueprintCallable, Category = "State Machine")
	UState* GetCurrentState();

	UFUNCTION(BlueprintCallable, Category = "State Machine")
	UState* GetMoveToState();

	UFUNCTION(BlueprintCallable, Category = "State Machine")
	UState* GetAttackState();

	UFUNCTION(BlueprintCallable, Category = "State Machine")
	UState* GetFleeState();

	UFUNCTION(BlueprintCallable, Category = "State Machine")
	UState* GetDeadState();

	// ========================================
	// BLACKBOARD INTEGRATION (Behavior Tree Communication)
	// ========================================

	/**
	 * Get the AI Controller that owns this StateMachine's pawn.
	 * Returns nullptr if the pawn is not controlled by an AI controller.
	 */
	UFUNCTION(BlueprintCallable, BlueprintPure, Category = "State Machine|Blackboard")
	AAIController* GetAIController() const;

	/**
	 * Get the Blackboard component from the AI Controller.
	 * Returns nullptr if there's no AI controller or no Blackboard.
	 */
	UFUNCTION(BlueprintCallable, BlueprintPure, Category = "State Machine|Blackboard")
	UBlackboardComponent* GetBlackboard() const;

	/**
	 * Set the current strategy on the Blackboard.
	 * This is used by FSM states to communicate with the Behavior Tree.
	 *
	 * @param Strategy The strategy name ("MoveTo", "Attack", "Flee", "Dead")
	 */
	UFUNCTION(BlueprintCallable, Category = "State Machine|Blackboard")
	void SetCurrentStrategy(const FString& Strategy);

	/**
	 * Set a target enemy on the Blackboard.
	 * Used by AttackState to specify which enemy to engage.
	 *
	 * @param TargetEnemy The enemy actor to target
	 */
	UFUNCTION(BlueprintCallable, Category = "State Machine|Blackboard")
	void SetTargetEnemy(AActor* TargetEnemy);

	/**
	 * Set a destination vector on the Blackboard.
	 * Used by MoveToState to specify where to move.
	 *
	 * @param Destination The target position
	 */
	UFUNCTION(BlueprintCallable, Category = "State Machine|Blackboard")
	void SetDestination(FVector Destination);

	/**
	 * Set a cover location on the Blackboard.
	 * Used by FleeState to specify where to take cover.
	 *
	 * @param CoverLocation The cover position
	 */
	UFUNCTION(BlueprintCallable, Category = "State Machine|Blackboard")
	void SetCoverLocation(FVector CoverLocation);

	/**
	 * Set the threat level on the Blackboard.
	 *
	 * @param ThreatLevel Normalized threat value (0.0 = safe, 1.0 = extreme danger)
	 */
	UFUNCTION(BlueprintCallable, Category = "State Machine|Blackboard")
	void SetThreatLevel(float ThreatLevel);

	// ========================================
	// OBSERVATION DATA
	// ========================================

	/**
	 * Current observation of the environment (71 features)
	 */
	UPROPERTY(BlueprintReadOnly, Category = "State Machine|Observation")
	FObservationElement CurrentObservation;

	// Legacy fields for backward compatibility with existing states
	// These are automatically synced with CurrentObservation
	float DistanceToDestination;
	float AgentHealth;
	int32 EnemiesNum;
	
private:
	UPROPERTY()
	UState* CurrentState;

	UPROPERTY(Transient)
	UState* MoveToState;

	UPROPERTY(Transient)
	UState* AttackState;

	UPROPERTY(Transient)
	UState* FleeState;

	UPROPERTY(Transient)
	UState* DeadState;

	FTimerHandle TimerHandle;
	AActor* Owner;

	float CurrentTime;
	float LastStateUpdateTime;

	void InitStateMachine();
};
