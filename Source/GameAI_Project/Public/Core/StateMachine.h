// Fill out your copyright notice in the Description page of Project Settings.

#pragma once

#include "CoreMinimal.h"
#include "Components/ActorComponent.h"
#include "StateMachine.generated.h"


class UState;


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

    UFUNCTION(BlueprintCallable)
    void ChangeState(UState* NewState);

	UFUNCTION(BlueprintCallable)
	void GetObservation(float Health, float Distance, int32 Num);

	void TriggerBlueprintEvent(const FName&);

	APawn* OwnerPawn;

	UFUNCTION(BlueprintCallable)
	UState* GetCurrentState();

	UFUNCTION(BlueprintCallable)
	UState* GetMoveToState();

	UFUNCTION(BlueprintCallable)
	UState* GetAttackState();

	UFUNCTION(BlueprintCallable)
	UState* GetFleeState();

	UFUNCTION(BlueprintCallable)
	UState* GetDeadState();

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
