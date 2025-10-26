// Fill out your copyright notice in the Description page of Project Settings.

#pragma once

#include "CoreMinimal.h"
#include "UObject/NoExportTypes.h"
#include "State.generated.h"

// Forward Declarations
class AAIController;
class UStateMachine;
class UAction;
class UMCTS;

UCLASS(Blueprintable)
class GAMEAI_PROJECT_API UState : public UObject
{
	GENERATED_BODY()
	
public:
	virtual void EnterState(UStateMachine* StateMachine);
	virtual void ExitState(UStateMachine* StateMachine);

	UFUNCTION(BlueprintCallable)
	virtual void UpdateState(UStateMachine* StateMachine, float Reward, float DeltaTime);

	UFUNCTION(BlueprintCallable)
	virtual TArray<UAction*> GetPossibleActions();
};
