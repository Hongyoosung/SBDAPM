// Fill out your copyright notice in the Description page of Project Settings.

#pragma once

#include "CoreMinimal.h"
#include "States/State.h"
#include "AI/MCTS.h"
#include "AttackState.generated.h"

/**
 * 
 */
UCLASS(Blueprintable)
class GAMEAI_PROJECT_API UAttackState : public UState
{
	GENERATED_BODY()
	
public:
	virtual void EnterState(UStateMachine* StateMachine) override;
	virtual void UpdateState(UStateMachine* StateMachine, float Reward, float DeltaTime) override;
	virtual void ExitState(UStateMachine* StateMachine) override;
	virtual TArray<UAction*> GetPossibleActions() override;

private:
	UPROPERTY(EditAnywhere, Category = "Attack State")
	UMCTS* MCTS;

	UPROPERTY(EditAnywhere, Category = "Attack State")
	UMCTSNode* BestChild;

	UPROPERTY(EditAnywhere, Category = "Attack State")
	TArray<UAction*> PossibleActions;
};
