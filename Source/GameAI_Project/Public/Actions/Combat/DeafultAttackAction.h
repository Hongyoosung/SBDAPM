// Fill out your copyright notice in the Description page of Project Settings.

#pragma once

#include "CoreMinimal.h"
#include "Actions/Action.h"
#include "DeafultAttackAction.generated.h"

/**
 * 
 */
UCLASS()
class GAMEAI_PROJECT_API UDeafultAttackAction : public UAction
{
	GENERATED_BODY()
	
public:
	virtual void ExecuteAction(UStateMachine* StateMachine) override;
};
