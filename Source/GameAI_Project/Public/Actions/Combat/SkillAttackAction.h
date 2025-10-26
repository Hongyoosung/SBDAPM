// Fill out your copyright notice in the Description page of Project Settings.

#pragma once

#include "CoreMinimal.h"
#include "Actions/Action.h"
#include "SkillAttackAction.generated.h"

/**
 * 
 */
UCLASS()
class GAMEAI_PROJECT_API USkillAttackAction : public UAction
{
	GENERATED_BODY()
	
public:
	virtual void ExecuteAction(UStateMachine* StateMachine) override;
};
