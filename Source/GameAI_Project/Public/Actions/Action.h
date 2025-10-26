// Fill out your copyright notice in the Description page of Project Settings.

#pragma once

#include "CoreMinimal.h"
#include "UObject/NoExportTypes.h"
#include "Action.generated.h"

/**
 * 
 */
class UStateMachine;

UCLASS()
class GAMEAI_PROJECT_API UAction : public UObject
{
	GENERATED_BODY()
	
public:
	virtual void ExecuteAction(UStateMachine* StateMachine);
};
