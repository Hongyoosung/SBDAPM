// Fill out your copyright notice in the Description page of Project Settings.


#include "Actions/Combat/DeafultAttackAction.h"
#include "Core/StateMachine.h"

void UDeafultAttackAction::ExecuteAction(UStateMachine* StateMachine)
{
	UE_LOG(LogTemp, Warning, TEXT("------------Default Attack"));
	//StateMachine->TriggerBlueprintEvent("Default Attack");
}
