// Copyright Epic Games, Inc. All Rights Reserved.

#include "StateTree/Conditions/STCondition_IsAlive.h"
#include "StateTree/FollowerStateTreeContext.h"

bool FSTCondition_IsAlive::TestCondition(FStateTreeExecutionContext& Context) const
{
	const FInstanceDataType& InstanceData = Context.GetInstanceData(*this);

	// Check if alive or dead depending on configuration
	bool bResult = InstanceData.bCheckIfDead ? !InstanceData.bIsAlive : InstanceData.bIsAlive;

	// DIAGNOSTIC: Log every evaluation
	static int32 EvalCounter = 0;
	EvalCounter++;

	UE_LOG(LogTemp, Warning, TEXT("🔍 [IS ALIVE] Eval #%d: bIsAlive=%d, bCheckIfDead=%d, Result=%d"),
		EvalCounter,
		InstanceData.bIsAlive ? 1 : 0,
		InstanceData.bCheckIfDead ? 1 : 0,
		bResult ? 1 : 0);

	return bResult;
}
