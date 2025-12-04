// Copyright Epic Games, Inc. All Rights Reserved.

#include "StateTree/Conditions/STCondition_HasObjective.h"
#include "Team/Objective.h"

bool FSTCondition_HasObjective::TestCondition(FStateTreeExecutionContext& Context) const
{
	const FInstanceDataType& InstanceData = Context.GetInstanceData(*this);

	// Check if objective exists and is active
	bool bHasObjective = (InstanceData.CurrentObjective != nullptr) && InstanceData.bHasActiveObjective;

	// Apply inversion if requested
	bool bResult = InstanceData.bInvertCondition ? !bHasObjective : bHasObjective;

	// DIAGNOSTIC: Log every evaluation (throttled to avoid spam)
	static int32 EvalCounter = 0;
	EvalCounter++;

	if (EvalCounter % 60 == 0) // Log every 60 evaluations (~1 second at 60fps)
	{
		UE_LOG(LogTemp, Display, TEXT("🔍 [HAS OBJECTIVE] Eval #%d: Objective=%s, Active=%d, Inverted=%d, Result=%d"),
			EvalCounter,
			InstanceData.CurrentObjective ? TEXT("Valid") : TEXT("NULL"),
			InstanceData.bHasActiveObjective ? 1 : 0,
			InstanceData.bInvertCondition ? 1 : 0,
			bResult ? 1 : 0);
	}

	return bResult;
}
