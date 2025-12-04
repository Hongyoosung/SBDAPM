// Copyright Epic Games, Inc. All Rights Reserved.

#include "StateTree/Conditions/STCondition_CheckObjectiveType.h"

bool FSTCondition_CheckObjectiveType::TestCondition(FStateTreeExecutionContext& Context) const
{
	const FInstanceDataType& InstanceData = Context.GetInstanceData(*this);

	// Get objective data from bound properties
	const UObjective* CurrentObjective = InstanceData.CurrentObjective;
	const bool bHasActiveObjective = InstanceData.bHasActiveObjective;

	// DIAGNOSTIC: Log every evaluation
	static int32 EvalCounter = 0;
	EvalCounter++;

	UE_LOG(LogTemp, Warning, TEXT("🔍 [CHECK OBJECTIVE TYPE] Eval #%d: Objective=%s, Active=%d, AcceptedTypes=%d, RequireActive=%d"),
		EvalCounter,
		CurrentObjective ? *UEnum::GetValueAsString(CurrentObjective->Type) : TEXT("NULL"),
		bHasActiveObjective ? 1 : 0,
		InstanceData.AcceptedObjectiveTypes.Num(),
		InstanceData.bRequireActiveObjective ? 1 : 0);

	// Check if no accepted types configured
	if (InstanceData.AcceptedObjectiveTypes.Num() == 0)
	{
		UE_LOG(LogTemp, Error, TEXT("[CHECK OBJECTIVE TYPE] ❌ No AcceptedObjectiveTypes configured!"));
		return InstanceData.bInvertCondition;
	}

	// Check validity if required
	if (InstanceData.bRequireActiveObjective && !bHasActiveObjective)
	{
		UE_LOG(LogTemp, Warning, TEXT("[CHECK OBJECTIVE TYPE] ❌ RequireActive=true but bHasActiveObjective=false → FAIL"));
		return InstanceData.bInvertCondition; // No active objective = false (or true if inverted)
	}

	// Check if objective is null
	if (!CurrentObjective)
	{
		UE_LOG(LogTemp, Warning, TEXT("[CHECK OBJECTIVE TYPE] ❌ CurrentObjective is NULL → FAIL"));
		return InstanceData.bInvertCondition; // Null objective = false (or true if inverted)
	}

	// Check if objective type matches any accepted types
	bool bMatches = InstanceData.AcceptedObjectiveTypes.Contains(CurrentObjective->Type);

	bool bResult = InstanceData.bInvertCondition ? !bMatches : bMatches;

	// Log accepted types for debugging
	FString AcceptedTypesStr;
	for (EObjectiveType Type : InstanceData.AcceptedObjectiveTypes)
	{
		AcceptedTypesStr += UEnum::GetValueAsString(Type) + TEXT(", ");
	}

	UE_LOG(LogTemp, Warning, TEXT("[CHECK OBJECTIVE TYPE] %s CurrentType=%s, AcceptedTypes=[%s], Matches=%d, Inverted=%d, Result=%d"),
		bResult ? TEXT("✅") : TEXT("❌"),
		*UEnum::GetValueAsString(CurrentObjective->Type),
		*AcceptedTypesStr,
		bMatches ? 1 : 0,
		InstanceData.bInvertCondition ? 1 : 0,
		bResult ? 1 : 0);

	return bResult;
}
