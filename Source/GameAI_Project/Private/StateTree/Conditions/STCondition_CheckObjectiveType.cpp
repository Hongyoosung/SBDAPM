// Copyright Epic Games, Inc. All Rights Reserved.

#include "StateTree/Conditions/STCondition_CheckObjectiveType.h"

bool FSTCondition_CheckObjectiveType::TestCondition(FStateTreeExecutionContext& Context) const
{
	const FInstanceDataType& InstanceData = Context.GetInstanceData(*this);

	// Get objective data from bound properties
	const UObjective* CurrentObjective = InstanceData.CurrentObjective;
	const bool bHasActiveObjective = InstanceData.bHasActiveObjective;

	// Check if no accepted types configured
	if (InstanceData.AcceptedObjectiveTypes.Num() == 0)
	{
		UE_LOG(LogTemp, Error, TEXT("[CHECK OBJECTIVE TYPE] No AcceptedObjectiveTypes configured!"));
		return InstanceData.bInvertCondition;
	}

	// Check validity if required
	if (InstanceData.bRequireActiveObjective && !bHasActiveObjective)
	{
		return InstanceData.bInvertCondition; // No active objective = false (or true if inverted)
	}

	// Check if objective is null
	if (!CurrentObjective)
	{
		return InstanceData.bInvertCondition; // Null objective = false (or true if inverted)
	}

	// Check if objective type matches any accepted types
	bool bMatches = InstanceData.AcceptedObjectiveTypes.Contains(CurrentObjective->Type);

	bool bResult = InstanceData.bInvertCondition ? !bMatches : bMatches;

	return bResult;
}
