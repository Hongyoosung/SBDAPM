// Copyright Epic Games, Inc. All Rights Reserved.

#include "StateTree/Conditions/STCondition_CheckCommandType.h"
#include "StateTree/FollowerStateTreeContext.h"

bool FSTCondition_CheckCommandType::TestCondition(FStateTreeExecutionContext& Context) const
{
	const FInstanceDataType& InstanceData = Context.GetInstanceData(*this);

	// Check validity if required
	if (InstanceData.bRequireValidCommand && !InstanceData.bIsCommandValid)
	{
		return InstanceData.bInvertCondition; // Invalid command = false (or true if inverted)
	}

	// Check if command type matches any accepted types
	bool bMatches = InstanceData.AcceptedCommandTypes.Contains(InstanceData.CurrentCommand.CommandType);

	// Apply inversion if needed
	return InstanceData.bInvertCondition ? !bMatches : bMatches;
}
