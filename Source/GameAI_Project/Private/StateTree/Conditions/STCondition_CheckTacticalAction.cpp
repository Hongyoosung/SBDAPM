// Copyright Epic Games, Inc. All Rights Reserved.

#include "StateTree/Conditions/STCondition_CheckTacticalAction.h"
#include "StateTree/FollowerStateTreeContext.h"

bool FSTCondition_CheckTacticalAction::TestCondition(FStateTreeExecutionContext& Context) const
{
	const FInstanceDataType& InstanceData = Context.GetInstanceData(*this);

	// Check if tactical action matches any accepted actions
	bool bMatches = InstanceData.AcceptedActions.Contains(InstanceData.CurrentTacticalAction);

	// Apply inversion if needed
	return InstanceData.bInvertCondition ? !bMatches : bMatches;
}
