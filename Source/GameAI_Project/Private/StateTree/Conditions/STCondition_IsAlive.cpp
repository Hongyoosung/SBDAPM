// Copyright Epic Games, Inc. All Rights Reserved.

#include "StateTree/Conditions/STCondition_IsAlive.h"
#include "StateTree/FollowerStateTreeContext.h"

bool FSTCondition_IsAlive::TestCondition(FStateTreeExecutionContext& Context) const
{
	const FInstanceDataType& InstanceData = Context.GetInstanceData(*this);

	// Check if alive or dead depending on configuration
	return InstanceData.bCheckIfDead ? !InstanceData.bIsAlive : InstanceData.bIsAlive;
}
