// Copyright Epic Games, Inc. All Rights Reserved.

#include "StateTree/FollowerStateTreeSchema.h"
#include "StateTreeExecutionContext.h"
#include "AIController.h"
#include "Team/FollowerAgentComponent.h"
#include "Team/TeamLeaderComponent.h"
#include "RL/RLPolicyNetwork.h"
#include "StateTree/Conditions/STCondition_IsAlive.h"
#include "StateTree/Conditions/STCondition_CheckCommandType.h"
#include "StateTree/Conditions/STCondition_CheckTacticalAction.h"
#include "StateTree/Tasks/STTask_ExecuteAssault.h"
#include "StateTree/Tasks/STTask_ExecuteDefend.h"
#include "StateTree/Tasks/STTask_ExecuteSupport.h"
#include "StateTree/Tasks/STTask_ExecuteMove.h"
#include "StateTree/Tasks/STTask_ExecuteRetreat.h"
#include "StateTree/Tasks/STTask_QueryRLPolicy.h"
#include "StateTree/Evaluators/STEvaluator_SyncCommand.h"
#include "StateTree/Evaluators/STEvaluator_UpdateObservation.h"


UFollowerStateTreeSchema::UFollowerStateTreeSchema()
{
	// Register individual components as bindable external data
	// This allows tasks/evaluators/conditions to bind directly to these components
	ContextDataDescs.Reset();

	// 1. FollowerAgentComponent - Primary component for agent logic
	FStateTreeExternalDataDesc& FollowerDesc = ContextDataDescs.AddDefaulted_GetRef();
	FollowerDesc.Struct = UFollowerAgentComponent::StaticClass();
	FollowerDesc.Name = FName(TEXT("FollowerComponent"));
	FollowerDesc.ID = FGuid::NewGuid();
	FollowerDesc.Requirement = EStateTreeExternalDataRequirement::Required;

	// 2. AIController - For movement/perception/navigation
	FStateTreeExternalDataDesc& AIControllerDesc = ContextDataDescs.AddDefaulted_GetRef();
	AIControllerDesc.Struct = AAIController::StaticClass();
	AIControllerDesc.Name = FName(TEXT("AIController"));
	AIControllerDesc.ID = FGuid::NewGuid();
	AIControllerDesc.Requirement = EStateTreeExternalDataRequirement::Required;

	// 3. StateTree Context - Contains shared state data
	FStateTreeExternalDataDesc& ContextDesc = ContextDataDescs.AddDefaulted_GetRef();
	ContextDesc.Struct = FFollowerStateTreeContext::StaticStruct();
	ContextDesc.Name = FName(TEXT("FollowerContext"));
	ContextDesc.ID = FGuid::NewGuid();
	ContextDesc.Requirement = EStateTreeExternalDataRequirement::Required;
}

bool UFollowerStateTreeSchema::IsStructAllowed(const UScriptStruct* InScriptStruct) const
{
	// Allow base State Tree structs
	if (Super::IsStructAllowed(InScriptStruct))
	{
		return true;
	}

	// Allow project-specific structs
	if (InScriptStruct)
	{
		// Allow all StateTree base types
		if (InScriptStruct->IsChildOf(FSTEvaluator_SyncCommand::StaticStruct()) ||
			InScriptStruct->IsChildOf(FSTEvaluator_UpdateObservation::StaticStruct()) ||
			InScriptStruct->IsChildOf(FSTCondition_IsAlive::StaticStruct()) ||
			InScriptStruct->IsChildOf(FSTCondition_CheckCommandType::StaticStruct()) ||
			InScriptStruct->IsChildOf(FSTCondition_CheckTacticalAction::StaticStruct()) ||
			InScriptStruct->IsChildOf(FSTTask_ExecuteAssault::StaticStruct()) ||
			InScriptStruct->IsChildOf(FSTTask_ExecuteDefend::StaticStruct()) ||
			InScriptStruct->IsChildOf(FSTTask_ExecuteSupport::StaticStruct()) ||
			InScriptStruct->IsChildOf(FSTTask_QueryRLPolicy::StaticStruct()) ||
			InScriptStruct->IsChildOf(FSTTask_ExecuteMove::StaticStruct()) ||
			InScriptStruct->IsChildOf(FSTTask_ExecuteRetreat::StaticStruct()))
		{
			return true;
		}
		
		// Allow observation structs
		if (InScriptStruct->GetFName() == FName(TEXT("ObservationElement")))
		{
			return true;
		}

		// Allow RL types
		if (InScriptStruct->GetFName() == FName(TEXT("TacticalAction")) ||
			InScriptStruct->GetFName() == FName(TEXT("StrategicCommand")))
		{
			return true;
		}

		// Allow team types
		if (InScriptStruct->GetFName() == FName(TEXT("TeamID")) ||
			InScriptStruct->GetFName() == FName(TEXT("TeamMessage")))
		{
			return true;
		}
	}

	return false;
}

bool UFollowerStateTreeSchema::IsClassAllowed(const UClass* InClass) const
{
	// Allow base State Tree classes
	if (Super::IsClassAllowed(InClass))
	{
		return true;
	}

	// Allow AI-related classes
	if (InClass)
	{
		if (InClass->IsChildOf(AAIController::StaticClass()) ||
			InClass->IsChildOf(UFollowerAgentComponent::StaticClass()) ||
			InClass->IsChildOf(UTeamLeaderComponent::StaticClass()) ||
			InClass->IsChildOf(URLPolicyNetwork::StaticClass()))
		{
			return true;
		}

		// Allow Actor and APawn for target tracking
		if (InClass->IsChildOf(AActor::StaticClass()))
		{
			return true;
		}
	}

	return false;
}

bool UFollowerStateTreeSchema::IsExternalItemAllowed(const UStruct& InStruct) const
{
	// Allow base State Tree external items
	if (Super::IsExternalItemAllowed(InStruct))
	{
		return true;
	}

	// Allow our context struct
	if (&InStruct == FFollowerStateTreeContext::StaticStruct())
	{
		return true;
	}

	return false;
}

TConstArrayView<FStateTreeExternalDataDesc> UFollowerStateTreeSchema::GetContextDataDescs() const
{
	return ContextDataDescs;
}
