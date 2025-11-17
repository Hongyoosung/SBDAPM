// Copyright Epic Games, Inc. All Rights Reserved.

#include "StateTree/FollowerStateTreeSchema.h"
#include "StateTreeExecutionContext.h"
#include "Components/StateTreeComponent.h"
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
	AIControllerClass = AAIController::StaticClass();
	PawnClass = APawn::StaticClass();
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
		// Allow all StateTree node types
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

		// Allow context struct
		if (InScriptStruct == FFollowerStateTreeContext::StaticStruct())
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
			InClass->IsChildOf(APawn::StaticClass()) ||
			InClass->IsChildOf(UFollowerAgentComponent::StaticClass()) ||
			InClass->IsChildOf(UTeamLeaderComponent::StaticClass()) ||
			InClass->IsChildOf(URLPolicyNetwork::StaticClass()))
		{
			return true;
		}

		// Allow Actor for target tracking
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
	TConstArrayView<FStateTreeExternalDataDesc> ParentDescs = Super::GetContextDataDescs();

	static TArray<FStateTreeExternalDataDesc> _ContextDataDescs;

	if (_ContextDataDescs.IsEmpty())
	{
		// Parent context descriptors
		_ContextDataDescs.Append(ParentDescs.GetData(), ParentDescs.Num());

		// UE 5.6 FIX: Use deterministic GUIDs based on descriptor names
		// These MUST be stable across editor restarts to preserve bindings

		// Primary context struct (contains all shared state data)
		FStateTreeExternalDataDesc ContextDesc;
		ContextDesc.Name = FName(TEXT("FollowerContext"));
		ContextDesc.Struct = FFollowerStateTreeContext::StaticStruct();
		ContextDesc.Requirement = EStateTreeExternalDataRequirement::Required;
		ContextDesc.ID = FGuid(0xA1B2C3D4, 0xE5F60001, 0x11223344, 0x55667788); // Deterministic GUID
		_ContextDataDescs.Add(ContextDesc);

		FStateTreeExternalDataDesc FollowerDesc;
		FollowerDesc.Name = FName(TEXT("FollowerComponent"));
		FollowerDesc.Struct = UFollowerAgentComponent::StaticClass();
		FollowerDesc.Requirement = EStateTreeExternalDataRequirement::Required;
		FollowerDesc.ID = FGuid(0xA1B2C3D4, 0xE5F60002, 0x11223344, 0x55667788); // Deterministic GUID
		_ContextDataDescs.Add(FollowerDesc);

		FStateTreeExternalDataDesc TeamLeaderDesc;
		TeamLeaderDesc.Name = FName(TEXT("TeamLeader"));
		TeamLeaderDesc.Struct = UTeamLeaderComponent::StaticClass();
		TeamLeaderDesc.Requirement = EStateTreeExternalDataRequirement::Optional;
		TeamLeaderDesc.ID = FGuid(0xA1B2C3D4, 0xE5F60003, 0x11223344, 0x55667788); // Deterministic GUID
		_ContextDataDescs.Add(TeamLeaderDesc);

		FStateTreeExternalDataDesc TacticalPolicyDesc;
		TacticalPolicyDesc.Name = FName(TEXT("TacticalPolicy"));
		TacticalPolicyDesc.Struct = URLPolicyNetwork::StaticClass();
		TacticalPolicyDesc.Requirement = EStateTreeExternalDataRequirement::Optional;
		TacticalPolicyDesc.ID = FGuid(0xA1B2C3D4, 0xE5F60004, 0x11223344, 0x55667788); // Deterministic GUID
		_ContextDataDescs.Add(TacticalPolicyDesc);
	}

	return _ContextDataDescs;
}