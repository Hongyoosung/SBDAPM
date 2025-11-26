// Copyright Epic Games, Inc. All Rights Reserved.

#include "StateTree/FollowerStateTreeSchema.h"
#include "StateTree/FollowerStateTreeComponent.h"
#include "StateTreeExecutionContext.h"
#include "Components/StateTreeComponent.h"
#include "AI/AIController/FollowerAIController.h"
#include "Actor/FollowerCharacter.h"
#include "Team/FollowerAgentComponent.h"
#include "Team/TeamLeaderComponent.h"
#include "RL/RLPolicyNetwork.h"
#include "StateTree/Conditions/STCondition_IsAlive.h"
#include "StateTree/Conditions/STCondition_CheckObjectiveType.h"
#include "StateTree/Tasks/STTask_ExecuteObjective.h"
#include "StateTree/Tasks/STTask_Dead.h"
#include "StateTree/Evaluators/STEvaluator_SyncObjective.h"
#include "StateTree/Evaluators/STEvaluator_UpdateObservation.h"
#include "StateTree/Evaluators/STEvaluator_SpatialContext.h"


UFollowerStateTreeSchema::UFollowerStateTreeSchema()
{
	AIControllerClass = AFollowerAIController::StaticClass();
	PawnClass = AFollowerCharacter::StaticClass();

	ContextDataDescs.Reset();

	{
		FStateTreeExternalDataDesc ActorDesc(
			FName("Actor"),
			AActor::StaticClass(),
			FGuid(0x1D971B00, 0x28884FDE, 0xB5436802, 0x36984FD5)
		);
		ActorDesc.Requirement = EStateTreeExternalDataRequirement::Required;
		ContextDataDescs.Add(ActorDesc);
	}


	// 1. FollowerContext
	{
		FStateTreeExternalDataDesc Desc(FName(TEXT("FollowerContext")), FFollowerStateTreeContext::StaticStruct(), FGuid(0x4F111111, 0x11112222, 0x33334444, 0x00000001));
		Desc.Requirement = EStateTreeExternalDataRequirement::Required;
		ContextDataDescs.Add(Desc);
	}

	// 2. FollowerComponent
	{
		FStateTreeExternalDataDesc Desc(FName(TEXT("FollowerComponent")), UFollowerAgentComponent::StaticClass(), FGuid(0x4F111111, 0x11112222, 0x33334444, 0x00000002));
		Desc.Requirement = EStateTreeExternalDataRequirement::Required;
		ContextDataDescs.Add(Desc);
	}

	// 3. FollowerStateTreeComponent
	{
		FStateTreeExternalDataDesc Desc(FName(TEXT("FollowerStateTreeComponent")), UFollowerStateTreeComponent::StaticClass(), FGuid(0x4F111111, 0x11112222, 0x33334444, 0x00000003));
		Desc.Requirement = EStateTreeExternalDataRequirement::Required;
		ContextDataDescs.Add(Desc);
	}

	// 4. TeamLeader
	{
		FStateTreeExternalDataDesc Desc(FName(TEXT("TeamLeader")), UTeamLeaderComponent::StaticClass(), FGuid(0x4F111111, 0x11112222, 0x33334444, 0x00000004));
		Desc.Requirement = EStateTreeExternalDataRequirement::Optional;
		ContextDataDescs.Add(Desc);
	}

	// 5. TacticalPolicy
	{
		FStateTreeExternalDataDesc Desc(FName(TEXT("TacticalPolicy")), URLPolicyNetwork::StaticClass(), FGuid(0x4F111111, 0x11112222, 0x33334444, 0x00000005));
		Desc.Requirement = EStateTreeExternalDataRequirement::Optional;
		ContextDataDescs.Add(Desc);
	}
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
		// Allow all StateTree node types (v3.0)
		if (InScriptStruct->IsChildOf(FSTEvaluator_SyncObjective::StaticStruct()) ||
			InScriptStruct->IsChildOf(FSTEvaluator_UpdateObservation::StaticStruct()) ||
			InScriptStruct->IsChildOf(FSTEvaluator_SpatialContext::StaticStruct()) ||
			InScriptStruct->IsChildOf(FSTCondition_IsAlive::StaticStruct()) ||
			InScriptStruct->IsChildOf(FSTCondition_CheckObjectiveType::StaticStruct()) ||
			InScriptStruct->IsChildOf(FSTTask_ExecuteObjective::StaticStruct()) ||
			InScriptStruct->IsChildOf(FSTTask_Dead::StaticStruct()))
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
			InClass->IsChildOf(URLPolicyNetwork::StaticClass()) ||
			InClass->IsChildOf(UFollowerStateTreeComponent::StaticClass()))
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