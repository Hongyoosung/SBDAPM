// Copyright (c) 2025 Advanced Micro Devices, Inc. All Rights Reserved.

#include "Inference/BT/BTT_AgentStep.h"

UBTTask_AgentStep::UBTTask_AgentStep(FObjectInitializer const& ObjectInitializer)
	: Super(ObjectInitializer)
{
	NodeName = "Agent Step";
	bCreateNodeInstance = true;
	INIT_TASK_NODE_NOTIFY_FLAGS();
}

EBTNodeResult::Type UBTTask_AgentStep::ExecuteTask(UBehaviorTreeComponent& OwnerComp, uint8* NodeMemory)
{
	if (!bUseTickTask)
	{
		Step(OwnerComp);
		return EBTNodeResult::Succeeded;
	}
	return EBTNodeResult::InProgress;
}

void UBTTask_AgentStep::TickTask(UBehaviorTreeComponent& OwnerComp, uint8* NodeMemory, float DeltaSeconds)
{
	if (bUseTickTask)
	{
		TaskResult = EBTNodeResult::InProgress;
		Step(OwnerComp);
	}
}

void UBTTask_AgentStep::Step(UBehaviorTreeComponent& OwnerComp)
{
	if (!bConnected)
	{
		OwnerPawn = Cast<APawn>(Cast<AAIController>(OwnerComp.GetAIOwner())->GetPawn());
		this->SetupDefaultTicking(this->ThinkTickFunction, this->ActTickFunction);
		this->Initialize();
		bConnected = true;
	}

	if (Status != EAgentStatus::Running)
	{
		this->SetStatus(EAgentStatus::Running);
	}
}
