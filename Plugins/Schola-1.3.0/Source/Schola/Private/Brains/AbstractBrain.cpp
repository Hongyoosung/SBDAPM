// Copyright (c) 2023 Advanced Micro Devices, Inc. All Rights Reserved.

#include "Brains/AbstractBrain.h"

UAbstractBrain::UAbstractBrain()
{
}

void UAbstractBrain::IncrementStep()
{
	this->Step += 1;
}

void UAbstractBrain::SetStep(int NewStep)
{
	this->Step = NewStep;
}

void UAbstractBrain::ResetStep()
{
	this->SetStep(0);
}

bool UAbstractBrain::IsActive()
{
	return this->GetStatus() != EBrainStatus::Error;
}

bool UAbstractBrain::IsDecisionStep(int StepToCheck)
{
	return (StepToCheck % this->DecisionRequestFrequency) == 0;
}

bool UAbstractBrain::IsDecisionStep()
{
	return IsDecisionStep(this->Step);
}

void UAbstractBrain::Init(UAbstractPolicy* InjectedPolicy)
{
	this->Policy = InjectedPolicy;
}

bool UAbstractBrain::GetAbstractSettingsVisibility() const
{
	return true;
}

EBrainStatus UAbstractBrain::GetStatus()
{
	return this->Status;
}

void UAbstractBrain::SetStatus(EBrainStatus NewStatus)
{
	this->Status = NewStatus;
}

void UAbstractBrain::UpdateStatusFromDecision(const FPolicyDecision& Decision)
{
	if (Decision.IsError())
	{
		this->Status = EBrainStatus::Error;
		// this->SetPolicyStatus(EPolicyStatus::Closed);
	}
	else if (Decision.IsAction())
	{
		this->Status = EBrainStatus::ActionReady;
	}
	// Do nothing if it was an empty decision
}

bool UAbstractBrain::IsActionStep()
{
	return this->HasAction() && (this->IsDecisionStep() || this->bTakeActionBetweenDecisions);
}
