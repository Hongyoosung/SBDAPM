// Copyright (c) 2023 Advanced Micro Devices, Inc. All Rights Reserved.

#include "Brains/SynchronousBrain.h"

USynchronousBrain::USynchronousBrain()
{
}

USynchronousBrain::~USynchronousBrain()
{
}

bool USynchronousBrain::RequestDecision(const FDictPoint& Observations)
{
	TRACE_CPUPROFILER_EVENT_SCOPE_STR("Schola:Decision Request");

	TFuture<FPolicyDecision*> DecisionFuture = this->Policy->RequestDecision(Observations);
	// This waits Timeout Seconds for the communicator to make the decision
	// we got here by not time-ing out or not setting a timeout in the first place
	this->InProgressActionRequest = std::move(DecisionFuture);
	if (this->Decision.IsSet())
	{
		this->Decision.Reset();
	}
	bHasInProgressAction = true;
	return true;
}

void USynchronousBrain::Reset()
{
	this->ResetStep();
}

FAction* USynchronousBrain::GetAction()
{
	// Get our current action
	return &(this->Decision.GetValue()->Action);
}

bool USynchronousBrain::HasAction()
{
	return bHasInProgressAction || this->Decision.IsSet();
}

void USynchronousBrain::ResolveDecision()
{
	TRACE_CPUPROFILER_EVENT_SCOPE_STR("Schola: Retrieving Action");
	// If we have an inflight request to the policy
	if (bHasInProgressAction)
	{
		if (bUseTimeout && !InProgressActionRequest.WaitFor(FTimespan(0, 0, Timeout)))
		{
			UE_LOG(LogSchola, Warning, TEXT("Synchronous Brain Timeout"))
			this->UpdateStatusFromDecision(FPolicyDecision(EDecisionType::ERRORED));
			InProgressActionRequest.Reset();
		}
		else
		{
			// We know this will deliver because we already waited
			// Could deliver a nullptr if we
			this->Decision = InProgressActionRequest.Get();
			this->UpdateStatusFromDecision(*(this->Decision.GetValue()));
		}
		// We either got our action or we tossed our old one
		bHasInProgressAction = false;
	}
}
