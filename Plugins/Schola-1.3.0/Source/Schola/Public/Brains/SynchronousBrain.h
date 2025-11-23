// Copyright (c) 2023 Advanced Micro Devices, Inc. All Rights Reserved.

#pragma once

#include "Brains/AbstractBrain.h"
#include "Common/LogSchola.h"
#include "CoreMinimal.h"
#include "SynchronousBrain.generated.h"

UCLASS(Blueprintable)
class SCHOLA_API USynchronousBrain : public UAbstractBrain
{
	GENERATED_BODY()

private:
	TOptional<FPolicyDecision*> Decision = TOptional<FPolicyDecision*>();
	// Can't use TOptional here because the copy/move constructor is private for TOptional
	TFuture<FPolicyDecision*> InProgressActionRequest = TFuture<FPolicyDecision*>();
	bool					  bHasInProgressAction = false;

public:
	/** How long should we wait before assuming decision request has failed. */
	UPROPERTY(EditAnywhere, BlueprintReadOnly, meta = (ClampMin = "0", EditCondition = "bUseTimeout", Units = "s"), Category = "Brain Settings")
	int Timeout = 30;

	UPROPERTY(EditAnywhere, BlueprintReadOnly, meta = (InlineEditConditionToggle), Category = "Brain Settings")
	bool bUseTimeout = true;

	USynchronousBrain();
	~USynchronousBrain();

	bool	 RequestDecision(const FDictPoint& Observations) override;
	void	 Reset() override;
	FAction* GetAction() override;
	bool	 HasAction() override;
	void	 ResolveDecision() override;
};
