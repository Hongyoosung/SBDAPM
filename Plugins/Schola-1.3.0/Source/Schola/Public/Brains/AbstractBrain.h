// Copyright (c) 2023 Advanced Micro Devices, Inc. All Rights Reserved.

#pragma once

#include "Common/Spaces.h"
#include "CoreMinimal.h"
#include "Policies/AbstractPolicy.h"
#include "Common/LogSchola.h"
#include "AbstractBrain.generated.h"

/**
 * @brief An Enum defining the status of a brain. 
 */
UENUM(BlueprintType)
enum class EBrainStatus : uint8
{
	ActionReady UMETA(DisplayName = "Action Ready"),
	Running		UMETA(DisplayName = "Running"),
	Error		UMETA(DisplayName = "Error"),
};

/**
 * @brief An AbstractBaseClass for subclasses representing different ways of synchronizing policy, observations and actions.
 */
UCLASS(Abstract, EditInlineNew, AutoExpandCategories = ("Brain Settings"))
class SCHOLA_API UAbstractBrain : public UObject
{
	GENERATED_BODY()

public:
	/** The underlying policy this brain wraps */
	UPROPERTY()
	UAbstractPolicy* Policy;

	UAbstractBrain();

	/** If true the agent will repeat it's last action each step between decision requests */
	UPROPERTY(EditAnywhere, meta = (EditCondition = "bAbstractSettingsVisibility", EditConditionHides, HideEditConditionToggle), Category = "Brain Settings")
	bool bTakeActionBetweenDecisions = true;

	/** The current step of the simulation */
	UPROPERTY(VisibleAnywhere, Category = "Brain Settings")
	int Step = 0;

	/** The number of steps between requests for new actions. If this is different across agents it may cause issues training in some frameworks (e.g. Stable Baselines 3). */
	UPROPERTY(EditAnywhere, meta = (EditCondition = "bAbstractSettingsVisibility", EditConditionHides, HideEditConditionToggle), Category = "Brain Settings")
	int DecisionRequestFrequency = 5;

	/** Toggle for whether the user can see the settings for this class. Use to hide in subclass if they aren't relevant */
	UPROPERTY()
	bool bAbstractSettingsVisibility = true;

	/** The status of the brain */
	UPROPERTY()
	EBrainStatus Status = EBrainStatus::Running;

	/**
	 * @brief Increment the current agent step
	 */
	void IncrementStep();

	/**
	 * @brief Set the current agent step
	 * @param NewStep the value to update the step too
	 */
	void SetStep(int NewStep);

	/**
	 * @brief Reset the agent's count of steps to 0
	 */
	void ResetStep();

	/**
	 * @brief Check if this brain is active (e.g. Not closed or errored out)
	 * @return true iff this brain is functional (e.g. Not closed or errored out)
	 */
	bool IsActive();

	/**
	 * @brief Reset this brain
	 */
	virtual void Reset() PURE_VIRTUAL(UAbstractBrain::Reset, return; );

	/**
	 * @brief Check whether a specific step will require a brain decision
	 * @param StepToCheck the timestep to check
	 * @return true iff the agent should be requesting a decision
	 */
	virtual bool IsDecisionStep(int StepToCheck);

	/**
	 * @brief If the current step is a decision step, as defined by the step frequency
	 * @return true iff the current step is a decision step
	 */
	virtual bool IsDecisionStep();

	/**
	 * @brief Initialize this brain by supplying a policy
	 * @param[in,out] InjectedPolicy The policy that this brain will use to make decisions
	 * @note this is so that we can avoid having massively nested structs in the editor when opening the Agent
	 */
	void Init(UAbstractPolicy* InjectedPolicy);

	/**
	 * @brief Check if this brain has an action prepared
	 * @return true iff this brain has an action prepared (e.g. a GetAction() call on this step will suceed)
	 */
	virtual bool HasAction() PURE_VIRTUAL(UAbstractBrain::HasAction, return true;);

	/**
	 * @brief get an action from this brain
	 * @return A pointer to the current action
	 */
	virtual FAction* GetAction() PURE_VIRTUAL(UAbstractBrain::GetAction, return nullptr;);
	/**
	 * @brief Request that the brain determine a new action
	 * @param[in] Observations The current state of the agent used to inform the brains choice of action
	 * @return Status True if decision request suceeded and False otherwise
	 */
	virtual bool RequestDecision(const FDictPoint& Observations) PURE_VIRTUAL(UAbstractBrain::RequestDecision, return true;);
	/**
	 * @brief Use by subclasses to set whether the settings are visible or not
	 * @return true if settings in this class are visible in the editor.
	 */
	virtual bool GetAbstractSettingsVisibility() const;

	/**
	 * @brief Get the last status of the brain
	 * @return the last status
	 */
	virtual EBrainStatus GetStatus();

	/**
	 * @brief Update the status of the brain
	 * @param[in] NewStatus The new status to set
	 */
	virtual void SetStatus(EBrainStatus NewStatus);

	/**
	 * @brief Update the status of the brain from a PolicyDecision
	 * @param[in] Decision The PolicyDecision to unpack and use when updating the status
	 */
	virtual void UpdateStatusFromDecision(const FPolicyDecision& Decision);
	
	/**
	 * @brief Inform the policy that you will require a response decision iminently, so it should resolve the open decision and update it's status accordingly
	 */
	virtual void ResolveDecision(){};

	/**
	 * @brief Check if brain has an action, and it's an action step
	 * @return true if the brain has an action and it's an action step
	 */
	virtual bool IsActionStep();
};
