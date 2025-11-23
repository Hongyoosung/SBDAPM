// Copyright (c) 2023 Advanced Micro Devices, Inc. All Rights Reserved.

#pragma once

#include "CoreMinimal.h"
#include "Agent/AgentAction.h"
#include "PolicyDecision.generated.h"

/**
 * @brief An enumeration representing the type of action taken by an agent.
 */
UENUM(BlueprintType)
enum class EDecisionType : uint8
{
	ACTION = 0 UMETA(DisplayName = "Action"),
	NONE = 1 UMETA(DisplayName = "(Meta)No Action"), // Unused
	ERRORED = 3 UMETA(DisplayName = "(Meta)Error"),
};

/**
 * @brief A struct representing a decision made by a policy.
 */
USTRUCT()
struct SCHOLA_API FPolicyDecision
{
	GENERATED_BODY()

	/** The type of decision made by the policy */
	EDecisionType DecisionType = EDecisionType::NONE;

	/** The action taken by the agent */
	FAction Action;

	/** 
	 * @brief Construct a default policy decision.
	 */
	FPolicyDecision(){};

	/** 
	 * @brief Construct a policy decision with a given decision type.
	 * @param[in] DecisionType The type of decision made by the policy.
	 */
	FPolicyDecision(EDecisionType DecisionType)
	{
		this->DecisionType = DecisionType;
	}

	/** 
	 * @brief Construct an policy decision of type action from a given action.
	 * @param[in] Action The action taken by the agent.
	 */
	FPolicyDecision(FAction& Action)
	{
		this->DecisionType = EDecisionType::ACTION;
		this->Action = Action;
	}

	// Helpers for checking the type of the Decision

	/**
	 * @brief Check if the decision is an action.
	 * @return True if the decision is an action, false otherwise.
	 */
	bool IsAction() const
	{
		return DecisionType == EDecisionType::ACTION;
	}

	/**
	 * @brief Check if the decision resulted in an error.
	 * @return True if the decision resulted in an error, false otherwise.
	 */
	bool IsError() const
	{
		return DecisionType == EDecisionType::ERRORED;
	}

	/**
	 * @brief Check if the decision is empty.
	 * @return True if the decision is empty, false otherwise.
	 */
	bool IsEmpty() const
	{
		return DecisionType == EDecisionType::NONE;
	}

	// Helpers for Creating Specific Policy Decisions Easier
	
	/**
	 * @brief Create an empty policy decision.
	 * @return A ptr to an empty policy decision.
	 */
	static FPolicyDecision* NoDecision()
	{
		return new FPolicyDecision(EDecisionType::NONE);
	}

	/**
	 * @brief Create a policy decision with a given action.
	 * @param[in] Action The action taken by the agent.
	 * @return A ptr to an action policy decision.
	 */
	static FPolicyDecision* ActionDecision(FAction& Action)
	{
		return new FPolicyDecision(Action);
	}

	/**
	 * @brief Create a policy decision representing an error
	 * @return A ptr to an error policy decision
	 */
	static FPolicyDecision* PolicyError()
	{
		return new FPolicyDecision(EDecisionType::ERRORED);
	}
};