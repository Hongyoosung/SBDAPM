// Copyright Epic Games, Inc. All Rights Reserved.

#pragma once

#include "CoreMinimal.h"
#include "StateTreeSchema.h"
#include "FollowerStateTreeContext.h"
#include "FollowerStateTreeSchema.generated.h"

/**
 * State Tree Schema for Follower Agents
 *
 * Defines the context structure and allowed types for follower agent State Trees.
 * This schema enables:
 * - Command-driven state transitions (from Team Leader)
 * - RL policy integration for tactical decisions
 * - Observation-based execution
 * - Component references without Blackboard overhead
 *
 * Usage:
 * 1. Create a State Tree asset in the Content Browser
 * 2. Select "Follower State Tree Schema" as the schema type
 * 3. The FFollowerStateTreeContext will be available to all nodes
 */
UCLASS(BlueprintType, EditInlineNew, CollapseCategories, meta = (DisplayName = "Follower State Tree Schema"))
class GAMEAI_PROJECT_API UFollowerStateTreeSchema : public UStateTreeSchema
{
	GENERATED_BODY()

public:
	UFollowerStateTreeSchema();

public:
	// UStateTreeSchema interface
	virtual bool IsStructAllowed(const UScriptStruct* InScriptStruct) const override;
	virtual bool IsClassAllowed(const UClass* InClass) const override;
	virtual bool IsExternalItemAllowed(const UStruct& InStruct) const override;
	virtual TConstArrayView<FStateTreeExternalDataDesc> GetContextDataDescs() const override;

	UPROPERTY()
	TArray<FStateTreeExternalDataDesc> ContextDataDescs;

};
