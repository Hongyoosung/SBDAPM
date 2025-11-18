// Copyright Epic Games, Inc. All Rights Reserved.

#pragma once

#include "CoreMinimal.h"
#include "Components\StateTreeComponentSchema.h"
#include "FollowerStateTreeContext.h"
#include "FollowerStateTreeSchema.generated.h"

class AAIController;

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
UCLASS(BlueprintType, meta = (DisplayName = "Follower State Tree Schema", ToolTip = "Schema for follower agent State Trees with RL policy integration", CommonSchema))
class GAMEAI_PROJECT_API UFollowerStateTreeSchema : public UStateTreeComponentSchema
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

protected:
	// Override to provide custom context data (FollowerContext, FollowerComponent, etc.)
	virtual void SetContextData(FContextDataSetter& ContextDataSetter, bool bLogErrors) const override;

	/** AIController class for this schema */
	UPROPERTY(EditAnywhere, Category = "Defaults", NoClear)
	TSubclassOf<AAIController> AIControllerClass;

	/** Pawn class (UE 5.6 - allows access to Pawn components) */
	UPROPERTY(EditAnywhere, Category = "Defaults", NoClear, meta = (DisplayName = "Pawn Class"))
	TSubclassOf<APawn> PawnClass;
};
