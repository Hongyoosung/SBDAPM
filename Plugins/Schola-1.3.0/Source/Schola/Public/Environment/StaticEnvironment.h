// Copyright (c) 2023-2025 Advanced Micro Devices, Inc. All Rights Reserved.

#pragma once
#include "Environment/AbstractEnvironment.h"
#include "StaticEnvironment.generated.h"

UCLASS(Blueprintable,Abstract)
class SCHOLA_API AStaticScholaEnvironment : public AAbstractScholaEnvironment
{
	GENERATED_BODY()

protected:
	AStaticScholaEnvironment(const FObjectInitializer& ObjectInitializer);
	virtual ~AStaticScholaEnvironment();

public:
	virtual void InternalRegisterAgents(TArray<FTrainerAgentPair>& OutAgentTrainerPairs);
	
	//TODO add PURE_VIRTUAL macro
	virtual void RegisterAgents(TArray<APawn*>& OutTrainerControlledPawns){};
};

/**
 * @brief A blueprintable version of the StaticScholaEnvironment, that features trainers spawned by the pawns they control
 */
UCLASS(Blueprintable, Abstract)
class SCHOLA_API ABlueprintStaticScholaEnvironment : public AStaticScholaEnvironment
{
	GENERATED_BODY()

public:
	UFUNCTION(BlueprintImplementableEvent)
	void ResetEnvironment();

	UFUNCTION(BlueprintImplementableEvent)
	void RegisterAgents(TArray<APawn*>& OutTrainerControlledPawns);

	UFUNCTION(BlueprintImplementableEvent)
	void InitializeEnvironment();

	UFUNCTION(BlueprintImplementableEvent)
	void SetEnvironmentOptions(const TMap<FString, FString>& Options);

	UFUNCTION(BlueprintImplementableEvent)
	void SeedEnvironment(int Seed);
};
