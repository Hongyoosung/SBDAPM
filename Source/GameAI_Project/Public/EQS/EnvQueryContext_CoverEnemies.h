#pragma once

#include "CoreMinimal.h"
#include "EnvironmentQuery/EnvQueryContext.h"
#include "EnvQueryContext_CoverEnemies.generated.h"

/**
 * EQS Context that provides enemy locations for cover evaluation
 * Used by cover queries to determine optimal cover positions away from enemies
 */
UCLASS()
class GAMEAI_PROJECT_API UEnvQueryContext_CoverEnemies : public UEnvQueryContext
{
	GENERATED_BODY()

public:
	virtual void ProvideContext(FEnvQueryInstance& QueryInstance, FEnvQueryContextData& ContextData) const override;
};
