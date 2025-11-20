#pragma once

#include "CoreMinimal.h"
#include "Kismet/BlueprintFunctionLibrary.h"
#include "GameAIHelper.generated.h"

/**
 * AI 및 StateTree 로직에서 공통으로 사용하는 헬퍼 함수 모음
 */
UCLASS()
class GAMEAI_PROJECT_API UGameAIHelper : public UBlueprintFunctionLibrary
{
	GENERATED_BODY()

public:
	// 타겟이 유효하고 살아있는지 확인 (BlueprintCallable을 붙이면 블루프린트에서도 사용 가능)
	UFUNCTION(BlueprintCallable, Category = "Game AI")
	static bool IsTargetValid(AActor* Target);

	// 보이는 적들 중 가장 가까운 적 찾기
	UFUNCTION(BlueprintCallable, Category = "Game AI")
	static AActor* FindNearestValidEnemy(const TArray<AActor*>& VisibleEnemies, APawn* FromPawn);
};