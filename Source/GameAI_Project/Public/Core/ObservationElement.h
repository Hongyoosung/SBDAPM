// Fill out your copyright notice in the Description page of Project Settings.

#pragma once

#include "CoreMinimal.h"
#include "UObject/NoExportTypes.h"
#include "ObservationElement.generated.h"

USTRUCT(BlueprintType)
struct FObservationElement
{
    GENERATED_BODY()

    // 목적지까지의 거리
    float DistanceToDestination;

    // 에이전트의 체력
    float AgentHealth;

    // 에이전트 주변 적의 수
    int32 EnemiesNum;
};
