// Fill out your copyright notice in the Description page of Project Settings.

#pragma once

#include "CoreMinimal.h"
#include "UObject/NoExportTypes.h"
#include "Actions/Action.h"
#include "Observation/ObservationElement.h"
#include "MCTSNode.generated.h"

UCLASS()
class GAMEAI_PROJECT_API UMCTSNode : public UObject
{
    GENERATED_BODY()

public:
    UMCTSNode();

    void InitializeNode(UMCTSNode* InParent, UAction* InAction);

    UPROPERTY()
    UMCTSNode* Parent;

    UPROPERTY()
    TArray<UMCTSNode*> Children;

    UPROPERTY()
    UAction* Action;

    UPROPERTY()
    int32 VisitCount;

    UPROPERTY()
    float TotalReward;

    UPROPERTY()
    float LastVisitTime;

    FObservationElement Observation;

    // ���� ����� ���¸� ǥ���ϴ� �Լ� (���� ������ ������Ʈ�� ���� ǥ�� ��Ŀ� ���� �޶��� �� �ֽ��ϴ�)
    virtual FString GetState() const;
};
