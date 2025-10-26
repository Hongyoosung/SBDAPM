// Fill out your copyright notice in the Description page of Project Settings.


#include "Actions/Movement/MoveLeftAction.h"
#include "Core/StateMachine.h"
#include "GameFramework/Character.h"
#include "TimerManager.h"
#include "GameFramework/CharacterMovementComponent.h"

void UMoveLeftAction::ExecuteAction(UStateMachine* StateMachine)
{
    UE_LOG(LogTemp, Warning, TEXT("------------MoveLeftAction"));

    //StateMachine->TriggerBlueprintEvent("MoveL");
    
    ACharacter* OwnerCharacter = Cast<ACharacter>(StateMachine->GetOwner());
    if (OwnerCharacter)
    {
        // world x vector
        FVector ForwardVector = OwnerCharacter->GetActorRightVector() * -1.0f; // Move left

        // �� �̵� �Ÿ��� �̵� �ӵ��� ����
        float TotalDistance = 100.0f; // 100cm (1m)
        float MoveDuration = 1.0f; // 1�� ���� �̵�
        float MoveSpeed = TotalDistance / MoveDuration; // �ʴ� �̵� �Ÿ�

        // �ʱ� �ð� ����
        ElapsedTime = 0.0f;

        MoveDelegate.BindLambda([OwnerCharacter, ForwardVector, MoveSpeed, this]()
            {
                // ��Ÿ �ð��� �����ɴϴ�.
                float DeltaTime = OwnerCharacter->GetWorld()->GetDeltaSeconds();

                // ��� �ð� ������Ʈ
                ElapsedTime += DeltaTime;

                if (ElapsedTime < 1.0f) // 1�� ���ȸ� �̵�
                {
                    OwnerCharacter->AddMovementInput(ForwardVector, MoveSpeed * DeltaTime);
                }
                else
                {
                    // Ÿ�̸� ���߱�
                    OwnerCharacter->GetWorld()->GetTimerManager().ClearTimer(MoveTimer);
                }
            });

        // Ÿ�̸� ����: ��Ÿ �ð� �������� MoveDelegate�� �ݺ� ȣ��
        if (OwnerCharacter->GetWorld())
        {
            OwnerCharacter->GetWorld()->GetTimerManager().SetTimer(MoveTimer, MoveDelegate, OwnerCharacter->GetWorld()->GetDeltaSeconds(), true);
        }
    }
}
