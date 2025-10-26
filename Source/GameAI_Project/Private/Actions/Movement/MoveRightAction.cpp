// Fill out your copyright notice in the Description page of Project Settings.


#include "Actions/MoveRightAction.h"
#include "Core/StateMachine.h"
#include "GameFramework/Character.h"
#include "TimerManager.h"
#include "GameFramework/CharacterMovementComponent.h"

void UMoveRightAction::ExecuteAction(UStateMachine* StateMachine)
{
    UE_LOG(LogTemp, Warning, TEXT("------------MoveRightAction"));

    //StateMachine->TriggerBlueprintEvent("MoveR");
    
    ACharacter* OwnerCharacter = Cast<ACharacter>(StateMachine->GetOwner());
    if (OwnerCharacter)
    {
        // world x vector
        FVector ForwardVector = OwnerCharacter->GetActorRightVector(); // Move right

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
