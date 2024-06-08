#include "MoveForwardAction.h"
#include "StateMachine.h"
#include "GameFramework/Character.h"
#include "TimerManager.h"
#include "GameFramework/CharacterMovementComponent.h"

void UMoveForwardAction::ExecuteAction(UStateMachine* StateMachine)
{
    UE_LOG(LogTemp, Warning, TEXT("MoveForwardAction"));

    ACharacter* OwnerCharacter = Cast<ACharacter>(StateMachine->GetOwner());
    if (OwnerCharacter)
    {
        FVector ForwardVector = OwnerCharacter->GetActorForwardVector();
        float TotalDistance = 100.0f; // 1m = 100cm
        float TimeToMove = 1.0f; // 1초 동안 이동
        float DistancePerTick = TotalDistance / (TimeToMove / GetWorld()->GetDeltaSeconds());

        ElapsedTime = 0.0f; // ElapsedTime 초기화

        MoveDelegate.BindLambda([OwnerCharacter, ForwardVector, DistancePerTick, TimeToMove, this]()
            {
                UWorld* World = GetWorld();
                if (World)
                {
                    ElapsedTime += World->GetDeltaSeconds();

                    if (ElapsedTime < TimeToMove)
                    {
                        OwnerCharacter->AddMovementInput(ForwardVector, DistancePerTick);
                    }
                    else
                    {
                        World->GetTimerManager().ClearTimer(MoveTimer);
                    }
                }
            });

        UWorld* World = GetWorld();
        if (World)
        {
            World->GetTimerManager().SetTimer(MoveTimer, MoveDelegate, World->GetDeltaSeconds(), true);
        }
    }
}
