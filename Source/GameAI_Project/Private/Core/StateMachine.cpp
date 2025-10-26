// Fill out your copyright notice in the Description page of Project Settings.


#include "Core/StateMachine.h"
#include "States/State.h"
#include "States/MoveToState.h"
#include "States/AttackState.h"
#include "States/FleeState.h"
#include "States/DeadState.h"
#include "AIController.h"
#include "BehaviorTree/BlackboardComponent.h"
#include "GameFramework/Character.h"


UStateMachine::UStateMachine()
{
    PrimaryComponentTick.bCanEverTick = true;
    Owner = GetOwner();
}

void UStateMachine::BeginPlay()
{
    Super::BeginPlay();
    InitStateMachine();
}

void UStateMachine::InitStateMachine()
{
    // ���� ��ü ����
    MoveToState     =   NewObject<UMoveToState>(this, UMoveToState::StaticClass());
    AttackState     =   NewObject<UAttackState>(this, UAttackState::StaticClass());
    FleeState       =   NewObject<UFleeState>(this, UFleeState::StaticClass());
    DeadState       =   NewObject<UDeadState>(this, UDeadState::StaticClass());

    // �ʱ� ���� ����
    CurrentState    =   MoveToState;


    // �ʱ� ���� ����
    /*
    if (CurrentState)
    {
        CurrentState->EnterState(this);
    }
    */
}

void UStateMachine::TickComponent(float DeltaTime, ELevelTick TickType, FActorComponentTickFunction* ThisTickFunction)
{
    //Super::TickComponent(DeltaTime, TickType, ThisTickFunction);
    
    /*
    // ���� ���� ������Ʈ
    if (CurrentState)
    {
        
        // Status updates every 2 seconds
        CurrentTime = GetWorld()->GetTimeSeconds();
        if (CurrentTime - LastStateUpdateTime > 2.0f)
        {
            LastStateUpdateTime = CurrentTime;
            CurrentState->UpdateState(this, DeltaTime);
        }
    }
    */
}

void UStateMachine::ChangeState(UState* NewState)
{
    if (CurrentState != NewState)
    {
        if (CurrentState)
        {
            CurrentState->  ExitState(this);
            CurrentState =  NewState;
            CurrentState->  EnterState(this);
        }
    }
}

void UStateMachine::GetObservation(float Health, float Distance, int32 Num)
{
	// Legacy function - updates both old and new observation structures
	this->AgentHealth = Health;
	this->DistanceToDestination = Distance;
	this->EnemiesNum = Num;

	// Sync with new observation structure
	CurrentObservation.Health = Health;
	CurrentObservation.DistanceToDestination = Distance;
	CurrentObservation.VisibleEnemyCount = Num;
}

void UStateMachine::UpdateObservation(const FObservationElement& NewObservation)
{
	CurrentObservation = NewObservation;

	// Sync legacy fields for backward compatibility with existing MCTS states
	AgentHealth = NewObservation.Health;
	DistanceToDestination = NewObservation.DistanceToDestination;
	EnemiesNum = NewObservation.VisibleEnemyCount;
}

void UStateMachine::UpdateAgentState(FVector Position, FVector Velocity, FRotator Rotation, float Health, float Stamina, float Shield)
{
	CurrentObservation.Position = Position;
	CurrentObservation.Velocity = Velocity;
	CurrentObservation.Rotation = Rotation;
	CurrentObservation.Health = Health;
	CurrentObservation.Stamina = Stamina;
	CurrentObservation.Shield = Shield;

	// Sync legacy field
	AgentHealth = Health;
}

void UStateMachine::UpdateCombatState(float WeaponCooldown, int32 Ammunition, int32 WeaponType)
{
	CurrentObservation.WeaponCooldown = WeaponCooldown;
	CurrentObservation.Ammunition = Ammunition;
	CurrentObservation.CurrentWeaponType = WeaponType;
}

void UStateMachine::UpdateEnemyInfo(int32 VisibleCount, const TArray<FEnemyObservation>& NearbyEnemies)
{
	CurrentObservation.VisibleEnemyCount = VisibleCount;
	CurrentObservation.NearbyEnemies = NearbyEnemies;

	// Ensure array has exactly 5 elements (pad with empty observations if needed)
	while (CurrentObservation.NearbyEnemies.Num() < 5)
	{
		CurrentObservation.NearbyEnemies.Add(FEnemyObservation());
	}
	if (CurrentObservation.NearbyEnemies.Num() > 5)
	{
		CurrentObservation.NearbyEnemies.SetNum(5);
	}

	// Sync legacy field
	EnemiesNum = VisibleCount;
}

void UStateMachine::UpdateTacticalContext(bool bHasCover, float CoverDistance, FVector2D CoverDirection, ETerrainType Terrain)
{
	CurrentObservation.bHasCover = bHasCover;
	CurrentObservation.NearestCoverDistance = CoverDistance;
	CurrentObservation.CoverDirection = CoverDirection;
	CurrentObservation.CurrentTerrain = Terrain;
}

void UStateMachine::TriggerBlueprintEvent(const FName& EventName)
{
    if (Owner)
    {
        UFunction* Function = Owner->FindFunction(EventName);
        if (Function)
        {
            Owner->ProcessEvent(Function, nullptr);
        }
    }
}

UState* UStateMachine::GetCurrentState()
{
    return CurrentState;
}

UState* UStateMachine::GetMoveToState()
{
    return MoveToState;
}

UState* UStateMachine::GetAttackState()
{
    return AttackState;
}

UState* UStateMachine::GetFleeState()
{
    return FleeState;
}

UState* UStateMachine::GetDeadState()
{
    return DeadState;
}

// ========================================
// BLACKBOARD INTEGRATION METHODS
// ========================================

AAIController* UStateMachine::GetAIController() const
{
	if (!OwnerPawn)
	{
		// Try to get owner pawn if not cached
		APawn* Pawn = Cast<APawn>(GetOwner());
		if (!Pawn)
		{
			return nullptr;
		}
	}

	APawn* Pawn = OwnerPawn ? OwnerPawn : Cast<APawn>(GetOwner());
	if (!Pawn)
	{
		return nullptr;
	}

	return Cast<AAIController>(Pawn->GetController());
}

UBlackboardComponent* UStateMachine::GetBlackboard() const
{
	AAIController* AIController = GetAIController();
	if (!AIController)
	{
		return nullptr;
	}

	return AIController->GetBlackboardComponent();
}

void UStateMachine::SetCurrentStrategy(const FString& Strategy)
{
	UBlackboardComponent* BlackboardComp = GetBlackboard();
	if (BlackboardComp)
	{
		BlackboardComp->SetValueAsString(FName("CurrentStrategy"), Strategy);
		UE_LOG(LogTemp, Log, TEXT("StateMachine: Set strategy to '%s'"), *Strategy);
	}
	else
	{
		UE_LOG(LogTemp, Warning, TEXT("StateMachine: Cannot set strategy - no Blackboard component available"));
	}
}

void UStateMachine::SetTargetEnemy(AActor* TargetEnemy)
{
	UBlackboardComponent* BlackboardComp = GetBlackboard();
	if (BlackboardComp)
	{
		BlackboardComp->SetValueAsObject(FName("TargetEnemy"), TargetEnemy);
		if (TargetEnemy)
		{
			UE_LOG(LogTemp, Log, TEXT("StateMachine: Set target enemy to '%s'"), *TargetEnemy->GetName());
		}
		else
		{
			UE_LOG(LogTemp, Log, TEXT("StateMachine: Cleared target enemy"));
		}
	}
	else
	{
		UE_LOG(LogTemp, Warning, TEXT("StateMachine: Cannot set target enemy - no Blackboard component available"));
	}
}

void UStateMachine::SetDestination(FVector Destination)
{
	UBlackboardComponent* BlackboardComp = GetBlackboard();
	if (BlackboardComp)
	{
		BlackboardComp->SetValueAsVector(FName("Destination"), Destination);
		UE_LOG(LogTemp, Log, TEXT("StateMachine: Set destination to %s"), *Destination.ToString());
	}
	else
	{
		UE_LOG(LogTemp, Warning, TEXT("StateMachine: Cannot set destination - no Blackboard component available"));
	}
}

void UStateMachine::SetCoverLocation(FVector CoverLocation)
{
	UBlackboardComponent* BlackboardComp = GetBlackboard();
	if (BlackboardComp)
	{
		BlackboardComp->SetValueAsVector(FName("CoverLocation"), CoverLocation);
		UE_LOG(LogTemp, Log, TEXT("StateMachine: Set cover location to %s"), *CoverLocation.ToString());
	}
	else
	{
		UE_LOG(LogTemp, Warning, TEXT("StateMachine: Cannot set cover location - no Blackboard component available"));
	}
}

void UStateMachine::SetThreatLevel(float ThreatLevel)
{
	UBlackboardComponent* BlackboardComp = GetBlackboard();
	if (BlackboardComp)
	{
		// Clamp threat level to [0, 1] range
		float ClampedThreat = FMath::Clamp(ThreatLevel, 0.0f, 1.0f);
		BlackboardComp->SetValueAsFloat(FName("ThreatLevel"), ClampedThreat);
		UE_LOG(LogTemp, Log, TEXT("StateMachine: Set threat level to %.2f"), ClampedThreat);
	}
	else
	{
		UE_LOG(LogTemp, Warning, TEXT("StateMachine: Cannot set threat level - no Blackboard component available"));
	}
}


