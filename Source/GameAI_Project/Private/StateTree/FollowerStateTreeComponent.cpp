// Copyright Epic Games, Inc. All Rights Reserved.

#include "StateTree/FollowerStateTreeComponent.h"
#include "StateTree/FollowerStateTreeSchema.h"
#include "Team/FollowerAgentComponent.h"
#include "Team/TeamLeaderComponent.h"
#include "RL/RLPolicyNetwork.h"
#include "StateTreeExecutionContext.h"
#include "AIController.h"
#include "GameFramework/Pawn.h"
#include "GameFramework/Character.h"
#include "GameFramework/CharacterMovementComponent.h"
#include "StateTreeModule\Public\StateTree.h"
#include "Team/Objective.h"
#include "GameplayTagsManager.h"
#include "StateTreeEvents.h"

#if WITH_EDITOR
#include "StateTreeDelegates.h"
#endif

// Define StateTree event tags
const FGameplayTag UFollowerStateTreeComponent::Event_ObjectiveReceived =
	FGameplayTag::RequestGameplayTag(FName("StateTree.Follower.ObjectiveReceived"));
const FGameplayTag UFollowerStateTreeComponent::Event_FollowerDied =
	FGameplayTag::RequestGameplayTag(FName("StateTree.Follower.Died"));
const FGameplayTag UFollowerStateTreeComponent::Event_FollowerRespawned =
	FGameplayTag::RequestGameplayTag(FName("StateTree.Follower.Respawned"));

UFollowerStateTreeComponent::UFollowerStateTreeComponent()
	: Super()
	, FollowerComponent(nullptr)
	, bAutoFindFollowerComponent(true)
	, TickLogCounter(0)
{
	PrimaryComponentTick.bCanEverTick = true;
	PrimaryComponentTick.TickGroup = TG_PrePhysics;
	bAutoActivate = true;

	bStartLogicAutomatically = false;
}

void UFollowerStateTreeComponent::BeginPlay()
{
	// [변경] 부모 클래스 초기화를 가장 먼저 실행하여 안전성 확보
	Super::BeginPlay();

	UE_LOG(LogTemp, Warning, TEXT("🔵 UFollowerStateTreeComponent::BeginPlay CALLED for '%s'"),
		GetOwner() ? *GetOwner()->GetName() : TEXT("NULL_OWNER"));

	// [중요] 상태 변경 델리게이트 바인딩 (종료 원인 파악용)
	// UStateTreeComponent에 정의된 OnStateTreeRunStatusChanged 델리게이트 사용
	OnStateTreeRunStatusChanged.AddDynamic(this, &UFollowerStateTreeComponent::HandleOnStateTreeRunStatusChanged);

	// ... (FollowerComponent 찾기) ...
	if (!FollowerComponent && bAutoFindFollowerComponent)
	{
		FollowerComponent = FindFollowerComponent();
	}

	if (!FollowerComponent)
	{
		UE_LOG(LogTemp, Error, TEXT("UFollowerStateTreeComponent: ❌ FollowerComponent not found!"));
		return;
	}

	// Initialize context 
	InitializeContext();

	// Bind to follower events
	BindToFollowerEvents();

	// [핵심] 초기화가 끝난 후 마지막에 시작 시도
	if (CheckRequirementsAndStart())
	{
		UE_LOG(LogTemp, Warning, TEXT("UFollowerStateTreeComponent: StateTree started immediately in BeginPlay!"));
	}
	else
	{
		UE_LOG(LogTemp, Warning, TEXT("UFollowerStateTreeComponent: StateTree waiting for AIController..."));
	}

	// 상태 확인 로그
	EStateTreeRunStatus Status = GetStateTreeRunStatus();
	if (Status == EStateTreeRunStatus::Running)
	{
		UE_LOG(LogTemp, Warning, TEXT("UFollowerStateTreeComponent: ✅ StateTree successfully started and running!"));
	}
	else
	{
		UE_LOG(LogTemp, Warning, TEXT("UFollowerStateTreeComponent: ❌ StateTree not running after BeginPlay. Status=%s"),
			*UEnum::GetValueAsString(Status));
	}
}
void UFollowerStateTreeComponent::TickComponent(float DeltaTime, ELevelTick TickType, FActorComponentTickFunction* ThisTickFunction)
{
	
	// [수정 1] 멤버 변수를 사용하여 개별 에이전트 로그 출력 (60프레임마다)
	if (TickLogCounter++ % 60 == 0)
	{
		EStateTreeRunStatus Status = GetStateTreeRunStatus();
		UE_LOG(LogTemp, Log, TEXT("UFollowerStateTreeComponent: 🔄 [Tick] '%s' | Status: %s"),
			*GetNameSafe(GetOwner()), *UEnum::GetValueAsString(Status));
	}

	Super::TickComponent(DeltaTime, TickType, ThisTickFunction);

	// Deferred initialization if BeginPlay failed to find FollowerComponent
	if (!FollowerComponent && bAutoFindFollowerComponent)
	{
		FollowerComponent = FindFollowerComponent();
		if (FollowerComponent)
		{
			UE_LOG(LogTemp, Log, TEXT("UFollowerStateTreeComponent: FollowerComponent found on deferred initialization for '%s'"), *GetOwner()->GetName());
			InitializeContext();
			BindToFollowerEvents();
		}
		else
		{
			// Only log error once per 2 seconds to avoid spam
			static float LastErrorTime = 0.0f;
			if (GetWorld()->GetTimeSeconds() - LastErrorTime > 2.0f)
			{
				UE_LOG(LogTemp, Error, TEXT("UFollowerStateTreeComponent: FollowerComponent still not found on '%s'. Ensure UFollowerAgentComponent is added to the same actor!"), *GetOwner()->GetName());
				LastErrorTime = GetWorld()->GetTimeSeconds();
			}
		}
		return; // Skip update until initialized
	}

	// Update context from follower component every tick
	if (FollowerComponent)
	{
		UpdateContextFromFollower();
	}

	EStateTreeRunStatus CurrentStatus = GetStateTreeRunStatus();
	if (CurrentStatus != EStateTreeRunStatus::Running)
	{
		// Schola 학습 중에는 컨트롤러가 늦게 붙을 수 있으므로 
		// 컨트롤러가 유효해질 때까지 재시도를 반복하는 것은 괜찮으나,
		// 종료 이유(Succeeded/Failed)를 확인해야 함.

		// 1초에 한 번만 재시작 시도 로그 출력 (스팸 방지)
		if (TickLogCounter % 60 == 0)
		{
			// CheckRequirementsAndStart 내부 로그가 이미 있으므로 여기선 생략 가능
			CheckRequirementsAndStart();
		}
		else
		{
			// 매 틱마다 시도는 하되 로그는 남기지 않음 (반응성 유지)
			CheckRequirementsAndStart();
		}
	}
}

void UFollowerStateTreeComponent::EndPlay(const EEndPlayReason::Type EndPlayReason)
{
	Super::EndPlay(EndPlayReason);
}

TSubclassOf<UStateTreeSchema> UFollowerStateTreeComponent::GetSchema() const
{
	return UFollowerStateTreeSchema::StaticClass();
}

bool UFollowerStateTreeComponent::SetContextRequirements(FStateTreeExecutionContext& InContext, bool bLogErrors)
{
	UE_LOG(LogTemp, Warning, TEXT("🔵 UFollowerStateTreeComponent::SetContextRequirements START for '%s'"),
		GetOwner() ? *GetOwner()->GetName() : TEXT("NULL"));

	InContext.SetLinkedStateTreeOverrides(LinkedStateTreeOverrides);
	InContext.SetCollectExternalDataCallback(FOnCollectStateTreeExternalData::CreateUObject(
		this, &UFollowerStateTreeComponent::CollectExternalData));

	// (A) Follower Context
	FStateTreeDataView ContextView(
		FFollowerStateTreeContext::StaticStruct(),
		reinterpret_cast<uint8*>(&Context)
	);
	if (!InContext.SetContextDataByName(FName(TEXT("FollowerContext")), ContextView))
	{
		if (bLogErrors) UE_LOG(LogTemp, Error, TEXT("  ❌ Failed to set FollowerContext"));
	}
	else
	{
		UE_LOG(LogTemp, Log, TEXT("  ✅ FollowerContext set"));
	}

	// (B) Follower Component
	if (!InContext.SetContextDataByName(FName(TEXT("FollowerComponent")), FStateTreeDataView(FollowerComponent)))
	{
		if (bLogErrors) UE_LOG(LogTemp, Error, TEXT("  ❌ Failed to set FollowerComponent"));
	}
	else
	{
		UE_LOG(LogTemp, Log, TEXT("  ✅ FollowerComponent set: %s"), FollowerComponent ? *FollowerComponent->GetName() : TEXT("NULL"));
	}

	// (C) Follower State Tree Component
	if (!InContext.SetContextDataByName(FName(TEXT("FollowerStateTreeComponent")), FStateTreeDataView(this)))
	{
		if (bLogErrors) UE_LOG(LogTemp, Error, TEXT("  ❌ Failed to set FollowerStateTreeComponent"));
	}
	else
	{
		UE_LOG(LogTemp, Log, TEXT("  ✅ FollowerStateTreeComponent (self) set"));
	}

	// (D) Team Leader (Optional)
	if (Context.TeamLeader)
	{
		InContext.SetContextDataByName(FName(TEXT("TeamLeader")), FStateTreeDataView(Context.TeamLeader));
		UE_LOG(LogTemp, Log, TEXT("  ✅ TeamLeader set: %s"), *Context.TeamLeader->GetName());
	}
	else
	{
		UE_LOG(LogTemp, Log, TEXT("  ⚠️ TeamLeader is NULL (optional)"));
	}

	// (E) Tactical Policy (Optional)
	if (Context.TacticalPolicy)
	{
		InContext.SetContextDataByName(FName(TEXT("TacticalPolicy")), FStateTreeDataView(Context.TacticalPolicy));
		UE_LOG(LogTemp, Log, TEXT("  ✅ TacticalPolicy set: %s"), *Context.TacticalPolicy->GetName());
	}
	else
	{
		UE_LOG(LogTemp, Log, TEXT("  ⚠️ TacticalPolicy is NULL (optional)"));
	}

	// Use our custom schema's SetContextRequirements which makes AIController optional for Schola
	UE_LOG(LogTemp, Warning, TEXT("  🔄 Calling Schema SetContextRequirements..."));
	const bool bResult = UFollowerStateTreeSchema::SetContextRequirements(*this, InContext, true);

	if (!bResult)
	{
		UE_LOG(LogTemp, Error, TEXT("🔵 UFollowerStateTreeComponent::SetContextRequirements FAILED"));
	}
	else
	{
		UE_LOG(LogTemp, Warning, TEXT("🔵 UFollowerStateTreeComponent::SetContextRequirements SUCCESS"));
	}

	return bResult;
}

TValueOrError<void, FString> UFollowerStateTreeComponent::HasValidStateTreeReference() const
{
	if (!StateTreeRef.IsValid())
	{
		return MakeError(TEXT("The State Tree asset is not set."));
	}

	const UStateTree* StateTree = StateTreeRef.GetStateTree();
	if (!StateTree)
	{
		return MakeError(TEXT("The State Tree reference is invalid."));
	}

	const UStateTreeSchema* Schema = StateTree->GetSchema();
	if (!Schema)
	{
		return MakeError(TEXT("The State Tree schema is not set."));
	}

	if (!Schema->GetClass()->IsChildOf(UFollowerStateTreeSchema::StaticClass()))
	{
		return MakeError(FString::Printf(
			TEXT("The State Tree schema is not compatible. Expected FollowerStateTreeSchema or child class, but got %s."),
			*Schema->GetClass()->GetName()
		));
	}

	return MakeValue();

}

void UFollowerStateTreeComponent::ValidateStateTreeReference()
{
	const TValueOrError<void, FString> Result = HasValidStateTreeReference();
	if (Result.HasError())
	{
		UE_LOG(LogTemp, Error, TEXT("UFollowerStateTreeComponent::ValidateStateTreeReference: %s Cannot initialize."),
			*Result.GetError());
	}
}

void UFollowerStateTreeComponent::InitializeContext()
{
	if (!FollowerComponent)
	{
		UE_LOG(LogTemp, Error, TEXT("UFollowerStateTreeComponent::InitializeContext: FollowerComponent not found on '%s'!"),
			GetOwner() ? *GetOwner()->GetName() : TEXT("Unknown"));
		return;
	}

	// Set context component reference
	Context.FollowerComponent = FollowerComponent;

	// Auto-find Pawn and AIController
	if (APawn* OwnerPawn = Cast<APawn>(GetOwner()))
	{
		Context.ControlledPawn = OwnerPawn;

		if (!Context.AIController)
		{
			Context.AIController = Cast<AAIController>(OwnerPawn->GetController());
		}
	}

	// Set component references
	if (!Context.TeamLeader && Context.FollowerComponent)
	{
		Context.TeamLeader = Context.FollowerComponent->GetTeamLeader();
	}

	if (!Context.TacticalPolicy && Context.FollowerComponent)
	{
		Context.TacticalPolicy = Context.FollowerComponent->GetTacticalPolicy();
	}

	// Initialize state flags
	Context.bIsAlive = FollowerComponent->bIsAlive;
	Context.bUseRLPolicy = FollowerComponent->bUseRLPolicy;

	// Initialize objective (v3.0)
	Context.CurrentObjective = FollowerComponent->GetCurrentObjective();
	Context.bHasActiveObjective = FollowerComponent->HasActiveObjective();

	// Initialize observation
	Context.CurrentObservation = FollowerComponent->GetLocalObservation();

	UE_LOG(LogTemp, Log, TEXT("UFollowerStateTreeComponent: Initialized context for '%s' - ControlledPawn: %s, AIController: %s"),
		*GetOwner()->GetName(),
		Context.ControlledPawn ? *Context.ControlledPawn->GetName() : TEXT("NULL"),
		Context.AIController ? *Context.AIController->GetName() : TEXT("NULL"));
}

void UFollowerStateTreeComponent::UpdateContextFromFollower()
{
	if (!FollowerComponent)
	{
		return;
	}

	// Sync basic state from follower component (v3.0)
	// (Detailed observation updates are handled by STEvaluator_UpdateObservation)
	Context.bIsAlive = FollowerComponent->bIsAlive;
	Context.CurrentObjective = FollowerComponent->GetCurrentObjective();
	Context.bHasActiveObjective = FollowerComponent->HasActiveObjective();
	Context.AccumulatedReward = FollowerComponent->GetAccumulatedReward();
}

bool UFollowerStateTreeComponent::IsStateTreeRunning() const
{
	return GetStateTreeRunStatus() == EStateTreeRunStatus::Running;
}

FString UFollowerStateTreeComponent::GetCurrentStateName() const
{
	// Get run status
	EStateTreeRunStatus Status = GetStateTreeRunStatus();

	switch (Status)
	{
		case EStateTreeRunStatus::Running:
			return TEXT("Running");
		case EStateTreeRunStatus::Succeeded:
			return TEXT("Succeeded");
		case EStateTreeRunStatus::Failed:
			return TEXT("Failed");
		case EStateTreeRunStatus::Stopped:
			return TEXT("Stopped");
		case EStateTreeRunStatus::Unset:
		default:
			return TEXT("Not Running");
	}

	// Note: In UE5.6, accessing individual state names requires accessing the execution context
	// during callbacks (EnterState, Tick, etc.). Direct access from component is not supported.
	// For detailed state information, use debug visualization or StateTree logging.
}

bool UFollowerStateTreeComponent::CollectExternalData(const FStateTreeExecutionContext& InContext,
	const UStateTree* StateTree, TArrayView<const FStateTreeExternalDataDesc> ExternalDataDescs,
	TArrayView<FStateTreeDataView> OutDataViews) const
{
	UE_LOG(LogTemp, Error, TEXT("🔥🔥🔥 CollectExternalData CALLED 🔥🔥🔥"));
    UE_LOG(LogTemp, Warning, TEXT("🔍 Collecting external data for %d descriptors"), 
        ExternalDataDescs.Num());
    
    // Get owner references (const-safe)
    const APawn* OwnerPawn = Cast<APawn>(GetOwner());
    const AAIController* AIController = OwnerPawn ? Cast<AAIController>(OwnerPawn->GetController()) : nullptr;

    int32 RequiredCount = 0;
    int32 ProvidedCount = 0;

    for (int32 Index = 0; Index < ExternalDataDescs.Num(); Index++)
    {
        const FStateTreeExternalDataDesc& Desc = ExternalDataDescs[Index];
        
        if (Desc.Requirement == EStateTreeExternalDataRequirement::Required)
        {
            RequiredCount++;
        }

        bool bProvided = false;

        // Handle base class descriptors
        if (Desc.Struct && Desc.Struct->IsChildOf(AAIController::StaticClass()))
        {
            if (AIController)
            {
                OutDataViews[Index] = FStateTreeDataView(const_cast<AAIController*>(AIController));
                bProvided = true;
                UE_LOG(LogTemp, Log, TEXT("  ✅ [%d] AIController provided"), Index);
            }
            else
            {
                UE_LOG(LogTemp, Error, TEXT("  ❌ [%d] AIController REQUIRED but NULL"), Index);
            }
        }
        else if (Desc.Struct && Desc.Struct->IsChildOf(APawn::StaticClass()))
        {
            if (OwnerPawn)
            {
                OutDataViews[Index] = FStateTreeDataView(const_cast<APawn*>(OwnerPawn));
                bProvided = true;
                UE_LOG(LogTemp, Log, TEXT("  ✅ [%d] Pawn provided"), Index);
            }
            else
            {
                UE_LOG(LogTemp, Error, TEXT("  ❌ [%d] Pawn REQUIRED but NULL"), Index);
            }
        }
        else if (Desc.Struct && Desc.Struct->IsChildOf(UStateTreeComponent::StaticClass()))
        {
            // Provide this component itself
            OutDataViews[Index] = FStateTreeDataView(const_cast<UFollowerStateTreeComponent*>(this));
            bProvided = true;
            UE_LOG(LogTemp, Log, TEXT("  ✅ [%d] StateTreeComponent (self) provided"), Index);
        }
        else if (Desc.Name == FName(TEXT("FollowerComponent")))
        {
            if (FollowerComponent)
            {
                OutDataViews[Index] = FStateTreeDataView(FollowerComponent);
                bProvided = true;
                UE_LOG(LogTemp, Log, TEXT("  ✅ [%d] FollowerComponent provided"), Index);
            }
            else
            {
                UE_LOG(LogTemp, Error, TEXT("  ❌ [%d] FollowerComponent REQUIRED but NULL"), Index);
            }
        }
        else if (Desc.Name == FName(TEXT("FollowerContext")))
        {
            // Provide the context struct - const_cast is safe here as StateTree needs mutable access
            OutDataViews[Index] = FStateTreeDataView(
                FFollowerStateTreeContext::StaticStruct(),
                reinterpret_cast<uint8*>(const_cast<FFollowerStateTreeContext*>(&Context))
            );
            bProvided = true;
            UE_LOG(LogTemp, Log, TEXT("  ✅ [%d] FollowerContext struct provided"), Index);
        }
        else if (Desc.Name == FName(TEXT("TeamLeader")))
        {
            // Access from the cached context member (if available)
            UTeamLeaderComponent* TeamLeader = Context.TeamLeader; // This won't work!
            
            // Better: Store as component member
            UTeamLeaderComponent* CachedTeamLeader = nullptr;
            if (FollowerComponent)
            {
                CachedTeamLeader = FollowerComponent->GetTeamLeader();
            }
            
            OutDataViews[Index] = FStateTreeDataView(CachedTeamLeader);
            bProvided = true;
            UE_LOG(LogTemp, Log, TEXT("  ✅ [%d] TeamLeader: %s"), Index,
                CachedTeamLeader ? TEXT("Valid") : TEXT("NULL (Optional)"));
        }
        else if (Desc.Name == FName(TEXT("TacticalPolicy")))
        {
            // Same issue - need to cache this separately
            URLPolicyNetwork* CachedPolicy = nullptr;
            if (FollowerComponent)
            {
                CachedPolicy = FollowerComponent->GetTacticalPolicy(); // Implement this getter
            }
            
            OutDataViews[Index] = FStateTreeDataView(CachedPolicy);
            bProvided = true;
            UE_LOG(LogTemp, Log, TEXT("  ✅ [%d] TacticalPolicy: %s"), Index,
                CachedPolicy ? TEXT("Valid") : TEXT("NULL (Optional)"));
        }

        if (!bProvided)
        {
            OutDataViews[Index] = FStateTreeDataView();
            
            if (Desc.Requirement == EStateTreeExternalDataRequirement::Required)
            {
                UE_LOG(LogTemp, Error, TEXT("  ❌ [%d] REQUIRED data missing: '%s' (Type: %s)"), 
                    Index, *Desc.Name.ToString(), 
                    Desc.Struct ? *Desc.Struct->GetName() : TEXT("NULL"));
                return false;
            }
            else
            {
                UE_LOG(LogTemp, Warning, TEXT("  ⚠️ [%d] Optional data not provided: '%s'"), 
                    Index, *Desc.Name.ToString());
            }
        }
        else if (Desc.Requirement == EStateTreeExternalDataRequirement::Required)
        {
            ProvidedCount++;
        }
    }

    UE_LOG(LogTemp, Warning, TEXT("🔍 CollectExternalData COMPLETE - %d/%d required items provided"), 
        ProvidedCount, RequiredCount);
    
    return ProvidedCount >= RequiredCount;
}



UFollowerAgentComponent* UFollowerStateTreeComponent::FindFollowerComponent()
{
	AActor* Owner = GetOwner();
	if (!Owner)
	{
		return nullptr;
	}

	UFollowerAgentComponent* OwnerFollowerComp = Owner->FindComponentByClass<UFollowerAgentComponent>();

	if (!OwnerFollowerComp)
	{
		UE_LOG(LogTemp, Warning, TEXT("UFollowerStateTreeComponent: No FollowerAgentComponent found on '%s'"), *Owner->GetName());
	}

	return OwnerFollowerComp;
}

void UFollowerStateTreeComponent::BindToFollowerEvents()
{
	if (!FollowerComponent)
	{
		return;
	}

	// Bind to objective received event (v3.0)
	FollowerComponent->OnObjectiveReceived.AddDynamic(this, &UFollowerStateTreeComponent::OnObjectiveReceived);

	UE_LOG(LogTemp, Log, TEXT("UFollowerStateTreeComponent: Bound to FollowerAgentComponent events"));
}

void UFollowerStateTreeComponent::OnObjectiveReceived(UObjective* Objective, EFollowerState NewState)
{
	// Update context immediately when objective changes (v3.0)
	Context.CurrentObjective = Objective;
	Context.bHasActiveObjective = Objective != nullptr && Objective->IsActive();

	// CRITICAL: Immediately set primary target from objective (don't wait for evaluator tick)
	if (Objective && Objective->TargetActor && Objective->TargetActor->IsValidLowLevel() && !Objective->TargetActor->IsPendingKillPending())
	{
		Context.PrimaryTarget = Objective->TargetActor;

		// Update distance if we have a pawn
		if (APawn* OwnerPawn = Cast<APawn>(GetOwner()))
		{
			Context.DistanceToPrimaryTarget = FVector::Dist(
				OwnerPawn->GetActorLocation(),
				Context.PrimaryTarget->GetActorLocation()
			);
		}
	}
	else
	{
		Context.PrimaryTarget = nullptr;
		Context.DistanceToPrimaryTarget = 0.0f;
	}

	FString ObjectiveStr = Objective ? UEnum::GetValueAsString(Objective->Type) : TEXT("None");
	UE_LOG(LogTemp, Log, TEXT("UFollowerStateTreeComponent: Objective received - Type: %s, State: %s"),
		*ObjectiveStr,
		*UEnum::GetValueAsString(NewState));

	// Send StateTree event for event-driven transition
	SendStateTreeEvent(Event_ObjectiveReceived);
}

void UFollowerStateTreeComponent::OnFollowerDied()
{
	Context.bIsAlive = false;

	UE_LOG(LogTemp, Log, TEXT("UFollowerStateTreeComponent: Follower died, transitioning to Dead state"));

	// Send StateTree event for event-driven transition
	SendStateTreeEvent(Event_FollowerDied);
}

void UFollowerStateTreeComponent::OnFollowerRespawned()
{
	Context.bIsAlive = true;
	AActor* Owner = GetOwner();

	UE_LOG(LogTemp, Warning, TEXT("🔄 Follower '%s' Respawned: Restarting StateTree"), *GetNameSafe(Owner));

	if (IsStateTreeRunning())
	{
		StopLogic("Respawn");
		UE_LOG(LogTemp, Log, TEXT("  → StateTree stopped to trigger ExitState"));
	}

	GetWorld()->GetTimerManager().SetTimerForNextTick([WeakThis = TWeakObjectPtr<UFollowerStateTreeComponent>(this)]()
		{
			UFollowerStateTreeComponent* Comp = WeakThis.Get();
			if (!Comp) return;

			Comp->StartLogic();
			UE_LOG(LogTemp, Warning, TEXT("✅ StateTree restarted for '%s'"), *GetNameSafe(Comp->GetOwner()));

			// Send StateTree event for event-driven transition
			Comp->SendStateTreeEvent(Event_FollowerRespawned);

			if (ACharacter* Character = Cast<ACharacter>(Comp->GetOwner()))
			{
				UCharacterMovementComponent* MoveComp = Character->GetCharacterMovement();
				if (MoveComp && !MoveComp->IsActive())
				{
					MoveComp->Activate();
					UE_LOG(LogTemp, Warning, TEXT("  → Force-activated CharacterMovementComponent"));
				}
			}
		});
}

bool UFollowerStateTreeComponent::CheckRequirementsAndStart()
{
	AActor* Owner = GetOwner();
	FString OwnerName = Owner ? Owner->GetName() : TEXT("NULL_OWNER");

	// 이미 실행 중이면 패스
	if (IsStateTreeRunning())
	{
		UE_LOG(LogTemp, Warning, TEXT("  UFollowerStateTreeComponent:✅ StateTree already running for '%s'"), *OwnerName);
		return true;
	}

	// 1. 필수 컴포넌트 확인
	if (!FollowerComponent)
	{
		UE_LOG(LogTemp, Warning, TEXT("  UFollowerStateTreeComponent:⏳ FollowerComponent = NULL for '%s'"), *OwnerName);
		return false;
	}

	// 2. AIController 확인 (AAbstractTrainer now inherits from AAIController)
	if (!Context.AIController)
	{
		APawn* OwnerPawn = Cast<APawn>(Owner);
		if (OwnerPawn)
		{
			Context.AIController = Cast<AAIController>(OwnerPawn->GetController());
			if (!Context.AIController)
			{
				UE_LOG(LogTemp, Warning, TEXT("  UFollowerStateTreeComponent:⏳ No AIController yet"));
			}
		}
		else
		{
			UE_LOG(LogTemp, Warning, TEXT("  UFollowerStateTreeComponent:❌ Owner is NOT a Pawn (it's %s)"),
				Owner ? *Owner->GetClass()->GetName() : TEXT("NULL"));
		}
	}


	// 3. 모든 조건 만족 시 시작
	UE_LOG(LogTemp, Warning, TEXT("  UFollowerStateTreeComponent:🚀 All requirements met! Calling StartLogic()..."));
	StartLogic();


	return IsStateTreeRunning();
}

void UFollowerStateTreeComponent::HandleOnStateTreeRunStatusChanged(const EStateTreeRunStatus Status)
{
	FString StatusStr = UEnum::GetValueAsString(Status);
	UE_LOG(LogTemp, Warning, TEXT("UFollowerStateTreeComponent: ⚠️ [StatusChanged] '%s' changed to: %s"),
		*GetNameSafe(GetOwner()), *StatusStr);

	if (Status == EStateTreeRunStatus::Failed)
	{
		UE_LOG(LogTemp, Error, TEXT("UFollowerStateTreeComponent: ❌ StateTree Failed! Check the active State/Task requirements."));
	}
	else if (Status == EStateTreeRunStatus::Succeeded)
	{
		UE_LOG(LogTemp, Warning, TEXT("UFollowerStateTreeComponent: ✅ StateTree Succeeded (Finished). Logic stopped."));
	}
}

void UFollowerStateTreeComponent::SendStateTreeEvent(const FGameplayTag& EventTag, FConstStructView Payload)
{
	if (!IsStateTreeRunning())
	{
		UE_LOG(LogTemp, Warning, TEXT("UFollowerStateTreeComponent: Cannot send event '%s' - StateTree not running"),
			*EventTag.ToString());
		return;
	}

	FStateTreeEvent Event(EventTag, Payload, FName(TEXT("FollowerComponent")));

	// 부모 클래스(UStateTreeComponent)의 SendStateTreeEvent 호출
	Super::SendStateTreeEvent(Event);

	UE_LOG(LogTemp, Log, TEXT("UFollowerStateTreeComponent: Event sent - '%s'"), *EventTag.ToString());
}
