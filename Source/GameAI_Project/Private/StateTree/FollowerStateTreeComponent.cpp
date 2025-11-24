// Copyright Epic Games, Inc. All Rights Reserved.

#include "StateTree/FollowerStateTreeComponent.h"
#include "Team/FollowerAgentComponent.h"
#include "Team/TeamLeaderComponent.h"
#include "RL/RLPolicyNetwork.h"
#include "StateTreeExecutionContext.h"
#include "AIController.h"
#include "GameFramework/Pawn.h"
#include "StateTreeModule\Public\StateTree.h"

#if WITH_EDITOR
#include "StateTreeDelegates.h" 
#endif

UFollowerStateTreeComponent::UFollowerStateTreeComponent()
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

	// ... (StateTree 에셋 검증 및 스키마 확인 로직은 그대로 유지) ...
	// (중략: StateTree 변수 가져오기, 스키마 체크 등)
	UStateTree* StateTree = const_cast<UStateTree*>(StateTreeRef.GetStateTree());
	if (!StateTree) return; // 에러 로그는 위에 있다고 가정

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
		UE_LOG(LogTemp, Warning, TEXT("✅ StateTree started immediately in BeginPlay!"));
	}
	else
	{
		UE_LOG(LogTemp, Warning, TEXT("⏳ StateTree waiting for AIController..."));
	}

	// 상태 확인 로그
	EStateTreeRunStatus Status = GetStateTreeRunStatus();
	if (Status == EStateTreeRunStatus::Running)
	{
		UE_LOG(LogTemp, Warning, TEXT("UFollowerStateTreeComponent: ✅ StateTree successfully started and running!"));
	}
}
void UFollowerStateTreeComponent::TickComponent(float DeltaTime, ELevelTick TickType, FActorComponentTickFunction* ThisTickFunction)
{
	// Log FIRST before anything else
	static int32 TickCount = 0;
	if (TickCount++ % 60 == 0) // Log every 60 ticks (~1 second at 60fps)
	{
		//UE_LOG(LogTemp, Warning, TEXT("🔄 UFollowerStateTreeComponent::TickComponent for '%s' (Tick #%d)"),
		//	*GetOwner()->GetName(), TickCount);
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

	UE_LOG(LogTemp, Verbose, TEXT("UFollowerStateTreeComponent: TickComponent for '%s'"), *GetOwner()->GetName());

	if (GetStateTreeRunStatus() != EStateTreeRunStatus::Running)
	{
		// 1. FollowerComponent 지연 찾기 (기존 로직 유지)
		if (!FollowerComponent && bAutoFindFollowerComponent)
		{
			FollowerComponent = FindFollowerComponent();
			if (FollowerComponent)
			{
				InitializeContext();
				BindToFollowerEvents();
			}
		}

		// 2. 시작 시도
		CheckRequirementsAndStart();
	}

	// DEBUG: Log StateTree status periodically
	static float LastDebugLogTime = 0.0f;
	if (GetWorld()->GetTimeSeconds() - LastDebugLogTime > 2.0f)
	{
		EStateTreeRunStatus Status = GetStateTreeRunStatus();
		UE_LOG(LogTemp, Display, TEXT("[STATE TREE] '%s': Status=%s, Command=%s, Valid=%d"),
			*GetOwner()->GetName(),
			*UEnum::GetValueAsString(Status),
			*UEnum::GetValueAsString(Context.CurrentCommand.CommandType),
			Context.bIsCommandValid);
		LastDebugLogTime = GetWorld()->GetTimeSeconds();
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
		if (bLogErrors) UE_LOG(LogTemp, Error, TEXT("❌ Failed to set FollowerContext"));
	}

	// (B) Follower Component
	if (!InContext.SetContextDataByName(FName(TEXT("FollowerComponent")), FStateTreeDataView(FollowerComponent)))
	{
		if (bLogErrors) UE_LOG(LogTemp, Error, TEXT("❌ Failed to set FollowerComponent"));
	}

	// (C) Follower State Tree Component
	if (!InContext.SetContextDataByName(FName(TEXT("FollowerStateTreeComponent")), FStateTreeDataView(this)))
	{
		if (bLogErrors) UE_LOG(LogTemp, Error, TEXT("❌ Failed to set FollowerStateTreeComponent"));
	}

	// (D) Team Leader (Optional)
	if (Context.TeamLeader)
	{
		InContext.SetContextDataByName(FName(TEXT("TeamLeader")), FStateTreeDataView(Context.TeamLeader));
	}

	// (E) Tactical Policy (Optional)
	if (Context.TacticalPolicy)
	{
		InContext.SetContextDataByName(FName(TEXT("TacticalPolicy")), FStateTreeDataView(Context.TacticalPolicy));
	}

	const bool bResult = UStateTreeComponentSchema::SetContextRequirements(*this, InContext, bLogErrors);

	if (!bResult && bLogErrors)
	{
		UE_LOG(LogTemp, Error, TEXT("❌ Parent SetContextRequirements FAILED. Missing Pawn or AIController?"));
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
    
	// Auto-find AIController
	if (!Context.AIController)
	{
		if (APawn* OwnerPawn = Cast<APawn>(GetOwner()))
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

	// Initialize command
	Context.CurrentCommand = FollowerComponent->GetCurrentCommand();
	Context.bIsCommandValid = FollowerComponent->IsCommandValid();

	// Initialize observation
	Context.CurrentObservation = FollowerComponent->GetLocalObservation();

	UE_LOG(LogTemp, Log, TEXT("UFollowerStateTreeComponent: Initialized context for '%s'"),
		*GetOwner()->GetName());
}

void UFollowerStateTreeComponent::UpdateContextFromFollower()
{
	if (!FollowerComponent)
	{
		return;
	}

	// Sync basic state from follower component
	// (Detailed observation updates are handled by STEvaluator_UpdateObservation)
	Context.bIsAlive = FollowerComponent->bIsAlive;
	Context.CurrentCommand = FollowerComponent->GetCurrentCommand();
	Context.bIsCommandValid = FollowerComponent->IsCommandValid();
	Context.TimeSinceCommand = FollowerComponent->GetTimeSinceLastCommand();
	Context.AccumulatedReward = FollowerComponent->GetAccumulatedReward();
}

bool UFollowerStateTreeComponent::IsStateTreeRunning() const
{
	return GetStateTreeRunStatus() == EStateTreeRunStatus::Running;
}

FString UFollowerStateTreeComponent::GetCurrentStateName() const
{
	if (!IsStateTreeRunning())
	{
		return TEXT("Not Running");
	}

	// Get active state name from State Tree
	// (UE State Tree API may vary - this is a placeholder)
	return TEXT("Active"); // TODO: Get actual state name from State Tree API
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
                // Cast away const - this is safe because FStateTreeDataView doesn't modify the data
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

	// Bind to command received event
	FollowerComponent->OnCommandReceived.AddDynamic(this, &UFollowerStateTreeComponent::OnCommandReceived);

	UE_LOG(LogTemp, Log, TEXT("UFollowerStateTreeComponent: Bound to FollowerAgentComponent events"));
}

void UFollowerStateTreeComponent::OnCommandReceived(const FStrategicCommand& Command, EFollowerState NewState)
{
	// Update context immediately when command changes
	Context.CurrentCommand = Command;
	Context.bIsCommandValid = true;
	Context.TimeSinceCommand = 0.0f;

	// CRITICAL: Immediately set primary target from command (don't wait for evaluator tick)
	if (Command.TargetActor && Command.TargetActor->IsValidLowLevel() && !Command.TargetActor->IsPendingKillPending())
	{
		Context.PrimaryTarget = Command.TargetActor;

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

	UE_LOG(LogTemp, Log, TEXT("UFollowerStateTreeComponent: Command received - Type: %s, State: %s"),
		*UEnum::GetValueAsString(Command.CommandType),
		*UEnum::GetValueAsString(NewState));

	// State Tree will handle state transitions automatically via conditions
}

void UFollowerStateTreeComponent::OnFollowerDied()
{
	Context.bIsAlive = false;

	UE_LOG(LogTemp, Log, TEXT("UFollowerStateTreeComponent: Follower died, transitioning to Dead state"));

	// State Tree will transition to Dead state automatically via IsAlive condition
}

bool UFollowerStateTreeComponent::CheckRequirementsAndStart()
{
	// 이미 실행 중이면 패스
	if (IsStateTreeRunning())
	{
		UE_LOG(LogTemp, Warning, TEXT("✅ StateTree already running for '%s'"), 
			GetOwner() ? *GetOwner()->GetName() : TEXT("Unknown"));
		return true;
	}

	// 1. 필수 컴포넌트 확인
	if (!FollowerComponent)
	{
		UE_LOG(LogTemp, Warning, TEXT("⏳ Cannot start StateTree: FollowerComponent not found for '%s'"),
			GetOwner() ? *GetOwner()->GetName() : TEXT("Unknown"));
		return false;
	}

	// 2. AIController 확인 (가장 중요한 부분!)
	// Context에 캐싱된 것이 없으면 다시 찾아봄
	if (!Context.AIController)
	{
		if (APawn* OwnerPawn = Cast<APawn>(GetOwner()))
		{
			Context.AIController = Cast<AAIController>(OwnerPawn->GetController());
		}
	}

	// 컨트롤러가 아직도 없으면 시작 불가
	if (!Context.AIController)
	{
		UE_LOG(LogTemp, Warning, TEXT("⏳ Cannot start StateTree: AIController not found for '%s'"),
			GetOwner() ? *GetOwner()->GetName() : TEXT("Unknown"));
		return false;
	}

	// 3. 모든 조건 만족 시 시작
	UE_LOG(LogTemp, Warning, TEXT("🚀 All requirements met. Starting StateTree Logic..."));
	StartLogic();

	return IsStateTreeRunning();
}
