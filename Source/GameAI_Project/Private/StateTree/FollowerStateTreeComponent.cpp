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

	bStartLogicAutomatically = true;
}

void UFollowerStateTreeComponent::BeginPlay()
{
	UE_LOG(LogTemp, Warning, TEXT("🔵 UFollowerStateTreeComponent::BeginPlay CALLED for '%s'"),
		GetOwner() ? *GetOwner()->GetName() : TEXT("NULL_OWNER"));

	// Validate State Tree asset is set (required by base UStateTreeComponent)
	UStateTree* StateTree = const_cast<UStateTree*>(StateTreeRef.GetStateTree());
	if (!StateTree)
	{
		UE_LOG(LogTemp, Error, TEXT("UFollowerStateTreeComponent: ❌ State Tree asset not set on '%s'!"), *GetOwner()->GetName());
		Super::BeginPlay(); // Still call Super even on error
		return;
	}

	// CRITICAL FIX: Validate schema compatibility BEFORE calling Super::BeginPlay
	const UStateTreeSchema* Schema = StateTree->GetSchema();
	if (!Schema)
	{
		UE_LOG(LogTemp, Error, TEXT("UFollowerStateTreeComponent: ❌ StateTree has no schema!"));
		Super::BeginPlay();
		return;
	}

	// Check schema CLASS compatibility (ignore instance name like "_0")
	if (!Schema->GetClass()->IsChildOf(UFollowerStateTreeSchema::StaticClass()))
	{
		UE_LOG(LogTemp, Error, TEXT("UFollowerStateTreeComponent: ❌ Schema mismatch! Expected FollowerStateTreeSchema, got %s"),
			*Schema->GetClass()->GetName());
		Super::BeginPlay();
		return;
	}

	// UE 5.6 BUG FIX: StateTree loses compilation on restart
	// Force recompilation if not ready OR if previously failed
	bool bNeedsRecompile = !StateTree->IsReadyToRun();

#if WITH_EDITOR
	if (bNeedsRecompile)
	{
		UE_LOG(LogTemp, Warning, TEXT("⚠️ StateTree not ready - forcing FULL recompilation..."));

		// Invalidate ALL cached compilation data
		StateTree->LastCompiledEditorDataHash = 0;

		// Force a full compile (bypass CompileIfChanged optimization)
		if (UE::StateTree::Delegates::OnRequestCompile.IsBound())
		{
			UE::StateTree::Delegates::OnRequestCompile.Execute(*StateTree);

			// Mark package dirty to force save (this is the KEY to persistence)
			StateTree->MarkPackageDirty();

			UE_LOG(LogTemp, Warning, TEXT("✅ Forced StateTree recompilation and marked dirty: %s"), *StateTree->GetName());
			UE_LOG(LogTemp, Warning, TEXT("🔧 To fix permanently: Open asset in editor, compile, and SAVE"));
		}
		else
		{
			UE_LOG(LogTemp, Error, TEXT("❌ Cannot recompile StateTree - delegates not bound!"));
			UE_LOG(LogTemp, Error, TEXT("🔧 WORKAROUND: Manually open '%s' in editor and click Compile"), *StateTree->GetName());
		}
	}
#else
	if (bNeedsRecompile)
	{
		UE_LOG(LogTemp, Error, TEXT("StateTree not ready in packaged build! Must fix in editor."));
	}
#endif


	UE_LOG(LogTemp, Warning, TEXT("UFollowerStateTreeComponent: ✅ State Tree asset found: '%s'"), *StateTree->GetName());

	// Find FollowerAgentComponent BEFORE calling Super::BeginPlay
	if (!FollowerComponent && bAutoFindFollowerComponent)
	{
		FollowerComponent = FindFollowerComponent();
		UE_LOG(LogTemp, Warning, TEXT("UFollowerStateTreeComponent: Auto-find FollowerComponent = %s"),
			FollowerComponent ? TEXT("✅ Found") : TEXT("❌ Not Found"));
	}

	if (!FollowerComponent)
	{
		UE_LOG(LogTemp, Error, TEXT("UFollowerStateTreeComponent: ❌ FollowerComponent not found on '%s'!"), *GetOwner()->GetName());
		Super::BeginPlay(); // Still call Super even on error
		return;
	}

	// Initialize context BEFORE Super::BeginPlay (which may call SetContextRequirements/CollectExternalData)
	UE_LOG(LogTemp, Warning, TEXT("UFollowerStateTreeComponent: Initializing context..."));
	InitializeContext();

	// Bind to follower events
	UE_LOG(LogTemp, Warning, TEXT("UFollowerStateTreeComponent: Binding to follower events..."));
	BindToFollowerEvents();

	// NOW call base class initialization (StateTree will use already-initialized context)
	UE_LOG(LogTemp, Warning, TEXT("UFollowerStateTreeComponent: Calling Super::BeginPlay()..."));
	Super::BeginPlay();
	UE_LOG(LogTemp, Warning, TEXT("UFollowerStateTreeComponent: ✅ Super::BeginPlay() completed"));

	EStateTreeRunStatus Status = GetStateTreeRunStatus();
	UE_LOG(LogTemp, Warning, TEXT("UFollowerStateTreeComponent: Status after BeginPlay = %s"),
		*UEnum::GetValueAsString(Status));

	if (Status != EStateTreeRunStatus::Running)
	{
		UE_LOG(LogTemp, Error, TEXT("UFollowerStateTreeComponent: ❌❌ STATETREE FAILED TO START! Status = %s"),
			*UEnum::GetValueAsString(Status));
		UE_LOG(LogTemp, Error, TEXT("Possible causes:"));
		UE_LOG(LogTemp, Error, TEXT("  1. StateTree asset '%s' has compilation errors (open in editor and check for errors)"), *StateTree->GetName());
		UE_LOG(LogTemp, Error, TEXT("  2. Root state is missing or invalid"));
		UE_LOG(LogTemp, Error, TEXT("  3. Context bindings are incorrect (Check CollectExternalData)"));
		UE_LOG(LogTemp, Error, TEXT("  4. Required external data not provided"));
	}
	else
	{
		UE_LOG(LogTemp, Warning, TEXT("UFollowerStateTreeComponent: ✅ StateTree successfully started and running!"));
	}


	UE_LOG(LogTemp, Warning, TEXT("UFollowerStateTreeComponent: ✅✅✅ BeginPlay COMPLETE for '%s'"), *GetOwner()->GetName());
}

void UFollowerStateTreeComponent::TickComponent(float DeltaTime, ELevelTick TickType, FActorComponentTickFunction* ThisTickFunction)
{
	// Log FIRST before anything else
	static int32 TickCount = 0;
	if (TickCount++ % 60 == 0) // Log every 60 ticks (~1 second at 60fps)
	{
		UE_LOG(LogTemp, Warning, TEXT("🔄 UFollowerStateTreeComponent::TickComponent for '%s' (Tick #%d)"),
			*GetOwner()->GetName(), TickCount);
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

	// Set custom context data BEFORE calling parent's SetContextRequirements
	// FStateTreeExecutionContext::SetContextDataByName is exported (UE_API)

	// Set FollowerContext struct
	FStateTreeDataView ContextView(
		FFollowerStateTreeContext::StaticStruct(),
		reinterpret_cast<uint8*>(&Context)
	);
	if (InContext.SetContextDataByName(FName(TEXT("FollowerContext")), ContextView))
	{
		//UE_LOG(LogTemp, Log, TEXT("  ✅ FollowerContext set via SetContextDataByName"));
	}
	else if (bLogErrors)
	{
		UE_LOG(LogTemp, Error, TEXT("  ❌ Failed to set FollowerContext"));
	}

	// Set FollowerComponent
	if (InContext.SetContextDataByName(FName(TEXT("FollowerComponent")), FStateTreeDataView(FollowerComponent)))
	{
		//UE_LOG(LogTemp, Log, TEXT("  ✅ FollowerComponent set: %s"), FollowerComponent ? TEXT("Valid") : TEXT("NULL"));
	}
	else if (bLogErrors)
	{
		UE_LOG(LogTemp, Error, TEXT("  ❌ Failed to set FollowerComponent"));
	}

	// Set TeamLeader (optional)
	UTeamLeaderComponent* TeamLeader = Context.TeamLeader;
	if (InContext.SetContextDataByName(FName(TEXT("TeamLeader")), FStateTreeDataView(TeamLeader)))
	{
		//UE_LOG(LogTemp, Log, TEXT("  ✅ TeamLeader set: %s"), TeamLeader ? TEXT("Valid") : TEXT("NULL (Optional)"));
	}

	// Set TacticalPolicy (optional)
	URLPolicyNetwork* TacticalPolicy = Context.TacticalPolicy;
	if (InContext.SetContextDataByName(FName(TEXT("TacticalPolicy")), FStateTreeDataView(TacticalPolicy)))
	{
		//UE_LOG(LogTemp, Log, TEXT("  ✅ TacticalPolicy set: %s"), TacticalPolicy ? TEXT("Valid") : TEXT("NULL (Optional)"));
	}

	// Now call parent to handle base data (AIController, Pawn, Actor) and validation
	const bool bResult = UStateTreeComponentSchema::SetContextRequirements(*this, InContext, bLogErrors);

	if (!bResult && bLogErrors)
	{
		UE_LOG(LogTemp, Error, TEXT("🔧 UStateTreeComponentSchema::SetContextRequirements FAILED."));
		UE_LOG(LogTemp, Error, TEXT("   Check that AIController and Pawn are valid."));
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
