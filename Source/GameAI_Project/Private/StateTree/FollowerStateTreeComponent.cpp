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

	if (StateTree && !StateTree->IsReadyToRun())
	{
#if WITH_EDITOR
		// 강제로 컴파일 해시를 무효화
		StateTree->LastCompiledEditorDataHash = 0;

		// 재컴파일 시도
		if (UE::StateTree::Delegates::OnRequestCompile.IsBound())
		{
			StateTree->CompileIfChanged();
			UE_LOG(LogTemp, Warning, TEXT("Forced StateTree recompilation: %s"), *StateTree->GetName());
		}
		else
		{
			UE_LOG(LogTemp, Error, TEXT("Cannot recompile StateTree - delegates not bound yet!"));
		}
#else
		UE_LOG(LogTemp, Error, TEXT("StateTree not ready in packaged build! Must fix in editor."));
#endif
	}


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

	// Auto-start State Tree if enabled
	UE_LOG(LogTemp, Warning, TEXT("UFollowerStateTreeComponent: bAutoStartStateTree = %s"), bAutoStartStateTree ? TEXT("true") : TEXT("false"));
	if (bAutoStartStateTree)
	{
		UE_LOG(LogTemp, Warning, TEXT("UFollowerStateTreeComponent: Calling StartLogic()..."));

		// Check if StateTree is valid and compiled
		if (!StateTree->IsReadyToRun())
		{
			UE_LOG(LogTemp, Error, TEXT("UFollowerStateTreeComponent: ❌ StateTree '%s' is NOT ready to run! Check for compilation errors in the asset."),
				*StateTree->GetName());
		}


		EStateTreeRunStatus Status = GetStateTreeRunStatus();
		UE_LOG(LogTemp, Warning, TEXT("UFollowerStateTreeComponent: StartLogic() returned - Status = %s"),
			*UEnum::GetValueAsString(Status));

		if (Status != EStateTreeRunStatus::Running)
		{
			UE_LOG(LogTemp, Error, TEXT("UFollowerStateTreeComponent: ❌❌ STATETREE FAILED TO START! Status = %s"),
				*UEnum::GetValueAsString(Status));
			UE_LOG(LogTemp, Error, TEXT("Possible causes:"));
			UE_LOG(LogTemp, Error, TEXT("  1. StateTree asset '%s' has compilation errors (open in editor and check for errors)"), *StateTree->GetName());
			UE_LOG(LogTemp, Error, TEXT("  2. Root state is missing or invalid"));
			UE_LOG(LogTemp, Error, TEXT("  3. Context bindings are incorrect"));
			UE_LOG(LogTemp, Error, TEXT("  4. Required external data not provided"));
		}
		else
		{
			UE_LOG(LogTemp, Warning, TEXT("UFollowerStateTreeComponent: ✅ StateTree successfully started and running!"));
		}
	}
	else
	{
		UE_LOG(LogTemp, Warning, TEXT("UFollowerStateTreeComponent: ⚠️ Auto-start disabled, StateTree NOT started"));
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
	if (!Super::SetContextRequirements(InContext, bLogErrors))
	{
		return false;
	}

	return true;
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
	TArrayView<FStateTreeDataView> OutDataViews)
{
	for (int32 Index = 0; Index < ExternalDataDescs.Num(); Index++)
	{
		const FStateTreeExternalDataDesc& Desc = ExternalDataDescs[Index];

		// Provide the FollowerContext struct
		if (Desc.Name == FName(TEXT("FollowerContext")))
		{
			OutDataViews[Index] = FStateTreeDataView(FFollowerStateTreeContext::StaticStruct(), reinterpret_cast<uint8*>(&Context));
		}
		// Provide FollowerAgentComponent
		else if (Desc.Name == FName(TEXT("FollowerComponent")))
		{
			OutDataViews[Index] = FStateTreeDataView(FollowerComponent);
		}
		// Provide TeamLeaderComponent
		else if (Desc.Name == FName(TEXT("TeamLeader")))
		{
			OutDataViews[Index] = FStateTreeDataView(Context.TeamLeader);
		}
		// Provide TacticalPolicy
		else if (Desc.Name == FName(TEXT("TacticalPolicy")))
		{
			OutDataViews[Index] = FStateTreeDataView(Context.TacticalPolicy);
		}
		else
		{
			OutDataViews[Index] = FStateTreeDataView();
		}
	}

	return true;
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
