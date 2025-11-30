// TacticalActuator.cpp - Schola actuator implementation

#include "Schola/TacticalActuator.h"
#include "Team/FollowerAgentComponent.h"
#include "StateTree/FollowerStateTreeComponent.h"
#include "StateTree/FollowerStateTreeContext.h"
#include "GameFramework/Pawn.h"
#include "Inference/InferenceComponent.h"

UTacticalActuator::UTacticalActuator()
{
	LastAction = FTacticalAction();
}

FBoxSpace UTacticalActuator::GetActionSpace()
{
	TArray<float> LowBounds = { -1.0f, -1.0f, 0.0f, -1.0f, -1.0f, 0.0f, 0.0f, 0.0f };
	TArray<float> HighBounds = { 1.0f,  1.0f, 1.0f,  1.0f,  1.0f, 1.0f, 1.0f, 1.0f };
	TArray<int> Shape = { 8 };

	FBoxSpace ActionSpace = FBoxSpace(LowBounds, HighBounds, Shape);

	// Debug: Verify shape is correctly set
	UE_LOG(LogTemp, Warning, TEXT("[TacticalActuator] GetActionSpace(): Dimensions=%d, Shape.Num()=%d"),
		ActionSpace.Dimensions.Num(), ActionSpace.Shape.Num());
	if (ActionSpace.Shape.Num() > 0)
	{
		UE_LOG(LogTemp, Warning, TEXT("[TacticalActuator] Shape[0]=%d"), ActionSpace.Shape[0]);
	}

	return ActionSpace;
}

void UTacticalActuator::TakeAction(const FBoxPoint& Action)
{
	if (!FollowerAgent)
	{
		UE_LOG(LogTemp, Warning, TEXT("[TacticalActuator] %s: FollowerAgent not found!"),
			*GetNameSafe(GetOuter()));
		return;
	}

	// Find state tree component
	UFollowerStateTreeComponent* StateTreeComp = FindStateTreeComponent();
	if (!StateTreeComp)
	{
		UE_LOG(LogTemp, Warning, TEXT("[TacticalActuator] %s: StateTreeComponent not found!"),
			*GetNameSafe(GetOuter()));
		return;
	}

	// Validate action dimensions
	if (Action.Values.Num() < 8)
	{
		UE_LOG(LogTemp, Error, TEXT("[TacticalActuator] %s: Invalid action dimensions (expected 8, got %d)"),
			*GetNameSafe(GetOuter()), Action.Values.Num());
		return;
	}

	// Parse 8-dimensional action vector
	FTacticalAction ParsedAction;

	// [0-1]: move_direction
	ParsedAction.MoveDirection = FVector2D(Action.Values[0], Action.Values[1]);

	// [2]: move_speed
	ParsedAction.MoveSpeed = Action.Values[2];

	// [3-4]: look_direction
	ParsedAction.LookDirection = FVector2D(Action.Values[3], Action.Values[4]);

	// [5]: fire (interpret as binary: >= 0.5 = true)
	ParsedAction.bFire = (Action.Values[5] >= 0.5f);

	// [6]: crouch
	ParsedAction.bCrouch = (Action.Values[6] >= 0.5f);

	// [7]: use_ability
	ParsedAction.bUseAbility = (Action.Values[7] >= 0.5f);

	// Store action in shared context for StateTree execution
	FFollowerStateTreeContext& SharedContext = StateTreeComp->GetSharedContext();
	SharedContext.CurrentAtomicAction = ParsedAction;
	SharedContext.bScholaActionReceived = true; // Flag that action came from Schola

	LastAction = ParsedAction;

	// Debug logging (ALWAYS log to diagnose integration)
	AActor* Owner = GetTypedOuter<AActor>();
	UE_LOG(LogTemp, Warning, TEXT("🎮 [SCHOLA ACTUATOR] '%s': Received action from Python → Move=(%.2f,%.2f) Speed=%.2f, Flag=TRUE"),
		*GetNameSafe(Owner),
		ParsedAction.MoveDirection.X, ParsedAction.MoveDirection.Y, ParsedAction.MoveSpeed);

	UE_LOG(LogTemp, Warning, TEXT("    → SharedContext.bScholaActionReceived = %d (should be TRUE)"),
		SharedContext.bScholaActionReceived ? 1 : 0);
}

void UTacticalActuator::InitializeActuator()
{
	// Auto-find follower agent if enabled
	if (bAutoFindFollower && !FollowerAgent)
	{
		FollowerAgent = FindFollowerAgent();
	}

	if (!FollowerAgent)
	{
		UE_LOG(LogTemp, Warning, TEXT("[TacticalActuator] %s: Failed to find FollowerAgentComponent!"),
			*GetNameSafe(GetOuter()));
		return;
	}

	UE_LOG(LogTemp, Log, TEXT("[TacticalActuator] %s: Initialized (Follower=%s, ActionSpace=8D Box)"),
		*GetNameSafe(GetOuter()), *GetNameSafe(FollowerAgent));
}

UFollowerAgentComponent* UTacticalActuator::FindFollowerAgent() const
{
	AActor* Owner = GetTypedOuter<AActor>();
	if (!Owner)
	{
		return nullptr;
	}

	return Owner->FindComponentByClass<UFollowerAgentComponent>();
}

UFollowerStateTreeComponent* UTacticalActuator::FindStateTreeComponent() const
{
	AActor* Owner = GetTypedOuter<AActor>();
	if (!Owner)
	{
		return nullptr;
	}

	return Owner->FindComponentByClass<UFollowerStateTreeComponent>();
}
