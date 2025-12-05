# StateTree Schola Compatibility Issue - Diagnostic Report

## Problem Summary
StateTree fails to start when using Schola training mode, remaining in `EStateTreeRunStatus::Unset` state despite all requirements appearing to be met.

## Root Cause Analysis

### Issue 1: Missing Pawn Context Descriptor
**Location:** `FollowerStateTreeSchema.cpp:22-74` (Constructor)

The schema constructor adds custom context descriptors (Actor, FollowerContext, FollowerComponent, etc.) but **does NOT** add a Pawn descriptor. However, `SetContextRequirements()` at line 100 attempts to set Pawn context, which fails validation because Pawn is not registered as an expected descriptor.

**Current Code:**
```cpp
// Constructor adds: Actor, FollowerContext, FollowerComponent, etc.
// But MISSING: Pawn descriptor

// SetContextRequirements tries to set Pawn (line 100)
if (!Context.SetContextDataByName(TEXT("Pawn"), FStateTreeDataView(OwnerPawn)))
{
    // This likely FAILS because Pawn is not in ContextDataDescs!
}
```

### Issue 2: Missing Parent Schema Initialization
**Location:** `FollowerStateTreeSchema.cpp:76-159`

The static `SetContextRequirements()` method does not call parent class's implementation:

```cpp
bool UFollowerStateTreeSchema::SetContextRequirements(...)
{
    // Missing: Super::SetContextRequirements() call
    // This means parent class setup is never executed!
}
```

UE5's StateTree architecture requires calling parent schema initialization to:
- Register base context requirements
- Set up execution framework
- Validate schema-level constraints

### Issue 3: Schola Controller Validation
**Location:** `FollowerStateTreeComponent.cpp:669-717`

When Schola is active, the controller is `AAbstractTrainer` (inherits `AController` but NOT `AAIController`). The current logic attempts to handle this but may not be propagating the "allow null AIController" intent correctly through the validation chain.

## Evidence from Logs

```
UFollowerStateTreeComponent:🔍 CheckRequirementsAndStart() CALLED for 'BP_FollowerAgent_C_8'
UFollowerStateTreeComponent:✅ FollowerComponent found
UFollowerStateTreeComponent:🚀 All requirements met! Calling StartLogic()...
UFollowerStateTreeComponent: ❌ StateTree is now STILL NOT RUNNING
UFollowerStateTreeComponent: ❌ StateTree not running after BeginPlay. Status=EStateTreeRunStatus::Unset
```

The status `Unset` indicates `StartLogic()` is being called but internal validation is failing silently.

## Impact

- StateTree cannot execute in Schola training mode
- Followers remain idle despite receiving objectives
- Real-time PPO training cannot proceed
- Multi-agent coordination broken

## Proposed Fix

### Fix 1: Add Pawn Descriptor to Schema Constructor
Add Pawn as a required context descriptor before custom descriptors:

```cpp
UFollowerStateTreeSchema::UFollowerStateTreeSchema()
{
    AIControllerClass = AFollowerAIController::StaticClass();
    PawnClass = AFollowerCharacter::StaticClass();

    ContextDataDescs.Reset();

    // 1. BASE CONTEXT (Parent schema requirements)
    // Pawn (REQUIRED for all StateTree operations)
    {
        FStateTreeExternalDataDesc PawnDesc(
            FName("Pawn"),
            APawn::StaticClass(),
            FGuid(0x2E11DB00, 0xC4084FDB, 0xB164E824, 0x347C7BB6)
        );
        PawnDesc.Requirement = EStateTreeExternalDataRequirement::Required;
        ContextDataDescs.Add(PawnDesc);
    }

    // AIController (OPTIONAL for Schola compatibility)
    {
        FStateTreeExternalDataDesc AIDesc(
            FName("AIController"),
            AAIController::StaticClass(),
            FGuid(0x1D291B00, 0x29994FDE, 0xC6546702, 0x47895FD6)
        );
        AIDesc.Requirement = EStateTreeExternalDataRequirement::Optional;
        ContextDataDescs.Add(AIDesc);
    }

    // Actor (base context)
    {
        FStateTreeExternalDataDesc ActorDesc(
            FName("Actor"),
            AActor::StaticClass(),
            FGuid(0x1D971B00, 0x28884FDE, 0xB5436802, 0x36984FD5)
        );
        ActorDesc.Requirement = EStateTreeExternalDataRequirement::Required;
        ContextDataDescs.Add(ActorDesc);
    }

    // 2. CUSTOM FOLLOWER CONTEXT
    // ... (rest of existing custom descriptors)
}
```

### Fix 2: Call Parent SetContextRequirements
Ensure parent class initialization runs:

```cpp
bool UFollowerStateTreeSchema::SetContextRequirements(UStateTreeComponent& InComponent, FStateTreeExecutionContext& Context, bool bLogErrors)
{
    // CRITICAL: Call parent implementation first
    if (!Super::SetContextRequirements(InComponent, Context, bLogErrors))
    {
        if (bLogErrors)
        {
            UE_LOG(LogTemp, Error, TEXT("FollowerStateTreeSchema: Parent SetContextRequirements failed"));
        }
        // Continue anyway - we'll handle Pawn/AIController manually for Schola
    }

    // Get owner actor and pawn
    AActor* Owner = InComponent.GetOwner();
    if (!Owner)
    {
        if (bLogErrors)
        {
            UE_LOG(LogTemp, Error, TEXT("FollowerStateTreeSchema: Owner is null"));
        }
        return false;
    }

    APawn* OwnerPawn = Cast<APawn>(Owner);
    if (!OwnerPawn)
    {
        if (bLogErrors)
        {
            UE_LOG(LogTemp, Error, TEXT("FollowerStateTreeSchema: Owner '%s' is not a Pawn"), *Owner->GetName());
        }
        return false;
    }

    // REQUIRED: Pawn (always needed) - overwrite what parent set
    if (!Context.SetContextDataByName(TEXT("Pawn"), FStateTreeDataView(OwnerPawn)))
    {
        if (bLogErrors)
        {
            UE_LOG(LogTemp, Error, TEXT("FollowerStateTreeSchema: Failed to set Pawn context for '%s'"), *Owner->GetName());
        }
        return false;
    }

    // OPTIONAL: AIController (not required for Schola compatibility)
    AController* Controller = OwnerPawn->GetController();
    AAIController* AIController = Cast<AAIController>(Controller);

    if (AIController)
    {
        Context.SetContextDataByName(TEXT("AIController"), FStateTreeDataView(AIController));
    }
    else if (Controller)
    {
        // Schola training mode - controller exists but is not AAIController
        UE_LOG(LogTemp, Log, TEXT("FollowerStateTreeSchema: '%s' has non-AI controller (%s). This is OK for Schola training."),
            *Owner->GetName(),
            *Controller->GetClass()->GetName());
        Context.SetContextDataByName(TEXT("AIController"), FStateTreeDataView(static_cast<AAIController*>(nullptr)));
    }
    else
    {
        // No controller at all
        if (bLogErrors)
        {
            UE_LOG(LogTemp, Warning, TEXT("FollowerStateTreeSchema: No controller for '%s'"), *Owner->GetName());
        }
        Context.SetContextDataByName(TEXT("AIController"), FStateTreeDataView(static_cast<AAIController*>(nullptr)));
    }

    // REQUIRED: Actor (base context) - already set by parent or here
    if (!Context.SetContextDataByName(TEXT("Actor"), FStateTreeDataView(Owner)))
    {
        if (bLogErrors)
        {
            UE_LOG(LogTemp, Error, TEXT("FollowerStateTreeSchema: Failed to set Actor context for '%s'"), *Owner->GetName());
        }
        return false;
    }

    return true;
}
```

### Fix 3: Enhanced Logging
Add comprehensive logging to `UFollowerStateTreeComponent::SetContextRequirements()` to diagnose validation failures:

```cpp
bool UFollowerStateTreeComponent::SetContextRequirements(FStateTreeExecutionContext& InContext, bool bLogErrors)
{
    UE_LOG(LogTemp, Warning, TEXT("🔵 UFollowerStateTreeComponent::SetContextRequirements START"));

    InContext.SetLinkedStateTreeOverrides(LinkedStateTreeOverrides);
    InContext.SetCollectExternalDataCallback(FOnCollectStateTreeExternalData::CreateUObject(
        this, &UFollowerStateTreeComponent::CollectExternalData));

    // Set custom context data...
    // (existing code)

    // Use our custom schema's SetContextRequirements
    const bool bResult = UFollowerStateTreeSchema::SetContextRequirements(*this, InContext, true);

    UE_LOG(LogTemp, Warning, TEXT("🔵 Schema SetContextRequirements result: %s"), bResult ? TEXT("SUCCESS") : TEXT("FAILED"));

    if (!bResult)
    {
        UE_LOG(LogTemp, Error, TEXT("UFollowerStateTreeComponent:❌ SetContextRequirements FAILED"));
    }

    return bResult;
}
```

## Testing Plan

1. **Verify Descriptor Registration**: Add logging in constructor to confirm Pawn/AIController descriptors are added
2. **Test Normal AI Mode**: Ensure existing AIController-based agents still work
3. **Test Schola Mode**: Verify StateTree starts with AAbstractTrainer controller
4. **Monitor Status Transitions**: Log StateTree status changes from Unset → Running

## Related Files

- `Source/GameAI_Project/Public/StateTree/FollowerStateTreeSchema.h:43`
- `Source/GameAI_Project/Private/StateTree/FollowerStateTreeSchema.cpp:22-74,76-159`
- `Source/GameAI_Project/Private/StateTree/FollowerStateTreeComponent.cpp:186-235,645-729`
