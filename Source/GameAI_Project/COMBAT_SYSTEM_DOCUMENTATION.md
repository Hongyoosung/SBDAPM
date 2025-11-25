# Combat System Documentation

**Version:** 2.0
**Engine:** Unreal Engine 5.6
**Language:** C++17
**Last Updated:** 2025-11-26

---

## Table of Contents

1. [System Overview](#system-overview)
2. [Health Component](#health-component)
3. [Weapon Component](#weapon-component)
4. [AI Integration](#ai-integration)
5. [Reward System Integration](#reward-system-integration)
6. [StateTree Integration](#statetree-integration)
7. [Configuration](#configuration)
8. [Troubleshooting](#troubleshooting)

---

## System Overview

### Design Philosophy

The combat system is designed for **AI-driven tactical combat** with seamless integration into the hierarchical MCTS+RL AI system. Key principles:

- **Modularity**: Health and Weapon are separate components
- **Event-Driven**: Broadcasts events for AI reward calculation
- **Predictive**: Weapon uses predictive aiming for moving targets
- **Configurable**: Extensive exposed properties for balancing

### System Architecture

```
FollowerAgentComponent (AI Controller)
    ↓
StateTree (Execution Framework)
    ├─ ExecuteAssault Task → WeaponComponent::Fire()
    ├─ ExecuteDefend Task → WeaponComponent::SuppressiveFire()
    └─ ExecuteSupport Task → (Future: Healing/Buffs)
    ↓
HealthComponent (Target)
    ├─ ApplyDamage() → Damage calculation
    ├─ OnDeath Event → FollowerAgentComponent::OnAllyKilled/OnEnemyKilled
    └─ Regen/Armor logic
    ↓
RewardCalculator
    ├─ +10 Kill reward
    ├─ +5 Damage dealt reward
    ├─ -5 Damage taken penalty
    └─ -10 Death penalty
```

### Key Features

1. **Health System**
   - Health pool with armor and regeneration
   - Death handling with respawn support
   - Damage events for AI feedback

2. **Weapon System**
   - Predictive aiming for moving targets
   - Cooldown and ammo management
   - Damage falloff with distance
   - Trace-based hit detection (hitscan)

3. **AI Integration**
   - Automatic reward calculation on damage/death
   - Observation integration (health, ammo, cooldown)
   - StateTree task integration

---

## Health Component

**File:** Combat/HealthComponent.h/cpp (HealthComponent.cpp:1-400)

### Core Functionality

The `UHealthComponent` manages agent health, armor, damage, death, and regeneration.

#### Properties

```cpp
// Health parameters
UPROPERTY(EditAnywhere, Category = "Health")
float MaxHealth = 100.0f;

UPROPERTY(EditAnywhere, Category = "Health")
float CurrentHealth = 100.0f;

UPROPERTY(EditAnywhere, Category = "Health")
float MaxArmor = 50.0f;

UPROPERTY(EditAnywhere, Category = "Health")
float CurrentArmor = 50.0f;

// Regeneration
UPROPERTY(EditAnywhere, Category = "Health|Regen")
bool bEnableRegen = true;

UPROPERTY(EditAnywhere, Category = "Health|Regen")
float RegenRate = 5.0f;  // HP per second

UPROPERTY(EditAnywhere, Category = "Health|Regen")
float RegenDelay = 5.0f;  // Seconds after last damage

// Death handling
UPROPERTY(EditAnywhere, Category = "Health|Death")
bool bAutoRespawn = false;

UPROPERTY(EditAnywhere, Category = "Health|Death")
float RespawnDelay = 5.0f;
```

#### Damage Application

**File:** HealthComponent.cpp:100-180

```cpp
void UHealthComponent::ApplyDamage(float DamageAmount, AActor* DamageInstigator,
                                    const FVector& HitLocation)
{
    if (!IsAlive() || DamageAmount <= 0.0f)
        return;

    // Reset regen timer
    TimeSinceLastDamage = 0.0f;

    // Apply to armor first
    float RemainingDamage = DamageAmount;
    if (CurrentArmor > 0.0f)
    {
        float ArmorAbsorb = FMath::Min(CurrentArmor, RemainingDamage * 0.5f);  // Armor absorbs 50%
        CurrentArmor -= ArmorAbsorb;
        RemainingDamage -= ArmorAbsorb;

        UE_LOG(LogCombat, Verbose, TEXT("%s: Armor absorbed %.1f damage (%.1f armor remaining)"),
               *GetOwner()->GetName(), ArmorAbsorb, CurrentArmor);
    }

    // Apply remaining to health
    CurrentHealth -= RemainingDamage;
    CurrentHealth = FMath::Max(CurrentHealth, 0.0f);

    UE_LOG(LogCombat, Log, TEXT("%s: Took %.1f damage (%.1f HP remaining)"),
           *GetOwner()->GetName(), DamageAmount, CurrentHealth);

    // Broadcast damage event
    OnDamageReceived.Broadcast(DamageAmount, DamageInstigator, HitLocation);

    // Check for death
    if (CurrentHealth <= 0.0f && !bIsDead)
    {
        Die(DamageInstigator);
    }
}
```

#### Death Handling

**File:** HealthComponent.cpp:200-250

```cpp
void UHealthComponent::Die(AActor* Killer)
{
    if (bIsDead)
        return;

    bIsDead = true;

    UE_LOG(LogCombat, Warning, TEXT("%s: Died (Killer: %s)"),
           *GetOwner()->GetName(),
           Killer ? *Killer->GetName() : TEXT("Unknown"));

    // Broadcast death event
    OnDeath.Broadcast(Killer);

    // Disable collision
    if (APawn* OwnerPawn = Cast<APawn>(GetOwner()))
    {
        OwnerPawn->SetActorEnableCollision(false);
    }

    // Auto-respawn if enabled
    if (bAutoRespawn)
    {
        GetWorld()->GetTimerManager().SetTimer(
            RespawnTimerHandle,
            this,
            &UHealthComponent::Respawn,
            RespawnDelay,
            false
        );
    }
}

void UHealthComponent::Respawn()
{
    bIsDead = false;
    CurrentHealth = MaxHealth;
    CurrentArmor = MaxArmor;

    UE_LOG(LogCombat, Log, TEXT("%s: Respawned"), *GetOwner()->GetName());

    // Re-enable collision
    if (APawn* OwnerPawn = Cast<APawn>(GetOwner()))
    {
        OwnerPawn->SetActorEnableCollision(true);
    }

    OnRespawn.Broadcast();
}
```

#### Regeneration

**File:** HealthComponent.cpp:280-320

```cpp
void UHealthComponent::TickComponent(float DeltaTime, ELevelTick TickType,
                                      FActorComponentTickFunction* ThisTickFunction)
{
    Super::TickComponent(DeltaTime, TickType, ThisTickFunction);

    if (!IsAlive() || !bEnableRegen)
        return;

    TimeSinceLastDamage += DeltaTime;

    // Start regen after delay
    if (TimeSinceLastDamage >= RegenDelay && CurrentHealth < MaxHealth)
    {
        float RegenAmount = RegenRate * DeltaTime;
        CurrentHealth = FMath::Min(CurrentHealth + RegenAmount, MaxHealth);

        UE_LOG(LogCombat, Verbose, TEXT("%s: Regenerating (%.1f HP)"),
               *GetOwner()->GetName(), CurrentHealth);
    }
}
```

#### Public API

```cpp
// Query
bool IsAlive() const { return !bIsDead; }
float GetHealth() const { return CurrentHealth; }
float GetHealthNormalized() const { return CurrentHealth / MaxHealth; }
float GetArmor() const { return CurrentArmor; }
float GetArmorNormalized() const { return CurrentArmor / MaxArmor; }

// Modification
void ApplyDamage(float DamageAmount, AActor* DamageInstigator, const FVector& HitLocation);
void Heal(float HealAmount);
void RestoreArmor(float ArmorAmount);
void Kill();

// Events
DECLARE_DYNAMIC_MULTICAST_DELEGATE_ThreeParams(FOnDamageReceivedSignature,
    float, DamageAmount, AActor*, DamageInstigator, FVector, HitLocation);

DECLARE_DYNAMIC_MULTICAST_DELEGATE_OneParam(FOnDeathSignature,
    AActor*, Killer);

DECLARE_DYNAMIC_MULTICAST_DELEGATE(FOnRespawnSignature);

FOnDamageReceivedSignature OnDamageReceived;
FOnDeathSignature OnDeath;
FOnRespawnSignature OnRespawn;
```

---

## Weapon Component

**File:** Combat/WeaponComponent.h/cpp (WeaponComponent.cpp:1-500)

### Core Functionality

The `UWeaponComponent` handles weapon firing, aiming, cooldown, ammo, and damage calculation.

#### Properties

```cpp
// Weapon stats
UPROPERTY(EditAnywhere, Category = "Weapon|Stats")
float Damage = 20.0f;

UPROPERTY(EditAnywhere, Category = "Weapon|Stats")
float FireRate = 0.1f;  // Seconds between shots (10 shots/sec)

UPROPERTY(EditAnywhere, Category = "Weapon|Stats")
float Range = 5000.0f;  // 50 meters

UPROPERTY(EditAnywhere, Category = "Weapon|Stats")
float Accuracy = 0.95f;  // 95% hit chance at optimal range

// Damage falloff
UPROPERTY(EditAnywhere, Category = "Weapon|Damage")
bool bEnableDamageFalloff = true;

UPROPERTY(EditAnywhere, Category = "Weapon|Damage")
float FalloffStartDistance = 2000.0f;  // 20 meters

UPROPERTY(EditAnywhere, Category = "Weapon|Damage")
float MinDamageMultiplier = 0.5f;  // 50% damage at max range

// Ammo
UPROPERTY(EditAnywhere, Category = "Weapon|Ammo")
int32 MaxAmmo = 30;

UPROPERTY(EditAnywhere, Category = "Weapon|Ammo")
int32 CurrentAmmo = 30;

UPROPERTY(EditAnywhere, Category = "Weapon|Ammo")
float ReloadTime = 2.0f;

// Predictive aiming
UPROPERTY(EditAnywhere, Category = "Weapon|Aiming")
bool bUsePredictiveAiming = true;

UPROPERTY(EditAnywhere, Category = "Weapon|Aiming")
float PredictionTime = 0.3f;  // Seconds to predict target movement
```

#### Firing

**File:** WeaponComponent.cpp:100-200

```cpp
bool UWeaponComponent::Fire(AActor* Target)
{
    if (!CanFire())
    {
        UE_LOG(LogCombat, Verbose, TEXT("%s: Cannot fire (cooldown/ammo)"),
               *GetOwner()->GetName());
        return false;
    }

    if (!Target || !IsInRange(Target))
    {
        UE_LOG(LogCombat, Verbose, TEXT("%s: Target out of range or invalid"),
               *GetOwner()->GetName());
        return false;
    }

    // Compute aim point
    FVector AimPoint = ComputeAimPoint(Target);

    // Perform line trace
    FHitResult HitResult;
    bool bHit = PerformWeaponTrace(AimPoint, HitResult);

    // Consume ammo
    CurrentAmmo--;
    LastFireTime = GetWorld()->GetTimeSeconds();

    if (bHit)
    {
        // Check if we hit the intended target
        AActor* HitActor = HitResult.GetActor();
        if (HitActor == Target)
        {
            // Apply damage
            float FinalDamage = CalculateDamage(HitResult.Distance);

            if (UHealthComponent* TargetHealth = HitActor->FindComponentByClass<UHealthComponent>())
            {
                TargetHealth->ApplyDamage(FinalDamage, GetOwner(), HitResult.ImpactPoint);

                UE_LOG(LogCombat, Log, TEXT("%s: Hit %s for %.1f damage"),
                       *GetOwner()->GetName(), *HitActor->GetName(), FinalDamage);

                // Broadcast hit event
                OnWeaponFired.Broadcast(Target, true, FinalDamage);

                return true;
            }
        }
    }

    // Miss
    UE_LOG(LogCombat, Verbose, TEXT("%s: Missed target"), *GetOwner()->GetName());
    OnWeaponFired.Broadcast(Target, false, 0.0f);

    return false;
}
```

#### Predictive Aiming

**File:** WeaponComponent.cpp:250-300

```cpp
FVector UWeaponComponent::ComputeAimPoint(AActor* Target)
{
    if (!Target)
        return FVector::ZeroVector;

    FVector TargetLocation = Target->GetActorLocation();

    if (!bUsePredictiveAiming)
        return TargetLocation;

    // Predict target movement
    if (APawn* TargetPawn = Cast<APawn>(Target))
    {
        FVector TargetVelocity = TargetPawn->GetVelocity();
        FVector PredictedLocation = TargetLocation + TargetVelocity * PredictionTime;

        UE_LOG(LogCombat, Verbose, TEXT("%s: Predictive aim (%.1f, %.1f, %.1f) -> (%.1f, %.1f, %.1f)"),
               *GetOwner()->GetName(),
               TargetLocation.X, TargetLocation.Y, TargetLocation.Z,
               PredictedLocation.X, PredictedLocation.Y, PredictedLocation.Z);

        return PredictedLocation;
    }

    return TargetLocation;
}

bool UWeaponComponent::PerformWeaponTrace(const FVector& AimPoint, FHitResult& OutHit)
{
    FVector StartLocation = GetComponentLocation();
    FVector Direction = (AimPoint - StartLocation).GetSafeNormal();
    FVector EndLocation = StartLocation + Direction * Range;

    // Add accuracy spread
    if (Accuracy < 1.0f)
    {
        float SpreadAngle = (1.0f - Accuracy) * 10.0f;  // Max 10° spread
        Direction = Direction.RotateAngleAxis(
            FMath::FRandRange(-SpreadAngle, SpreadAngle),
            FVector::UpVector
        );
    }

    // Line trace
    FCollisionQueryParams QueryParams;
    QueryParams.AddIgnoredActor(GetOwner());

    return GetWorld()->LineTraceSingleByChannel(
        OutHit,
        StartLocation,
        EndLocation,
        ECC_Pawn,
        QueryParams
    );
}
```

#### Damage Calculation

**File:** WeaponComponent.cpp:320-350

```cpp
float UWeaponComponent::CalculateDamage(float Distance)
{
    float FinalDamage = Damage;

    // Apply damage falloff
    if (bEnableDamageFalloff && Distance > FalloffStartDistance)
    {
        float FalloffRange = Range - FalloffStartDistance;
        float FalloffDistance = Distance - FalloffStartDistance;
        float FalloffRatio = FMath::Clamp(FalloffDistance / FalloffRange, 0.0f, 1.0f);

        float DamageMultiplier = FMath::Lerp(1.0f, MinDamageMultiplier, FalloffRatio);
        FinalDamage *= DamageMultiplier;

        UE_LOG(LogCombat, Verbose, TEXT("Damage falloff: %.1f -> %.1f (distance: %.1f)"),
               Damage, FinalDamage, Distance);
    }

    return FinalDamage;
}
```

#### Cooldown and Ammo

**File:** WeaponComponent.cpp:380-420

```cpp
bool UWeaponComponent::CanFire() const
{
    // Check ammo
    if (CurrentAmmo <= 0)
        return false;

    // Check cooldown
    float TimeSinceLastFire = GetWorld()->GetTimeSeconds() - LastFireTime;
    if (TimeSinceLastFire < FireRate)
        return false;

    // Check if reloading
    if (bIsReloading)
        return false;

    return true;
}

void UWeaponComponent::Reload()
{
    if (bIsReloading || CurrentAmmo == MaxAmmo)
        return;

    bIsReloading = true;

    UE_LOG(LogCombat, Log, TEXT("%s: Reloading..."), *GetOwner()->GetName());

    // Set reload timer
    GetWorld()->GetTimerManager().SetTimer(
        ReloadTimerHandle,
        this,
        &UWeaponComponent::FinishReload,
        ReloadTime,
        false
    );
}

void UWeaponComponent::FinishReload()
{
    CurrentAmmo = MaxAmmo;
    bIsReloading = false;

    UE_LOG(LogCombat, Log, TEXT("%s: Reload complete"), *GetOwner()->GetName());

    OnReloadComplete.Broadcast();
}

float UWeaponComponent::GetCooldownNormalized() const
{
    float TimeSinceLastFire = GetWorld()->GetTimeSeconds() - LastFireTime;
    return FMath::Clamp(TimeSinceLastFire / FireRate, 0.0f, 1.0f);
}

float UWeaponComponent::GetAmmoNormalized() const
{
    return (float)CurrentAmmo / (float)MaxAmmo;
}
```

#### Public API

```cpp
// Firing
bool Fire(AActor* Target);
bool CanFire() const;
bool IsInRange(AActor* Target) const;

// Ammo management
void Reload();
int32 GetCurrentAmmo() const { return CurrentAmmo; }
int32 GetMaxAmmo() const { return MaxAmmo; }
float GetAmmoNormalized() const;

// Cooldown
float GetCooldownNormalized() const;
bool IsOnCooldown() const;

// Query
float GetRange() const { return Range; }
float GetDamage() const { return Damage; }
float GetFireRate() const { return FireRate; }

// Events
DECLARE_DYNAMIC_MULTICAST_DELEGATE_ThreeParams(FOnWeaponFiredSignature,
    AActor*, Target, bool, bHit, float, Damage);

DECLARE_DYNAMIC_MULTICAST_DELEGATE(FOnReloadCompleteSignature);

FOnWeaponFiredSignature OnWeaponFired;
FOnReloadCompleteSignature OnReloadComplete;
```

---

## AI Integration

### Reward Calculation

**File:** FollowerAgentComponent.cpp:426-470

```cpp
void UFollowerAgentComponent::OnDamageDealt(AActor* Target, float DamageAmount)
{
    if (!RewardCalculator) return;

    // Build reward context
    FRewardContext Context;
    Context.DamageDealt = DamageAmount;

    // Check if coordinated
    Context.bExecutingStrategicCommand = IsExecutingCommand();
    Context.bCombinedFire = RewardCalculator->DetectCombinedFire(
        GetOwner(), Target, TeamLeader->GetFollowers());

    // Calculate reward
    float Reward = RewardCalculator->CalculateReward(Context);

    // Accumulate
    AccumulatedReward += Reward;

    UE_LOG(LogAI, Verbose, TEXT("%s: Damage reward +%.1f (total: %.1f)"),
           *GetName(), Reward, AccumulatedReward);
}

void UFollowerAgentComponent::OnEnemyKilled(AActor* Enemy, AActor* Killer)
{
    if (Killer != GetOwner()) return;
    if (!RewardCalculator) return;

    // Build reward context
    FRewardContext Context;
    Context.Kills = 1;
    Context.bExecutingStrategicCommand = IsExecutingCommand();
    Context.bCombinedFire = RewardCalculator->DetectCombinedFire(
        GetOwner(), Enemy, TeamLeader->GetFollowers());

    // Calculate reward
    float Reward = RewardCalculator->CalculateReward(Context);

    // Accumulate
    AccumulatedReward += Reward;

    UE_LOG(LogAI, Log, TEXT("%s: Kill reward +%.1f (total: %.1f)"),
           *GetName(), Reward, AccumulatedReward);

    // Report to team leader
    TeamLeader->OnFollowerKilledEnemy(this, Enemy);
}

void UFollowerAgentComponent::OnDamageTaken(float DamageAmount, AActor* DamageInstigator)
{
    if (!RewardCalculator) return;

    // Build reward context
    FRewardContext Context;
    Context.DamageTaken = DamageAmount;

    // Calculate penalty
    float Penalty = RewardCalculator->CalculateReward(Context);

    // Accumulate
    AccumulatedReward += Penalty;  // Will be negative

    UE_LOG(LogAI, Verbose, TEXT("%s: Damage penalty %.1f (total: %.1f)"),
           *GetName(), Penalty, AccumulatedReward);
}

void UFollowerAgentComponent::OnDeath(AActor* Killer)
{
    if (!RewardCalculator) return;

    // Build reward context
    FRewardContext Context;
    Context.bDied = true;

    // Calculate penalty
    float Penalty = RewardCalculator->CalculateReward(Context);

    // Accumulate
    AccumulatedReward += Penalty;  // Will be negative

    UE_LOG(LogAI, Warning, TEXT("%s: Death penalty %.1f (total: %.1f)"),
           *GetName(), Penalty, AccumulatedReward);

    // Report to team leader
    TeamLeader->OnFollowerDied(this);

    // Store final experience
    StoreExperience();
}
```

### Observation Integration

**File:** FollowerAgentComponent.cpp:634-699

```cpp
FObservationElement UFollowerAgentComponent::BuildObservation()
{
    FObservationElement Obs;

    // === Self State ===
    if (HealthComponent)
    {
        Obs.SelfHealth = HealthComponent->GetHealthNormalized();
        Obs.SelfArmor = HealthComponent->GetArmorNormalized();
    }

    if (WeaponComponent)
    {
        Obs.SelfAmmo = WeaponComponent->GetAmmoNormalized();
        Obs.SelfCooldown = WeaponComponent->GetCooldownNormalized();
    }

    Obs.SelfPosition = GetActorLocation() / 10000.0f;
    Obs.SelfRotation = GetActorRotation().Vector();
    Obs.SelfVelocity = GetVelocity() / 600.0f;

    // ... (rest of observation building)

    return Obs;
}
```

---

## StateTree Integration

### Execute Assault Task

**File:** StateTree/Tasks/STTask_ExecuteAssault.cpp:50-120

```cpp
EStateTreeRunStatus USTTask_ExecuteAssault::Tick(FStateTreeExecutionContext& Context, float DeltaTime) const
{
    FSTTask_ExecuteAssaultInstanceData& InstanceData = Context.GetInstanceData(*this);
    FFollowerStateTreeContext& FollowerContext = InstanceData.Context;

    AActor* Owner = Context.GetOwner();
    UFollowerAgentComponent* Follower = Owner->FindComponentByClass<UFollowerAgentComponent>();

    if (!Follower || !Follower->GetWeaponComponent())
        return EStateTreeRunStatus::Failed;

    UWeaponComponent* Weapon = Follower->GetWeaponComponent();

    // Get current target
    AActor* Target = FollowerContext.CurrentTarget;
    if (!Target)
    {
        // Find target via perception
        Target = Follower->GetPerceptionComponent()->GetNearestEnemy();
        FollowerContext.CurrentTarget = Target;
    }

    if (!Target)
    {
        UE_LOG(LogStateTree, Warning, TEXT("ExecuteAssault: No target available"));
        return EStateTreeRunStatus::Running;
    }

    // Move toward target if out of range
    if (!Weapon->IsInRange(Target))
    {
        FVector TargetLocation = Target->GetActorLocation();
        Follower->MoveToLocation(TargetLocation, Weapon->GetRange() * 0.8f);
        return EStateTreeRunStatus::Running;
    }

    // Fire at target
    if (Weapon->CanFire())
    {
        bool bHit = Weapon->Fire(Target);

        if (bHit)
        {
            UE_LOG(LogStateTree, Log, TEXT("ExecuteAssault: Hit target %s"),
                   *Target->GetName());
        }
    }

    // Check if target is dead
    if (UHealthComponent* TargetHealth = Target->FindComponentByClass<UHealthComponent>())
    {
        if (!TargetHealth->IsAlive())
        {
            FollowerContext.CurrentTarget = nullptr;
            return EStateTreeRunStatus::Succeeded;
        }
    }

    return EStateTreeRunStatus::Running;
}
```

### Execute Defend Task

**File:** StateTree/Tasks/STTask_ExecuteDefend.cpp:50-100

```cpp
EStateTreeRunStatus USTTask_ExecuteDefend::Tick(FStateTreeExecutionContext& Context, float DeltaTime) const
{
    FSTTask_ExecuteDefendInstanceData& InstanceData = Context.GetInstanceData(*this);
    FFollowerStateTreeContext& FollowerContext = InstanceData.Context;

    AActor* Owner = Context.GetOwner();
    UFollowerAgentComponent* Follower = Owner->FindComponentByClass<UFollowerAgentComponent>();

    if (!Follower)
        return EStateTreeRunStatus::Failed;

    // Get defend location
    FVector DefendLocation = FollowerContext.CurrentStrategicCommand.TargetLocation;

    // Move to defend location if not there
    float DistanceToLocation = FVector::Dist(Owner->GetActorLocation(), DefendLocation);
    if (DistanceToLocation > 200.0f)  // 2 meters tolerance
    {
        Follower->MoveToLocation(DefendLocation);
        return EStateTreeRunStatus::Running;
    }

    // Find cover if available
    if (!FollowerContext.bInCover)
    {
        FVector CoverLocation = Follower->FindNearestCover(DefendLocation);
        if (!CoverLocation.IsZero())
        {
            Follower->MoveToLocation(CoverLocation);
            FollowerContext.bInCover = true;
        }
    }

    // Overwatch: Fire at enemies that approach
    if (UWeaponComponent* Weapon = Follower->GetWeaponComponent())
    {
        AActor* ThreatTarget = Follower->GetPerceptionComponent()->GetNearestEnemy();

        if (ThreatTarget && Weapon->IsInRange(ThreatTarget) && Weapon->CanFire())
        {
            Weapon->Fire(ThreatTarget);
        }
    }

    return EStateTreeRunStatus::Running;
}
```

---

## Configuration

### Health Component Defaults

**Recommended values for different agent types:**

```cpp
// Assault Agent (frontline fighter)
MaxHealth = 120.0f;
MaxArmor = 50.0f;
RegenRate = 5.0f;
RegenDelay = 5.0f;

// Defender (tank)
MaxHealth = 150.0f;
MaxArmor = 80.0f;
RegenRate = 3.0f;
RegenDelay = 6.0f;

// Support (backline)
MaxHealth = 80.0f;
MaxArmor = 30.0f;
RegenRate = 7.0f;
RegenDelay = 4.0f;
```

### Weapon Component Defaults

**Recommended values for different weapon types:**

```cpp
// Assault Rifle (balanced)
Damage = 20.0f;
FireRate = 0.1f;  // 10 shots/sec
Range = 5000.0f;  // 50 meters
Accuracy = 0.95f;
MaxAmmo = 30;
ReloadTime = 2.0f;

// Sniper Rifle (high damage, low rate)
Damage = 80.0f;
FireRate = 1.0f;  // 1 shot/sec
Range = 10000.0f;  // 100 meters
Accuracy = 0.98f;
MaxAmmo = 5;
ReloadTime = 3.0f;

// SMG (high rate, low damage)
Damage = 12.0f;
FireRate = 0.05f;  // 20 shots/sec
Range = 3000.0f;  // 30 meters
Accuracy = 0.90f;
MaxAmmo = 50;
ReloadTime = 1.5f;
```

---

## Troubleshooting

### Weapon Not Firing

**Symptom:** `CanFire()` returns false

**Causes:**
1. On cooldown (fired too recently)
2. Out of ammo
3. Currently reloading

**Solutions:**
- Check `FireRate` is reasonable (0.1-1.0s typical)
- Verify ammo system is working (`GetCurrentAmmo()`)
- Ensure reload completes (check `ReloadTime`)

### Damage Not Applied

**Symptom:** Hit detected but no damage dealt

**Causes:**
1. Target has no HealthComponent
2. Target already dead
3. Damage calculation returning 0

**Solutions:**
- Verify target has `UHealthComponent`
- Check `IsAlive()` before applying damage
- Verify `Damage` property is > 0
- Check damage falloff settings

### Predictive Aiming Missing

**Symptom:** All shots miss moving targets

**Causes:**
1. `PredictionTime` too high/low
2. Target velocity incorrect
3. Accuracy too low

**Solutions:**
- Tune `PredictionTime` (typical: 0.2-0.5s)
- Verify target has valid velocity
- Increase `Accuracy` (0.90-0.98)

### Regeneration Not Working

**Symptom:** Health not regenerating

**Causes:**
1. `bEnableRegen` disabled
2. Taking continuous damage (resets timer)
3. `RegenDelay` too long

**Solutions:**
- Enable `bEnableRegen` in component properties
- Check `TimeSinceLastDamage` is reaching `RegenDelay`
- Reduce `RegenDelay` (3-5s typical)

### Reward Events Not Firing

**Symptom:** No rewards calculated on damage/death

**Causes:**
1. Events not bound in FollowerAgentComponent
2. RewardCalculator not initialized
3. HealthComponent/WeaponComponent not broadcasting

**Solutions:**
- Verify event binding in `FollowerAgentComponent::BeginPlay()`
- Check `RewardCalculator` is valid pointer
- Ensure combat components call `OnDamageReceived.Broadcast()`, etc.

---

## Performance Notes

- **HealthComponent Tick:** ~0.05ms per agent (regen logic)
- **WeaponComponent Tick:** Minimal (event-driven)
- **Line Traces:** ~0.1ms per shot (hitscan)
- **Damage Calculation:** <0.01ms per hit

For large battles (20+ agents), consider:
- Stagger regen ticks across agents
- Pool line trace queries
- Reduce trace frequency for distant agents

---

## Next Steps

1. **Tuning**: Balance damage, health, and fire rates for gameplay
2. **Expansion**: Add healing, buffs, status effects
3. **Weapons**: Implement multiple weapon types (projectile-based)
4. **Cover**: Deep integration with EQS cover system
5. **Visual Feedback**: Add VFX/SFX for hits, deaths, reloads

---

**Document Version:** 1.0
**Last Updated:** 2025-11-26
**Maintained By:** SBDAPM Team
