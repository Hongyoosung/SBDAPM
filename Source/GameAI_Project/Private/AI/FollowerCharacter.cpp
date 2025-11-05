// Copyright Epic Games, Inc. All Rights Reserved.

#include "AI/FollowerCharacter.h"
#include "Team/FollowerAgentComponent.h"
#include "StateTree/FollowerStateTreeComponent.h"

AFollowerCharacter::AFollowerCharacter()
{
	PrimaryActorTick.bCanEverTick = true;

	// Create follower agent component
	FollowerAgentComponent = CreateDefaultSubobject<UFollowerAgentComponent>(TEXT("FollowerAgentComponent"));

	// Create state tree component
	StateTreeComponent = CreateDefaultSubobject<UFollowerStateTreeComponent>(TEXT("StateTreeComponent"));
}

void AFollowerCharacter::BeginPlay()
{
	Super::BeginPlay();
}

void AFollowerCharacter::Tick(float DeltaTime)
{
	Super::Tick(DeltaTime);

	// Update combat timers
	UpdateTimers(DeltaTime);
}

//------------------------------------------------------------------------------
// COMBAT STATS (ICombatStatsInterface Implementation)
//------------------------------------------------------------------------------

float AFollowerCharacter::GetHealthPercentage_Implementation() const
{
	return (MaxHealth > 0.0f) ? (CurrentHealth / MaxHealth) * 100.0f : 0.0f;
}

float AFollowerCharacter::GetStaminaPercentage_Implementation() const
{
	return (MaxStamina > 0.0f) ? (CurrentStamina / MaxStamina) * 100.0f : 0.0f;
}

float AFollowerCharacter::GetShieldPercentage_Implementation() const
{
	return (MaxShield > 0.0f) ? (CurrentShield / MaxShield) * 100.0f : 0.0f;
}

bool AFollowerCharacter::IsAlive_Implementation() const
{
	return !bIsDead && CurrentHealth > 0.0f;
}

float AFollowerCharacter::GetWeaponCooldown_Implementation() const
{
	return CurrentWeaponCooldown;
}

float AFollowerCharacter::GetAmmunition_Implementation() const
{
	return (MaxAmmo > 0.0f) ? (CurrentAmmo / MaxAmmo) * 100.0f : 0.0f;
}

int32 AFollowerCharacter::GetWeaponType_Implementation() const
{
	return WeaponType;
}

bool AFollowerCharacter::CanFireWeapon_Implementation() const
{
	return IsAlive_Implementation() && CurrentWeaponCooldown <= 0.0f && CurrentAmmo >= AmmoCostPerShot;
}

//------------------------------------------------------------------------------
// HEALTH SYSTEM
//------------------------------------------------------------------------------

void AFollowerCharacter::TakeDamage(float DamageAmount)
{
	if (bIsDead || DamageAmount <= 0.0f)
	{
		return;
	}

	// Apply to shield first
	if (CurrentShield > 0.0f)
	{
		float ShieldAbsorbed = FMath::Min(CurrentShield, DamageAmount);
		CurrentShield -= ShieldAbsorbed;
		DamageAmount -= ShieldAbsorbed;
	}

	// Apply remaining to health
	if (DamageAmount > 0.0f)
	{
		CurrentHealth = FMath::Max(0.0f, CurrentHealth - DamageAmount);
	}

	// Reset shield regen timer
	TimeSinceLastDamage = 0.0f;

	// Check for death
	if (CurrentHealth <= 0.0f)
	{
		Kill();
	}
}

void AFollowerCharacter::Heal(float HealAmount)
{
	if (bIsDead)
	{
		return;
	}

	CurrentHealth = FMath::Min(MaxHealth, CurrentHealth + HealAmount);
}

void AFollowerCharacter::Kill()
{
	if (bIsDead)
	{
		return;
	}

	bIsDead = true;
	CurrentHealth = 0.0f;

	// Notify follower component
	if (FollowerAgentComponent)
	{
		FollowerAgentComponent->MarkAsDead();
	}

	UE_LOG(LogTemp, Log, TEXT("FollowerCharacter '%s' died"), *GetName());
}

void AFollowerCharacter::Respawn()
{
	bIsDead = false;
	CurrentHealth = MaxHealth;
	CurrentShield = 0.0f;
	CurrentStamina = MaxStamina;
	CurrentAmmo = MaxAmmo;
	CurrentWeaponCooldown = 0.0f;

	// Notify follower component
	if (FollowerAgentComponent)
	{
		FollowerAgentComponent->MarkAsAlive();
		FollowerAgentComponent->ResetEpisode();
	}

	UE_LOG(LogTemp, Log, TEXT("FollowerCharacter '%s' respawned"), *GetName());
}

//------------------------------------------------------------------------------
// STAMINA SYSTEM
//------------------------------------------------------------------------------

bool AFollowerCharacter::ConsumeStamina(float Amount)
{
	if (CurrentStamina >= Amount)
	{
		CurrentStamina -= Amount;
		return true;
	}
	return false;
}

//------------------------------------------------------------------------------
// WEAPON SYSTEM
//------------------------------------------------------------------------------

bool AFollowerCharacter::FireWeapon()
{
	if (!CanFireWeapon_Implementation())
	{
		return false;
	}

	// Consume ammo
	CurrentAmmo -= AmmoCostPerShot;

	// Start cooldown
	CurrentWeaponCooldown = WeaponCooldownDuration;

	return true;
}

void AFollowerCharacter::ReloadWeapon()
{
	CurrentAmmo = MaxAmmo;
}

//------------------------------------------------------------------------------
// PRIVATE
//------------------------------------------------------------------------------

void AFollowerCharacter::UpdateTimers(float DeltaTime)
{
	UpdateWeaponCooldown(DeltaTime);
	UpdateStaminaRegeneration(DeltaTime);
	UpdateShieldRegeneration(DeltaTime);
}

void AFollowerCharacter::UpdateShieldRegeneration(float DeltaTime)
{
	if (bIsDead)
	{
		return;
	}

	TimeSinceLastDamage += DeltaTime;

	// Regenerate shield after delay
	if (TimeSinceLastDamage >= ShieldRegenDelay && CurrentShield < MaxShield)
	{
		CurrentShield = FMath::Min(MaxShield, CurrentShield + ShieldRegenRate * DeltaTime);
	}
}

void AFollowerCharacter::UpdateStaminaRegeneration(float DeltaTime)
{
	if (bIsDead)
	{
		return;
	}

	if (CurrentStamina < MaxStamina)
	{
		CurrentStamina = FMath::Min(MaxStamina, CurrentStamina + StaminaRegenRate * DeltaTime);
	}
}

void AFollowerCharacter::UpdateWeaponCooldown(float DeltaTime)
{
	if (CurrentWeaponCooldown > 0.0f)
	{
		CurrentWeaponCooldown = FMath::Max(0.0f, CurrentWeaponCooldown - DeltaTime);
	}
}
