// Copyright Epic Games, Inc. All Rights Reserved.

#include "AI/LeaderCharacter.h"
#include "Team/TeamLeaderComponent.h"

ALeaderCharacter::ALeaderCharacter()
{
	PrimaryActorTick.bCanEverTick = true;

	// Create team leader component
	TeamLeaderComponent = CreateDefaultSubobject<UTeamLeaderComponent>(TEXT("TeamLeaderComponent"));
}

void ALeaderCharacter::BeginPlay()
{
	Super::BeginPlay();
}

void ALeaderCharacter::Tick(float DeltaTime)
{
	Super::Tick(DeltaTime);

	// Update combat timers
	UpdateTimers(DeltaTime);
}

//------------------------------------------------------------------------------
// COMBAT STATS (ICombatStatsInterface Implementation)
//------------------------------------------------------------------------------

float ALeaderCharacter::GetHealthPercentage_Implementation() const
{
	return (MaxHealth > 0.0f) ? (CurrentHealth / MaxHealth) * 100.0f : 0.0f;
}

float ALeaderCharacter::GetStaminaPercentage_Implementation() const
{
	return (MaxStamina > 0.0f) ? (CurrentStamina / MaxStamina) * 100.0f : 0.0f;
}

float ALeaderCharacter::GetShieldPercentage_Implementation() const
{
	return (MaxShield > 0.0f) ? (CurrentShield / MaxShield) * 100.0f : 0.0f;
}

bool ALeaderCharacter::IsAlive_Implementation() const
{
	return !bIsDead && CurrentHealth > 0.0f;
}

float ALeaderCharacter::GetWeaponCooldown_Implementation() const
{
	return CurrentWeaponCooldown;
}

float ALeaderCharacter::GetAmmunition_Implementation() const
{
	return (MaxAmmo > 0.0f) ? (CurrentAmmo / MaxAmmo) * 100.0f : 0.0f;
}

int32 ALeaderCharacter::GetWeaponType_Implementation() const
{
	return WeaponType;
}

bool ALeaderCharacter::CanFireWeapon_Implementation() const
{
	return IsAlive_Implementation() && CurrentWeaponCooldown <= 0.0f && CurrentAmmo >= AmmoCostPerShot;
}

//------------------------------------------------------------------------------
// HEALTH SYSTEM
//------------------------------------------------------------------------------

void ALeaderCharacter::TakeDamage(float DamageAmount)
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

void ALeaderCharacter::Heal(float HealAmount)
{
	if (bIsDead)
	{
		return;
	}

	CurrentHealth = FMath::Min(MaxHealth, CurrentHealth + HealAmount);
}

void ALeaderCharacter::Kill()
{
	if (bIsDead)
	{
		return;
	}

	bIsDead = true;
	CurrentHealth = 0.0f;

	UE_LOG(LogTemp, Log, TEXT("LeaderCharacter '%s' died"), *GetName());
}

void ALeaderCharacter::Respawn()
{
	bIsDead = false;
	CurrentHealth = MaxHealth;
	CurrentShield = 0.0f;
	CurrentStamina = MaxStamina;
	CurrentAmmo = MaxAmmo;
	CurrentWeaponCooldown = 0.0f;

	UE_LOG(LogTemp, Log, TEXT("LeaderCharacter '%s' respawned"), *GetName());
}

//------------------------------------------------------------------------------
// STAMINA SYSTEM
//------------------------------------------------------------------------------

bool ALeaderCharacter::ConsumeStamina(float Amount)
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

bool ALeaderCharacter::FireWeapon()
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

void ALeaderCharacter::ReloadWeapon()
{
	CurrentAmmo = MaxAmmo;
}

//------------------------------------------------------------------------------
// PRIVATE
//------------------------------------------------------------------------------

void ALeaderCharacter::UpdateTimers(float DeltaTime)
{
	UpdateWeaponCooldown(DeltaTime);
	UpdateStaminaRegeneration(DeltaTime);
	UpdateShieldRegeneration(DeltaTime);
}

void ALeaderCharacter::UpdateShieldRegeneration(float DeltaTime)
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

void ALeaderCharacter::UpdateStaminaRegeneration(float DeltaTime)
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

void ALeaderCharacter::UpdateWeaponCooldown(float DeltaTime)
{
	if (CurrentWeaponCooldown > 0.0f)
	{
		CurrentWeaponCooldown = FMath::Max(0.0f, CurrentWeaponCooldown - DeltaTime);
	}
}
