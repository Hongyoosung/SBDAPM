// Copyright Epic Games, Inc. All Rights Reserved.

#include "AI/GameAICharacter.h"
#include "Core/StateMachine.h"
#include "Team/FollowerAgentComponent.h"
#include "GameFramework/CharacterMovementComponent.h"

AGameAICharacter::AGameAICharacter()
{
	PrimaryActorTick.bCanEverTick = true;

	// Create AI components
	StateMachine = CreateDefaultSubobject<UStateMachine>(TEXT("StateMachine"));
	FollowerAgent = CreateDefaultSubobject<UFollowerAgentComponent>(TEXT("FollowerAgent"));

	// Configure character movement for AI
	GetCharacterMovement()->bOrientRotationToMovement = true;
	GetCharacterMovement()->RotationRate = FRotator(0.0f, 360.0f, 0.0f);
	GetCharacterMovement()->MaxWalkSpeed = 500.0f;

	// AI-controlled, not player-controlled
	AutoPossessAI = EAutoPossessAI::PlacedInWorldOrSpawned;
}

void AGameAICharacter::BeginPlay()
{
	Super::BeginPlay();

	// Initialize health/stamina/shield to max
	CurrentHealth = MaxHealth;
	CurrentStamina = MaxStamina;
	CurrentShield = MaxShield;
	CurrentAmmo = MaxAmmo;

	UE_LOG(LogTemp, Log, TEXT("GameAICharacter '%s': Initialized with Health=%.1f, Stamina=%.1f, Shield=%.1f"),
		*GetName(), CurrentHealth, CurrentStamina, CurrentShield);
}

void AGameAICharacter::Tick(float DeltaTime)
{
	Super::Tick(DeltaTime);

	if (!bIsDead)
	{
		UpdateTimers(DeltaTime);
	}
}

//------------------------------------------------------------------------------
// COMBAT STATS INTERFACE IMPLEMENTATION
//------------------------------------------------------------------------------

float AGameAICharacter::GetHealthPercentage_Implementation() const
{
	return (CurrentHealth / MaxHealth) * 100.0f;
}

float AGameAICharacter::GetStaminaPercentage_Implementation() const
{
	return (CurrentStamina / MaxStamina) * 100.0f;
}

float AGameAICharacter::GetShieldPercentage_Implementation() const
{
	return (CurrentShield / MaxShield) * 100.0f;
}

bool AGameAICharacter::IsAlive_Implementation() const
{
	return !bIsDead && CurrentHealth > 0.0f;
}

float AGameAICharacter::GetWeaponCooldown_Implementation() const
{
	return CurrentWeaponCooldown;
}

float AGameAICharacter::GetAmmunition_Implementation() const
{
	return CurrentAmmo;
}

int32 AGameAICharacter::GetWeaponType_Implementation() const
{
	return WeaponType;
}

bool AGameAICharacter::CanFireWeapon_Implementation() const
{
	return !bIsDead
		&& CurrentWeaponCooldown <= 0.0f
		&& CurrentAmmo >= AmmoCostPerShot;
}

//------------------------------------------------------------------------------
// HEALTH SYSTEM
//------------------------------------------------------------------------------

void AGameAICharacter::TakeDamage(float DamageAmount)
{
	if (bIsDead || DamageAmount <= 0.0f)
	{
		return;
	}

	// Reset shield regeneration timer
	TimeSinceLastDamage = 0.0f;

	// Apply damage to shield first
	if (CurrentShield > 0.0f)
	{
		float ShieldDamage = FMath::Min(DamageAmount, CurrentShield);
		CurrentShield -= ShieldDamage;
		DamageAmount -= ShieldDamage;

		UE_LOG(LogTemp, Verbose, TEXT("GameAICharacter '%s': Shield absorbed %.1f damage (%.1f remaining)"),
			*GetName(), ShieldDamage, CurrentShield);
	}

	// Apply remaining damage to health
	if (DamageAmount > 0.0f)
	{
		CurrentHealth -= DamageAmount;
		UE_LOG(LogTemp, Log, TEXT("GameAICharacter '%s': Took %.1f damage (Health: %.1f/%.1f)"),
			*GetName(), DamageAmount, CurrentHealth, MaxHealth);

		// Check for death
		if (CurrentHealth <= 0.0f)
		{
			Kill();
		}
	}
}

void AGameAICharacter::Heal(float HealAmount)
{
	if (bIsDead || HealAmount <= 0.0f)
	{
		return;
	}

	float OldHealth = CurrentHealth;
	CurrentHealth = FMath::Clamp(CurrentHealth + HealAmount, 0.0f, MaxHealth);

	UE_LOG(LogTemp, Log, TEXT("GameAICharacter '%s': Healed %.1f (%.1f → %.1f)"),
		*GetName(), HealAmount, OldHealth, CurrentHealth);
}

void AGameAICharacter::Kill()
{
	if (bIsDead)
	{
		return;
	}

	CurrentHealth = 0.0f;
	bIsDead = true;

	UE_LOG(LogTemp, Warning, TEXT("GameAICharacter '%s': Killed"), *GetName());

	// Notify FollowerAgent
	if (FollowerAgent)
	{
		FollowerAgent->MarkAsDead();
	}

	// Disable movement
	GetCharacterMovement()->DisableMovement();

	// TODO: Play death animation, disable collision, etc.
}

void AGameAICharacter::Respawn()
{
	if (!bIsDead)
	{
		return;
	}

	// Reset stats
	CurrentHealth = MaxHealth;
	CurrentStamina = MaxStamina;
	CurrentShield = MaxShield;
	CurrentAmmo = MaxAmmo;
	bIsDead = false;

	// Notify FollowerAgent
	if (FollowerAgent)
	{
		FollowerAgent->MarkAsAlive();
	}

	// Re-enable movement
	GetCharacterMovement()->SetMovementMode(MOVE_Walking);

	UE_LOG(LogTemp, Log, TEXT("GameAICharacter '%s': Respawned"), *GetName());
}

//------------------------------------------------------------------------------
// STAMINA SYSTEM
//------------------------------------------------------------------------------

bool AGameAICharacter::ConsumeStamina(float Amount)
{
	if (bIsDead || Amount <= 0.0f)
	{
		return false;
	}

	if (CurrentStamina >= Amount)
	{
		CurrentStamina -= Amount;
		UE_LOG(LogTemp, Verbose, TEXT("GameAICharacter '%s': Consumed %.1f stamina (%.1f remaining)"),
			*GetName(), Amount, CurrentStamina);
		return true;
	}

	UE_LOG(LogTemp, Verbose, TEXT("GameAICharacter '%s': Insufficient stamina (%.1f < %.1f)"),
		*GetName(), CurrentStamina, Amount);
	return false;
}

//------------------------------------------------------------------------------
// WEAPON SYSTEM
//------------------------------------------------------------------------------

bool AGameAICharacter::FireWeapon()
{
	if (!CanFireWeapon_Implementation())
	{
		UE_LOG(LogTemp, Verbose, TEXT("GameAICharacter '%s': Cannot fire weapon (Dead=%d, Cooldown=%.2f, Ammo=%.1f)"),
			*GetName(), bIsDead, CurrentWeaponCooldown, CurrentAmmo);
		return false;
	}

	// Consume ammo
	CurrentAmmo -= AmmoCostPerShot;

	// Start cooldown
	CurrentWeaponCooldown = WeaponCooldownDuration;

	UE_LOG(LogTemp, Verbose, TEXT("GameAICharacter '%s': Fired weapon (Ammo: %.1f, Cooldown: %.2fs)"),
		*GetName(), CurrentAmmo, CurrentWeaponCooldown);

	// TODO: Spawn projectile, play effects, etc.

	return true;
}

void AGameAICharacter::ReloadWeapon()
{
	if (bIsDead)
	{
		return;
	}

	float OldAmmo = CurrentAmmo;
	CurrentAmmo = MaxAmmo;

	UE_LOG(LogTemp, Log, TEXT("GameAICharacter '%s': Reloaded weapon (%.1f → %.1f)"),
		*GetName(), OldAmmo, CurrentAmmo);

	// TODO: Play reload animation, set cooldown
}

//------------------------------------------------------------------------------
// PRIVATE - UPDATE TIMERS
//------------------------------------------------------------------------------

void AGameAICharacter::UpdateTimers(float DeltaTime)
{
	UpdateWeaponCooldown(DeltaTime);
	UpdateStaminaRegeneration(DeltaTime);
	UpdateShieldRegeneration(DeltaTime);

	// Update damage timer
	TimeSinceLastDamage += DeltaTime;
}

void AGameAICharacter::UpdateWeaponCooldown(float DeltaTime)
{
	if (CurrentWeaponCooldown > 0.0f)
	{
		CurrentWeaponCooldown -= DeltaTime;
		if (CurrentWeaponCooldown < 0.0f)
		{
			CurrentWeaponCooldown = 0.0f;
		}
	}
}

void AGameAICharacter::UpdateStaminaRegeneration(float DeltaTime)
{
	if (CurrentStamina < MaxStamina)
	{
		CurrentStamina = FMath::Clamp(
			CurrentStamina + (StaminaRegenRate * DeltaTime),
			0.0f,
			MaxStamina
		);
	}
}

void AGameAICharacter::UpdateShieldRegeneration(float DeltaTime)
{
	// Only regenerate shield if enough time has passed since last damage
	if (CurrentShield < MaxShield && TimeSinceLastDamage >= ShieldRegenDelay)
	{
		CurrentShield = FMath::Clamp(
			CurrentShield + (ShieldRegenRate * DeltaTime),
			0.0f,
			MaxShield
		);
	}
}
