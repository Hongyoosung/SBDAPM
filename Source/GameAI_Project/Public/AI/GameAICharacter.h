// Copyright Epic Games, Inc. All Rights Reserved.

#pragma once

#include "CoreMinimal.h"
#include "GameFramework/Character.h"
#include "Interfaces/CombatStatsInterface.h"
#include "GameAICharacter.generated.h"

class UStateMachine;
class UFollowerAgentComponent;

/**
 * Game AI Character - Example implementation of CombatStatsInterface
 *
 * This character class provides a complete example of implementing
 * the CombatStatsInterface in C++ (Option B from Phase 4 Week 12).
 *
 * Features:
 * - Health, Stamina, Shield systems
 * - Weapon system with cooldowns and ammo
 * - Integration with FollowerAgentComponent
 * - Integration with StateMachine
 *
 * Usage:
 * 1. Use this class directly for AI agents
 * 2. Or use it as a reference for implementing CombatStatsInterface in your own characters
 */
UCLASS()
class GAMEAI_PROJECT_API AGameAICharacter : public ACharacter, public ICombatStatsInterface
{
	GENERATED_BODY()

public:
	AGameAICharacter();

protected:
	virtual void BeginPlay() override;

public:
	virtual void Tick(float DeltaTime) override;

	//--------------------------------------------------------------------------
	// COMBAT STATS (Implementation of ICombatStatsInterface)
	//--------------------------------------------------------------------------

	/** Get current health percentage (0.0 - 100.0) */
	virtual float GetHealthPercentage_Implementation() const override;

	/** Get current stamina percentage (0.0 - 100.0) */
	virtual float GetStaminaPercentage_Implementation() const override;

	/** Get current shield percentage (0.0 - 100.0) */
	virtual float GetShieldPercentage_Implementation() const override;

	/** Is the character alive? */
	virtual bool IsAlive_Implementation() const override;

	/** Get current weapon cooldown (seconds) */
	virtual float GetWeaponCooldown_Implementation() const override;

	/** Get current ammunition count/percentage */
	virtual float GetAmmunition_Implementation() const override;

	/** Get current weapon type ID */
	virtual int32 GetWeaponType_Implementation() const override;

	/** Can fire weapon right now? */
	virtual bool CanFireWeapon_Implementation() const override;

	//--------------------------------------------------------------------------
	// HEALTH SYSTEM
	//--------------------------------------------------------------------------

	/** Maximum health */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Combat|Health")
	float MaxHealth = 100.0f;

	/** Current health */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Combat|Health")
	float CurrentHealth = 100.0f;

	/** Apply damage to this character */
	UFUNCTION(BlueprintCallable, Category = "Combat|Health")
	void TakeDamage(float DamageAmount);

	/** Heal this character */
	UFUNCTION(BlueprintCallable, Category = "Combat|Health")
	void Heal(float HealAmount);

	/** Kill this character instantly */
	UFUNCTION(BlueprintCallable, Category = "Combat|Health")
	void Kill();

	/** Respawn this character */
	UFUNCTION(BlueprintCallable, Category = "Combat|Health")
	void Respawn();

	//--------------------------------------------------------------------------
	// STAMINA SYSTEM
	//--------------------------------------------------------------------------

	/** Maximum stamina */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Combat|Stamina")
	float MaxStamina = 100.0f;

	/** Current stamina */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Combat|Stamina")
	float CurrentStamina = 100.0f;

	/** Stamina regeneration rate per second */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Combat|Stamina")
	float StaminaRegenRate = 10.0f;

	/** Consume stamina (e.g., for sprinting, abilities) */
	UFUNCTION(BlueprintCallable, Category = "Combat|Stamina")
	bool ConsumeStamina(float Amount);

	//--------------------------------------------------------------------------
	// SHIELD SYSTEM
	//--------------------------------------------------------------------------

	/** Maximum shield */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Combat|Shield")
	float MaxShield = 50.0f;

	/** Current shield */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Combat|Shield")
	float CurrentShield = 0.0f;

	/** Shield regeneration rate per second */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Combat|Shield")
	float ShieldRegenRate = 5.0f;

	/** Time delay before shield starts regenerating after damage */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Combat|Shield")
	float ShieldRegenDelay = 3.0f;

	//--------------------------------------------------------------------------
	// WEAPON SYSTEM
	//--------------------------------------------------------------------------

	/** Current weapon type */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Combat|Weapon")
	int32 WeaponType = 1;

	/** Maximum ammunition */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Combat|Weapon")
	float MaxAmmo = 100.0f;

	/** Current ammunition */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Combat|Weapon")
	float CurrentAmmo = 100.0f;

	/** Weapon cooldown duration (seconds) */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Combat|Weapon")
	float WeaponCooldownDuration = 0.5f;

	/** Current weapon cooldown timer */
	UPROPERTY(BlueprintReadOnly, Category = "Combat|Weapon")
	float CurrentWeaponCooldown = 0.0f;

	/** Ammo cost per shot */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Combat|Weapon")
	float AmmoCostPerShot = 1.0f;

	/** Fire weapon */
	UFUNCTION(BlueprintCallable, Category = "Combat|Weapon")
	bool FireWeapon();

	/** Reload weapon */
	UFUNCTION(BlueprintCallable, Category = "Combat|Weapon")
	void ReloadWeapon();

	//--------------------------------------------------------------------------
	// AI COMPONENTS
	//--------------------------------------------------------------------------

	/** State Machine component (legacy FSM system) */
	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "AI")
	UStateMachine* StateMachine = nullptr;

	/** Follower Agent component (new hierarchical system) */
	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "AI")
	UFollowerAgentComponent* FollowerAgent = nullptr;

private:
	/** Time since last damage (for shield regen) */
	float TimeSinceLastDamage = 0.0f;

	/** Is character dead? */
	bool bIsDead = false;

	/** Update timers (cooldowns, regeneration) */
	void UpdateTimers(float DeltaTime);

	/** Update shield regeneration */
	void UpdateShieldRegeneration(float DeltaTime);

	/** Update stamina regeneration */
	void UpdateStaminaRegeneration(float DeltaTime);

	/** Update weapon cooldown */
	void UpdateWeaponCooldown(float DeltaTime);
};
