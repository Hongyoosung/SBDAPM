// TacticalActuator.h - Schola actuator for 16 discrete tactical actions

#pragma once

#include "CoreMinimal.h"
#include "Actuators/AbstractActuators.h"
#include "RL/RLTypes.h"
#include "TacticalActuator.generated.h"

class UFollowerAgentComponent;
class UFollowerStateTreeComponent;

/**
 * Schola actuator that maps discrete actions to ETacticalAction enum.
 * Used for live training via gRPC connection to Python/RLlib.
 *
 * Actions (16 total - ETacticalAction):
 * 0  = Advance
 * 1  = Retreat
 * 2  = FlankLeft
 * 3  = FlankRight
 * 4  = TakeCover
 * 5  = SuppressiveFire
 * 6  = HoldPosition
 * 7  = AssaultTarget
 * 8  = DefendPosition
 * 9  = Regroup
 * 10 = Heal
 * 11 = Reload
 * 12 = CallSupport
 * 13 = SetAmbush
 * 14 = AggressivePush
 * 15 = DefensiveHold
 */
UCLASS(BlueprintType, meta = (DisplayName = "Tactical Actuator"))
class GAMEAI_PROJECT_API UTacticalActuator : public UDiscreteActuator
{
	GENERATED_BODY()

public:
	UTacticalActuator();

	// UDiscreteActuator interface
	virtual FDiscreteSpace GetActionSpace() override;
	virtual void TakeAction(const FDiscretePoint& Action) override;
	virtual void InitializeActuator() override;
	virtual void ResetActuator() override;

	/** The follower agent component to control */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Actuator")
	UFollowerAgentComponent* FollowerAgent = nullptr;

	/** Auto-find follower agent on owner actor */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Actuator")
	bool bAutoFindFollower = true;

	/** Last action taken */
	UPROPERTY(BlueprintReadOnly, Category = "Actuator")
	ETacticalAction LastAction = ETacticalAction::DefensiveHold;

protected:
	/** Cached action space (16 discrete actions) */
	FDiscreteSpace CachedActionSpace;

	/** Find follower agent component */
	UFollowerAgentComponent* FindFollowerAgent() const;

	/** Convert discrete action to ETacticalAction */
	ETacticalAction MapActionToTactical(int32 ActionIndex) const;
};
