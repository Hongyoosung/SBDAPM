// Copyright (c) 2023 Advanced Micro Devices, Inc. All Rights Reserved.

#pragma once
#include "CoreMinimal.h"
#include "GymConnectors/AbstractGymConnector.h"
#include "Common/LogSchola.h"

#include "Subsystem/SubsystemSettings/LaunchableScript.h"
#include "Subsystem/SubsystemSettings/CommunicatorSettings.h"
#include "Subsystem/SubsystemSettings/ScriptSettings.h"
#include "Interfaces/IPluginManager.h"
#include "SubsystemSettings.generated.h"

/**
 * @brief A class to make UERL subsystem settings available in the Game tab of project settings
 */
UCLASS(config = Engine, defaultconfig, meta = (DisplayName = "Schola Settings"))
class SCHOLA_API UScholaManagerSubsystemSettings : public UObject
{
	GENERATED_BODY()

public:
	UScholaManagerSubsystemSettings();

	/** The class of the gym connector to use */
	UPROPERTY(Config, EditAnywhere, Category = "General")
	TSubclassOf<UAbstractGymConnector> GymConnectorClass;

	/** Whether to run the script on play. Can be overriden by a CLI arg. */
	UPROPERTY(Config, EditAnywhere, Category = "General")
	bool bRunScriptOnPlay = false;

	/** The settings for the script */
	UPROPERTY(Config, EditAnywhere, meta = (EditCondition = "bRunScriptOnPlay", ShowOnlyInnerProperties), Category = "Script Settings")
	FScriptSettings ScriptSettings;

	/** The settings for communication */
	UPROPERTY(Config, EditAnywhere, meta = (ShowOnlyInnerProperties), Category = "Communicator Settings")
	FCommunicatorSettings CommunicatorSettings;

	FLaunchableScript GetScript() const;
};
