// Copyright (c) 2024-2025 Advanced Micro Devices, Inc. All Rights Reserved.

#pragma once
#include "CoreMinimal.h"

#include "CommunicatorSettings.generated.h"
/**
 * @brief A struct to hold settings relating to external communication (e.g. sockets)
 */
USTRUCT(BlueprintType)
struct SCHOLA_API FCommunicatorSettings
{
	GENERATED_BODY()

public:
	UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "Communicator Settings")
	FString Address = FString("127.0.0.1");

	UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "Communicator Settings")
	int Port = 50051;
	
	UPROPERTY(EditAnywhere, BlueprintReadOnly, meta=(ClampMin=1), Category = "Communicator Settings")
	int Timeout = 30;

};