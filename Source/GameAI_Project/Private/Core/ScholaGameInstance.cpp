// ScholaGameInstance.cpp - Custom GameInstance implementation

#include "Core/ScholaGameInstance.h"
#include "Communicator/CommunicationManager.h"

UScholaGameInstance::UScholaGameInstance()
{
	// CommunicationManager will be created on-demand in GetCommunicationManager()
}

void UScholaGameInstance::Init()
{
	Super::Init();

	UE_LOG(LogTemp, Log, TEXT("[ScholaGameInstance] Initialized"));
}

void UScholaGameInstance::Shutdown()
{
	// Stop server before shutdown
	StopCommunicationServer();

	Super::Shutdown();

	UE_LOG(LogTemp, Log, TEXT("[ScholaGameInstance] Shutdown complete"));
}

UCommunicationManager* UScholaGameInstance::GetCommunicationManager()
{
	// Create CommunicationManager if it doesn't exist
	if (!CommunicationManager)
	{
		// Create new instance
		CommunicationManager = NewObject<UCommunicationManager>(this, UCommunicationManager::StaticClass());

		if (!CommunicationManager)
		{
			UE_LOG(LogTemp, Error, TEXT("[ScholaGameInstance] Failed to create CommunicationManager!"));
			UE_LOG(LogTemp, Error, TEXT("[ScholaGameInstance] Make sure Schola plugin is enabled in .uproject"));
			return nullptr;
		}

		UE_LOG(LogTemp, Log, TEXT("[ScholaGameInstance] CommunicationManager created"));
	}

	return CommunicationManager;
}

bool UScholaGameInstance::StartCommunicationServer(int32 Port)
{
	if (bServerRunning)
	{
		UE_LOG(LogTemp, Warning, TEXT("[ScholaGameInstance] Server already running on port %d"), ServerPort);
		return true;
	}

	// Get or create CommunicationManager
	UCommunicationManager* ComManager = GetCommunicationManager();
	if (!ComManager)
	{
		return false;
	}

	// Store port
	ServerPort = Port;

	// Override command line to set port (CommunicationManager reads from -ScholaPort)
	FString CommandLineOverride = FCommandLine::Get();
	if (!CommandLineOverride.Contains(TEXT("ScholaPort")))
	{
		CommandLineOverride += FString::Printf(TEXT(" -ScholaPort=%d"), Port);
		FCommandLine::Set(*CommandLineOverride);
		UE_LOG(LogTemp, Log, TEXT("[ScholaGameInstance] Set command line port to %d"), Port);
	}

	// Initialize CommunicationManager (sets up gRPC server)
	ComManager->Initialize();

	// Start gRPC backends (launches server on configured port)
	bool bSuccess = ComManager->StartBackends();
	if (!bSuccess)
	{
		UE_LOG(LogTemp, Error, TEXT("[ScholaGameInstance] Failed to start gRPC server on port %d"), Port);
		return false;
	}

	bServerRunning = true;
	UE_LOG(LogTemp, Warning, TEXT("[ScholaGameInstance] ✓ gRPC server started on port %d"), Port);
	UE_LOG(LogTemp, Warning, TEXT("[ScholaGameInstance] ✓ Ready for Python RLlib connection"));

	return true;
}

void UScholaGameInstance::StopCommunicationServer()
{
	if (!bServerRunning)
	{
		return;
	}

	if (CommunicationManager)
	{
		CommunicationManager->ShutdownServer();
		UE_LOG(LogTemp, Log, TEXT("[ScholaGameInstance] gRPC server stopped"));
	}

	bServerRunning = false;
}
