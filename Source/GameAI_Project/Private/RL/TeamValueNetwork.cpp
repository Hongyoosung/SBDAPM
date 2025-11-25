// Copyright Epic Games, Inc. All Rights Reserved.

#include "RL/TeamValueNetwork.h"
#include "Misc/FileHelper.h"
#include "Misc/Paths.h"
#include "NNE.h"
#include "NNEModelData.h"
#include "NNERuntimeCPU.h"

UTeamValueNetwork::UTeamValueNetwork()
	: bIsInitialized(false)
	, bModelLoaded(false)
	, MaxAgents(10)
	, InputSize(0)
{
}

// ========================================
// Initialization
// ========================================

bool UTeamValueNetwork::Initialize(int32 InMaxAgents)
{
	MaxAgents = InMaxAgents;

	// Calculate input size: 40 team features + MaxAgents × 71 individual features
	InputSize = 40 + (MaxAgents * 71);

	bIsInitialized = true;

	UE_LOG(LogTemp, Log, TEXT("UTeamValueNetwork: Initialized with MaxAgents=%d, InputSize=%d"),
		MaxAgents, InputSize);

	return true;
}

bool UTeamValueNetwork::LoadModel(const FString& InModelPath)
{
	if (!bIsInitialized)
	{
		UE_LOG(LogTemp, Error, TEXT("UTeamValueNetwork: Must call Initialize() before LoadModel()"));
		return false;
	}

	ModelPath = InModelPath;

	// Resolve path - support both absolute and relative paths
	FString ResolvedPath = ModelPath;
	if (!FPaths::FileExists(ResolvedPath))
	{
		// Try relative to project content directory
		ResolvedPath = FPaths::ProjectContentDir() / ModelPath;
	}
	if (!FPaths::FileExists(ResolvedPath))
	{
		// Try relative to project saved directory
		ResolvedPath = FPaths::ProjectSavedDir() / ModelPath;
	}

	if (!FPaths::FileExists(ResolvedPath))
	{
		UE_LOG(LogTemp, Warning, TEXT("UTeamValueNetwork: Model file not found: %s (will use heuristic fallback)"), *ModelPath);
		bModelLoaded = false;
		return false;
	}

	UE_LOG(LogTemp, Log, TEXT("UTeamValueNetwork: Loading ONNX model from: %s"), *ResolvedPath);

	// Load model data from file
	TArray<uint8> ModelBytes;
	if (!FFileHelper::LoadFileToArray(ModelBytes, *ResolvedPath))
	{
		UE_LOG(LogTemp, Error, TEXT("UTeamValueNetwork: Failed to read model file"));
		bModelLoaded = false;
		return false;
	}

	// Get NNE runtime
	TWeakInterfacePtr<INNERuntimeCPU> RuntimeCPU = UE::NNE::GetRuntime<INNERuntimeCPU>(TEXT("NNERuntimeORTCpu"));
	if (!RuntimeCPU.IsValid())
	{
		UE_LOG(LogTemp, Error, TEXT("UTeamValueNetwork: NNERuntimeORTCpu not available. Using heuristic fallback."));
		bModelLoaded = false;
		return false;
	}

	// Create model from bytes
	TObjectPtr<UNNEModelData> NewModelData = NewObject<UNNEModelData>();
	NewModelData->Init(TEXT("Onnx"), ModelBytes, TMap<FString, TConstArrayView64<uint8>>());
	TSharedPtr<UE::NNE::IModelCPU> Model = RuntimeCPU->CreateModelCPU(NewModelData);
	if (!Model.IsValid())
	{
		UE_LOG(LogTemp, Error, TEXT("UTeamValueNetwork: Failed to create NNE model from ONNX data"));
		bModelLoaded = false;
		return false;
	}

	// Create model instance
	ModelInstance = Model->CreateModelInstanceCPU();
	if (!ModelInstance.IsValid())
	{
		UE_LOG(LogTemp, Error, TEXT("UTeamValueNetwork: Failed to create NNE model instance"));
		bModelLoaded = false;
		return false;
	}

	// Get input/output tensor info
	TConstArrayView<UE::NNE::FTensorDesc> InputDescs = ModelInstance->GetInputTensorDescs();
	TConstArrayView<UE::NNE::FTensorDesc> OutputDescs = ModelInstance->GetOutputTensorDescs();

	if (InputDescs.Num() < 1 || OutputDescs.Num() < 1)
	{
		UE_LOG(LogTemp, Error, TEXT("UTeamValueNetwork: Model must have at least 1 input and 1 output"));
		ModelInstance.Reset();
		bModelLoaded = false;
		return false;
	}

	// Log tensor info
	UE_LOG(LogTemp, Log, TEXT("UTeamValueNetwork: Model loaded successfully"));
	UE_LOG(LogTemp, Log, TEXT("  Input tensors: %d"), InputDescs.Num());
	UE_LOG(LogTemp, Log, TEXT("  Output tensors: %d"), OutputDescs.Num());

	// Setup buffers
	InputBuffer.SetNum(InputSize);
	OutputBuffer.SetNum(1);  // Single value output

	bModelLoaded = true;

	UE_LOG(LogTemp, Log, TEXT("UTeamValueNetwork: ONNX model ready for inference"));
	return true;
}

void UTeamValueNetwork::UnloadModel()
{
	bModelLoaded = false;
	ModelPath = TEXT("");

	// Reset NNE model instance
	if (ModelInstance.IsValid())
	{
		ModelInstance.Reset();
	}
	ModelData = nullptr;
	InputBuffer.Empty();
	OutputBuffer.Empty();

	UE_LOG(LogTemp, Log, TEXT("UTeamValueNetwork: Model unloaded"));
}

// ========================================
// Value Estimation
// ========================================

float UTeamValueNetwork::EvaluateState(const FTeamObservation& TeamObs)
{
	if (!bIsInitialized)
	{
		UE_LOG(LogTemp, Warning, TEXT("UTeamValueNetwork: Not initialized, returning 0.0"));
		return 0.0f;
	}

	// If model not loaded, return neutral value (heuristic fallback should be used by caller)
	if (!bModelLoaded || !ModelInstance.IsValid())
	{
		return 0.0f;
	}

	// Convert observation to features
	TArray<float> InputFeatures = TeamObservationToFeatures(TeamObs);

	// Forward pass
	float Value = ForwardPass(InputFeatures);

	return Value;
}

TArray<float> UTeamValueNetwork::EvaluateStateBatch(const TArray<FTeamObservation>& TeamObservations)
{
	TArray<float> Values;
	Values.Reserve(TeamObservations.Num());

	// For now, process sequentially (future: batch inference)
	for (const FTeamObservation& Obs : TeamObservations)
	{
		Values.Add(EvaluateState(Obs));
	}

	return Values;
}

// ========================================
// Neural Network Inference
// ========================================

float UTeamValueNetwork::ForwardPass(const TArray<float>& InputFeatures)
{
	if (!ModelInstance.IsValid())
	{
		UE_LOG(LogTemp, Warning, TEXT("UTeamValueNetwork: Model instance not valid, returning 0.0"));
		return 0.0f;
	}

	// Validate input size
	if (InputFeatures.Num() != InputSize)
	{
		UE_LOG(LogTemp, Warning, TEXT("UTeamValueNetwork: Input size mismatch (got %d, expected %d)"),
			InputFeatures.Num(), InputSize);
		return 0.0f;
	}

	// Copy input features to buffer
	InputBuffer = InputFeatures;

	// Create tensor shapes for binding
	UE::NNE::FTensorShape InputTensorShape = UE::NNE::FTensorShape::Make({ 1u, static_cast<uint32>(InputSize) });

	// Create input tensor bindings
	TArray<UE::NNE::FTensorBindingCPU> InputBindings;
	InputBindings.Add(UE::NNE::FTensorBindingCPU{
		InputBuffer.GetData(),
		static_cast<uint64>(InputBuffer.Num() * sizeof(float))
	});

	// Create output tensor bindings
	TArray<UE::NNE::FTensorBindingCPU> OutputBindings;
	OutputBindings.Add(UE::NNE::FTensorBindingCPU{
		OutputBuffer.GetData(),
		static_cast<uint64>(OutputBuffer.Num() * sizeof(float))
	});

	// Set input tensor shapes
	TArray<UE::NNE::FTensorShape> InputShapes;
	InputShapes.Add(InputTensorShape);

	UE::NNE::EResultStatus SetInputStatus = ModelInstance->SetInputTensorShapes(InputShapes);
	if (SetInputStatus != UE::NNE::EResultStatus::Ok)
	{
		UE_LOG(LogTemp, Warning, TEXT("UTeamValueNetwork: Failed to set input tensor shapes"));
		return 0.0f;
	}

	// Run inference
	UE::NNE::EResultStatus RunStatus = ModelInstance->RunSync(InputBindings, OutputBindings);

	if (RunStatus != UE::NNE::EResultStatus::Ok)
	{
		UE_LOG(LogTemp, Warning, TEXT("UTeamValueNetwork: Inference failed, returning 0.0"));
		return 0.0f;
	}

	// Return value (already in [-1, 1] from Tanh activation)
	return FMath::Clamp(OutputBuffer[0], -1.0f, 1.0f);
}

TArray<float> UTeamValueNetwork::TeamObservationToFeatures(const FTeamObservation& TeamObs)
{
	TArray<float> Features;
	Features.Reserve(InputSize);

	// Add team-level features (40 features)
	Features.Append(TeamObs.ToFeatureVector());

	// Add individual agent features (N × 71 features)
	for (int32 i = 0; i < TeamObs.IndividualObservations.Num() && i < MaxAgents; i++)
	{
		Features.Append(TeamObs.IndividualObservations[i].ToFeatureVector());
	}

	// Pad with zeros if fewer agents than MaxAgents
	int32 MissingAgents = MaxAgents - TeamObs.IndividualObservations.Num();
	if (MissingAgents > 0)
	{
		int32 PaddingSize = MissingAgents * 71;
		for (int32 i = 0; i < PaddingSize; i++)
		{
			Features.Add(0.0f);
		}
	}

	return Features;
}
