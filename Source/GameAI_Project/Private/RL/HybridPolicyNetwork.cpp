// Copyright Epic Games, Inc. All Rights Reserved.

#include "RL/HybridPolicyNetwork.h"
#include "Team/Objective.h"
#include "Observation/TeamObservation.h"
#include "Misc/FileHelper.h"
#include "Misc/Paths.h"

UHybridPolicyNetwork::UHybridPolicyNetwork()
	: bIsInitialized(false)
	, bModelLoaded(false)
{
}

// ========================================
// Initialization
// ========================================

bool UHybridPolicyNetwork::Initialize(const FRLPolicyConfig& InConfig)
{
	Config = InConfig;
	bIsInitialized = true;

	UE_LOG(LogTemp, Log, TEXT("HybridPolicyNetwork: Initialized (InputSize=%d, OutputSize=%d)"),
		Config.InputSize, Config.OutputSize);

	return true;
}

bool UHybridPolicyNetwork::LoadModel(const FString& InModelPath)
{
	ModelPath = InModelPath;

	// Resolve path - support both absolute and relative paths
	FString ResolvedPath = ModelPath;
	if (!FPaths::FileExists(ResolvedPath))
	{
		ResolvedPath = FPaths::ProjectContentDir() / ModelPath;
	}
	if (!FPaths::FileExists(ResolvedPath))
	{
		ResolvedPath = FPaths::ProjectSavedDir() / ModelPath;
	}

	if (!FPaths::FileExists(ResolvedPath))
	{
		UE_LOG(LogTemp, Error, TEXT("HybridPolicyNetwork: Model file not found: %s"), *ModelPath);
		bModelLoaded = false;
		return false;
	}

	UE_LOG(LogTemp, Log, TEXT("HybridPolicyNetwork: Loading ONNX model from: %s"), *ResolvedPath);

	// Load model data from file
	TArray<uint8> ModelBytes;
	if (!FFileHelper::LoadFileToArray(ModelBytes, *ResolvedPath))
	{
		UE_LOG(LogTemp, Error, TEXT("HybridPolicyNetwork: Failed to read model file"));
		bModelLoaded = false;
		return false;
	}

	// Get NNE runtime
	TWeakInterfacePtr<INNERuntimeCPU> RuntimeCPU = UE::NNE::GetRuntime<INNERuntimeCPU>(TEXT("NNERuntimeORTCpu"));
	if (!RuntimeCPU.IsValid())
	{
		UE_LOG(LogTemp, Error, TEXT("HybridPolicyNetwork: NNERuntimeORTCpu not available"));
		bModelLoaded = false;
		return false;
	}

	// Create model from bytes
	TObjectPtr<UNNEModelData> NewModelData = NewObject<UNNEModelData>();
	NewModelData->Init(TEXT("Onnx"), ModelBytes, TMap<FString, TConstArrayView64<uint8>>());
	TSharedPtr<UE::NNE::IModelCPU> Model = RuntimeCPU->CreateModelCPU(NewModelData);
	if (!Model.IsValid())
	{
		UE_LOG(LogTemp, Error, TEXT("HybridPolicyNetwork: Failed to create NNE model"));
		bModelLoaded = false;
		return false;
	}

	// Create model instance
	ModelInstance = Model->CreateModelInstanceCPU();
	if (!ModelInstance.IsValid())
	{
		UE_LOG(LogTemp, Error, TEXT("HybridPolicyNetwork: Failed to create model instance"));
		bModelLoaded = false;
		return false;
	}

	// Get input/output tensor info
	TConstArrayView<UE::NNE::FTensorDesc> InputDescs = ModelInstance->GetInputTensorDescs();
	TConstArrayView<UE::NNE::FTensorDesc> OutputDescs = ModelInstance->GetOutputTensorDescs();

	if (InputDescs.Num() < 1 || OutputDescs.Num() < 2)
	{
		UE_LOG(LogTemp, Error, TEXT("HybridPolicyNetwork: Model must have 1 input and 2 outputs (action + prior)"));
		ModelInstance.Reset();
		bModelLoaded = false;
		return false;
	}

	// Setup buffers
	InputBuffer.SetNum(Config.InputSize);
	ActionLogitsBuffer.SetNum(Config.OutputSize);  // 8 action dims
	PriorLogitsBuffer.SetNum(7);  // 7 objective types

	bModelLoaded = true;
	UE_LOG(LogTemp, Log, TEXT("HybridPolicyNetwork: ONNX model loaded successfully"));
	return true;
}

void UHybridPolicyNetwork::UnloadModel()
{
	bModelLoaded = false;
	ModelPath = TEXT("");

	if (ModelInstance.IsValid())
	{
		ModelInstance.Reset();
	}
	ModelData = nullptr;
	InputBuffer.Empty();
	ActionLogitsBuffer.Empty();
	PriorLogitsBuffer.Empty();

	UE_LOG(LogTemp, Log, TEXT("HybridPolicyNetwork: Model unloaded"));
}

// ========================================
// Policy Head (Immediate Action Selection)
// ========================================

FTacticalAction UHybridPolicyNetwork::GetAction(const FObservationElement& Observation, UObjective* CurrentObjective)
{
	if (!IsReady())
	{
		UE_LOG(LogTemp, Warning, TEXT("HybridPolicyNetwork: Not ready, returning default action"));
		return FTacticalAction();
	}

	// Build input features
	TArray<float> InputFeatures = Observation.ToFeatureVector();

	// Add objective embedding (7 features)
	TArray<float> ObjectiveEmbed;
	ObjectiveEmbed.Init(0.0f, 7);
	if (CurrentObjective)
	{
		int32 ObjTypeIndex = static_cast<int32>(CurrentObjective->Type);
		if (ObjTypeIndex >= 0 && ObjTypeIndex < 7)
		{
			ObjectiveEmbed[ObjTypeIndex] = 1.0f;
		}
	}
	InputFeatures.Append(ObjectiveEmbed);

	// Forward pass
	TArray<float> ActionLogits, PriorLogits;
	ForwardPass(InputFeatures, ActionLogits, PriorLogits);

	// Convert to tactical action (only use policy head output)
	return ActionLogitsToTacticalAction(ActionLogits);
}

// ========================================
// Prior Head (MCTS Initialization)
// ========================================

TArray<float> UHybridPolicyNetwork::GetObjectivePriors(const FTeamObservation& TeamObs)
{
	if (!IsReady())
	{
		UE_LOG(LogTemp, Warning, TEXT("HybridPolicyNetwork: Not ready, returning uniform priors"));
		TArray<float> UniformPriors;
		UniformPriors.Init(1.0f / 7.0f, 7);
		return UniformPriors;
	}

	// Build input from team observation
	TArray<float> InputFeatures = TeamObs.ToFeatureVector();

	// Forward pass
	TArray<float> ActionLogits, PriorLogits;
	ForwardPass(InputFeatures, ActionLogits, PriorLogits);

	// Convert to probabilities (only use prior head output)
	return PriorLogitsToProbabilities(PriorLogits);
}

void UHybridPolicyNetwork::GetActionAndPriors(
	const FObservationElement& Observation,
	const FTeamObservation& TeamObs,
	FTacticalAction& OutAction,
	TArray<float>& OutPriors)
{
	// Optimized single forward pass for both outputs
	// Prepare input combining individual and team observations
	TArray<float> InputFeatures = Observation.ToFeatureVector();
	InputFeatures.Append(TeamObs.Flatten());

	// Single forward pass through dual-head network
	TArray<float> ActionLogits;
	TArray<float> PriorLogits;
	ForwardPass(InputFeatures, ActionLogits, PriorLogits);

	// Decode action from logits
	OutAction = ActionLogitsToTacticalAction(ActionLogits);

	// Store priors directly
	OutPriors = PriorLogits;
}

// ========================================
// Neural Network Inference (Private)
// ========================================

void UHybridPolicyNetwork::ForwardPass(
	const TArray<float>& InputFeatures,
	TArray<float>& OutActionLogits,
	TArray<float>& OutPriorLogits)
{
	if (!ModelInstance.IsValid())
	{
		UE_LOG(LogTemp, Warning, TEXT("HybridPolicyNetwork: Model instance not valid"));
		OutActionLogits.Init(0.0f, 8);
		OutPriorLogits.Init(0.0f, 7);
		return;
	}

	// Copy input
	InputBuffer = InputFeatures;

	// Create tensor shapes
	UE::NNE::FTensorShape InputTensorShape = UE::NNE::FTensorShape::Make({ 1u, static_cast<uint32>(InputBuffer.Num()) });

	// Create input bindings
	TArray<UE::NNE::FTensorBindingCPU> InputBindings;
	InputBindings.Add(UE::NNE::FTensorBindingCPU{
		InputBuffer.GetData(),
		static_cast<uint64>(InputBuffer.Num() * sizeof(float))
	});

	// Create output bindings
	TArray<UE::NNE::FTensorBindingCPU> OutputBindings;
	OutputBindings.Add(UE::NNE::FTensorBindingCPU{
		ActionLogitsBuffer.GetData(),
		static_cast<uint64>(ActionLogitsBuffer.Num() * sizeof(float))
	});
	OutputBindings.Add(UE::NNE::FTensorBindingCPU{
		PriorLogitsBuffer.GetData(),
		static_cast<uint64>(PriorLogitsBuffer.Num() * sizeof(float))
	});

	// Set input tensor shapes
	TArray<UE::NNE::FTensorShape> InputShapes;
	InputShapes.Add(InputTensorShape);

	UE::NNE::EResultStatus SetInputStatus = ModelInstance->SetInputTensorShapes(InputShapes);
	if (SetInputStatus != UE::NNE::EResultStatus::Ok)
	{
		UE_LOG(LogTemp, Warning, TEXT("HybridPolicyNetwork: Failed to set input tensor shapes"));
		OutActionLogits.Init(0.0f, 8);
		OutPriorLogits.Init(0.0f, 7);
		return;
	}

	// Run inference
	UE::NNE::EResultStatus RunStatus = ModelInstance->RunSync(InputBindings, OutputBindings);
	if (RunStatus != UE::NNE::EResultStatus::Ok)
	{
		UE_LOG(LogTemp, Warning, TEXT("HybridPolicyNetwork: Inference failed"));
		OutActionLogits.Init(0.0f, 8);
		OutPriorLogits.Init(0.0f, 7);
		return;
	}

	// Copy outputs
	OutActionLogits = ActionLogitsBuffer;
	OutPriorLogits = PriorLogitsBuffer;
}

FTacticalAction UHybridPolicyNetwork::ActionLogitsToTacticalAction(const TArray<float>& ActionLogits)
{
	FTacticalAction Action;

	if (ActionLogits.Num() >= 8)
	{
		// Continuous actions (already in [-1,1] or [0,1] range)
		Action.MoveDirection.X = FMath::Clamp(ActionLogits[0], -1.0f, 1.0f);
		Action.MoveDirection.Y = FMath::Clamp(ActionLogits[1], -1.0f, 1.0f);
		Action.MoveSpeed = FMath::Clamp(ActionLogits[2], 0.0f, 1.0f);
		Action.LookDirection.X = FMath::Clamp(ActionLogits[3], -1.0f, 1.0f);
		Action.LookDirection.Y = FMath::Clamp(ActionLogits[4], -1.0f, 1.0f);

		// Discrete actions (sigmoid threshold)
		Action.bFire = ActionLogits[5] > 0.5f;
		Action.bCrouch = ActionLogits[6] > 0.5f;
		Action.bUseAbility = ActionLogits[7] > 0.5f;
	}

	return Action;
}

TArray<float> UHybridPolicyNetwork::PriorLogitsToProbabilities(const TArray<float>& PriorLogits)
{
	TArray<float> Probabilities;

	if (PriorLogits.Num() != 7)
	{
		UE_LOG(LogTemp, Warning, TEXT("HybridPolicyNetwork: Invalid prior logits size %d, expected 7"),
			PriorLogits.Num());
		Probabilities.Init(1.0f / 7.0f, 7);
		return Probabilities;
	}

	Probabilities.SetNum(7);

	// Softmax
	float MaxLogit = -MAX_FLT;
	for (float Logit : PriorLogits)
	{
		if (Logit > MaxLogit)
		{
			MaxLogit = Logit;
		}
	}

	float SumExp = 0.0f;
	for (int32 i = 0; i < PriorLogits.Num(); ++i)
	{
		Probabilities[i] = FMath::Exp(PriorLogits[i] - MaxLogit);
		SumExp += Probabilities[i];
	}

	// Normalize
	for (int32 i = 0; i < Probabilities.Num(); ++i)
	{
		Probabilities[i] /= SumExp;
	}

	return Probabilities;
}
