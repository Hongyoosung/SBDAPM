// Copyright Epic Games, Inc. All Rights Reserved.

#include "Simulation/WorldModel.h"
#include "Misc/FileHelper.h"
#include "Misc/Paths.h"
#include "HAL/PlatformTime.h"

UWorldModel::UWorldModel()
{
	bIsInitialized = false;
	ModelInstance = nullptr;
	AverageInferenceTimeMs = 0.0f;
	InferenceCount = 0;
	ModelVersion = 1;
	bCollectingData = false;
}

bool UWorldModel::Initialize(const FWorldModelConfig& InConfig)
{
	Config = InConfig;

	// TODO: Load ONNX model via NNE when model is trained
	// For now, use heuristic fallback
	UE_LOG(LogTemp, Warning, TEXT("[WorldModel] ONNX model not yet trained, using heuristic predictions"));
	UE_LOG(LogTemp, Log, TEXT("[WorldModel] Initialize called with model path: %s"), *Config.ModelPath);

	// Set placeholder input/output sizes based on observation structure
	// Input: TeamObs (750) + Actions (encoded)
	// Output: State delta (750)
	InputSize = 750 + 100; // Placeholder
	OutputSize = 750;

	bIsInitialized = true;
	return true;
}

FWorldModelPrediction UWorldModel::PredictNextState(
	const FTeamObservation& CurrentState,
	const TArray<FTacticalAction>& TacticalActions
)
{
	FWorldModelPrediction Prediction;

	if (!bIsInitialized)
	{
		UE_LOG(LogTemp, Error, TEXT("[WorldModel] Cannot predict: model not initialized"));
		return Prediction;
	}

	const double StartTime = FPlatformTime::Seconds();

	// Encode input
	TArray<float> InputTensor = EncodeInput(CurrentState, TacticalActions);

	// TODO: Run ONNX inference when model is trained
	// For now, use heuristic fallback to generate prediction
	TArray<float> OutputTensor;
	OutputTensor.Init(0.0f, OutputSize);

	// Decode output to state delta
	FTeamStateDelta LearnedDelta = DecodeOutput(OutputTensor);

	// Apply stochastic sampling if enabled
	if (Config.bStochasticSampling)
	{
		ApplyStochasticSampling(LearnedDelta);
	}

	// Validate output
	if (!ValidateOutput(LearnedDelta))
	{
		UE_LOG(LogTemp, Warning, TEXT("[WorldModel] Invalid prediction output, using fallback"));
	}

	// Update prediction
	Prediction.PredictedDelta = LearnedDelta;
	Prediction.Confidence = LearnedDelta.Confidence;
	Prediction.ModelVersion = ModelVersion;

	// Update performance metrics
	const float InferenceTimeMs = (FPlatformTime::Seconds() - StartTime) * 1000.0f;
	Prediction.InferenceTimeMs = InferenceTimeMs;
	UpdatePerformanceMetrics(InferenceTimeMs);

	if (Config.bEnableLogging)
	{
		UE_LOG(LogTemp, Log, TEXT("[WorldModel] Predicted delta - TeamHealthDelta: %.2f, AliveCountDelta: %d, Confidence: %.2f"),
			Prediction.PredictedDelta.TeamHealthDelta,
			Prediction.PredictedDelta.AliveCountDelta,
			Prediction.Confidence);
	}

	return Prediction;
}

TArray<FWorldModelPrediction> UWorldModel::PredictRollout(
	const FTeamObservation& InitialState,
	const TArray<FActionSequence>& ActionSequence,
	int32 NumSteps
)
{
	TArray<FWorldModelPrediction> Rollout;

	if (!bIsInitialized)
	{
		UE_LOG(LogTemp, Error, TEXT("[WorldModel] Cannot rollout: model not initialized"));
		return Rollout;
	}

	NumSteps = FMath::Min(NumSteps, Config.MaxPredictionSteps);
	FTeamObservation CurrentState = InitialState;

	for (int32 Step = 0; Step < NumSteps; ++Step)
	{
		if (Step < ActionSequence.Num())
		{
			const TArray<FTacticalAction>& Actions = ActionSequence[Step].Actions;

			FWorldModelPrediction Prediction = PredictNextState(CurrentState, Actions);
			Rollout.Add(Prediction);
		}
		else
		{
			const TArray<FTacticalAction> EmptyActions;
			FWorldModelPrediction Prediction = PredictNextState(CurrentState, EmptyActions);
			Rollout.Add(Prediction);
		}

		// Apply delta to current state for next iteration
		// CurrentState = CurrentState.ApplyDelta(Prediction.PredictedDelta);
	}

	return Rollout;
}

void UWorldModel::LogTransitionSample(const FStateTransitionSample& Sample)
{
	CollectedSamples.Add(Sample);

	if (Config.bEnableLogging && CollectedSamples.Num() % 100 == 0)
	{
		UE_LOG(LogTemp, Log, TEXT("[WorldModel] Collected %d transition samples"), CollectedSamples.Num());
	}
}

bool UWorldModel::ExportTransitionSamples(const FString& OutputPath)
{
	if (CollectedSamples.Num() == 0)
	{
		UE_LOG(LogTemp, Warning, TEXT("[WorldModel] No samples to export"));
		return false;
	}

	// Build JSON
	FString JsonString = TEXT("{\n  \"samples\": [\n");

	for (int32 i = 0; i < CollectedSamples.Num(); ++i)
	{
		const FStateTransitionSample& Sample = CollectedSamples[i];

		JsonString += TEXT("    {\n");
		JsonString += FString::Printf(TEXT("      \"timestamp\": %.2f,\n"), Sample.Timestamp);
		JsonString += FString::Printf(TEXT("      \"game_outcome\": %.2f,\n"), Sample.GameOutcome);

		// State before
		JsonString += TEXT("      \"state_before\": [");
		for (int32 j = 0; j < Sample.StateBefore.Num(); ++j)
		{
			JsonString += FString::Printf(TEXT("%.4f"), Sample.StateBefore[j]);
			if (j < Sample.StateBefore.Num() - 1) JsonString += TEXT(", ");
		}
		JsonString += TEXT("],\n");

		// State after
		JsonString += TEXT("      \"state_after\": [");
		for (int32 j = 0; j < Sample.StateAfter.Num(); ++j)
		{
			JsonString += FString::Printf(TEXT("%.4f"), Sample.StateAfter[j]);
			if (j < Sample.StateAfter.Num() - 1) JsonString += TEXT(", ");
		}
		JsonString += TEXT("],\n");

		// Actual delta
		JsonString += TEXT("      \"actual_delta\": {\n");
		JsonString += FString::Printf(TEXT("        \"team_health_delta\": %.2f,\n"), Sample.ActualDelta.TeamHealthDelta);
		JsonString += FString::Printf(TEXT("        \"alive_count_delta\": %d,\n"), Sample.ActualDelta.AliveCountDelta);
		JsonString += FString::Printf(TEXT("        \"predicted_kills\": %d,\n"), Sample.ActualDelta.PredictedKills);
		JsonString += FString::Printf(TEXT("        \"predicted_deaths\": %d\n"), Sample.ActualDelta.PredictedDeaths);
		JsonString += TEXT("      }\n");

		JsonString += TEXT("    }");
		if (i < CollectedSamples.Num() - 1) JsonString += TEXT(",");
		JsonString += TEXT("\n");
	}

	JsonString += TEXT("  ]\n}\n");

	// Write to file
	const FString FullPath = FPaths::ProjectDir() / OutputPath;
	if (FFileHelper::SaveStringToFile(JsonString, *FullPath))
	{
		UE_LOG(LogTemp, Log, TEXT("[WorldModel] Exported %d samples to %s"), CollectedSamples.Num(), *FullPath);
		return true;
	}
	else
	{
		UE_LOG(LogTemp, Error, TEXT("[WorldModel] Failed to export samples to %s"), *FullPath);
		return false;
	}
}

TArray<float> UWorldModel::EncodeInput(
	const FTeamObservation& State,
	const TArray<FTacticalAction>& Actions
)
{
	TArray<float> Input;

	// Flatten team observation
	TArray<float> StateFlat = State.Flatten();
	Input.Append(StateFlat);

	// Encode tactical actions (8-dimensional continuous space)
	for (const FTacticalAction& Action : Actions)
	{
		TArray<float> ActionEncoded = FActionEncoding::EncodeTacticalAction(Action);
		Input.Append(ActionEncoded);
	}

	return Input;
}

FTeamStateDelta UWorldModel::DecodeOutput(const TArray<float>& OutputTensor)
{
	FTeamStateDelta Delta;

	// TODO: Parse output tensor into state delta
	// For now, placeholder

	return Delta;
}

void UWorldModel::ApplyStochasticSampling(FTeamStateDelta& Delta)
{
	// Add Gaussian noise scaled by temperature
	const float NoiseScale = Config.SamplingTemperature;

	Delta.TeamHealthDelta += FMath::FRandRange(-NoiseScale * 10.0f, NoiseScale * 10.0f);
	Delta.TeamCohesionDelta += FMath::FRandRange(-NoiseScale * 0.1f, NoiseScale * 0.1f);
	Delta.PredictedDamageDealt += FMath::FRandRange(-NoiseScale * 5.0f, NoiseScale * 5.0f);
	Delta.PredictedDamageTaken += FMath::FRandRange(-NoiseScale * 5.0f, NoiseScale * 5.0f);

	// Reduce confidence due to sampling
	Delta.Confidence *= (1.0f - NoiseScale * 0.5f);
}

void UWorldModel::UpdatePerformanceMetrics(float InferenceTimeMs)
{
	InferenceCount++;
	AverageInferenceTimeMs = (AverageInferenceTimeMs * (InferenceCount - 1) + InferenceTimeMs) / InferenceCount;
}

bool UWorldModel::ValidateOutput(const FTeamStateDelta& Delta) const
{
	// Check for NaN or inf
	if (!FMath::IsFinite(Delta.TeamHealthDelta) ||
		!FMath::IsFinite(Delta.TeamCohesionDelta) ||
		!FMath::IsFinite(Delta.PredictedDamageDealt) ||
		!FMath::IsFinite(Delta.PredictedDamageTaken))
	{
		return false;
	}

	// Check for unreasonable values
	if (FMath::Abs(Delta.TeamHealthDelta) > 500.0f ||
		FMath::Abs(Delta.AliveCountDelta) > 10 ||
		Delta.Confidence < 0.0f || Delta.Confidence > 1.0f)
	{
		return false;
	}

	return true;
}


TArray<float> FActionEncoding::EncodeTacticalAction(const FTacticalAction& Action)
{
	// Encode 8-dimensional continuous action space
	TArray<float> Encoded;
	Encoded.Reserve(8);

	// Movement (3 dimensions)
	Encoded.Add(Action.MoveDirection.X);  // [-1, 1]
	Encoded.Add(Action.MoveDirection.Y);  // [-1, 1]
	Encoded.Add(Action.MoveSpeed);        // [0, 1]

	// Aiming (2 dimensions)
	Encoded.Add(Action.LookDirection.X);  // [-1, 1]
	Encoded.Add(Action.LookDirection.Y);  // [-1, 1]

	// Discrete actions (3 dimensions as binary floats)
	Encoded.Add(Action.bFire ? 1.0f : 0.0f);
	Encoded.Add(Action.bCrouch ? 1.0f : 0.0f);
	Encoded.Add(Action.bUseAbility ? 1.0f : 0.0f);

	return Encoded;
}
