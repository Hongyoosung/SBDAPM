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
	const TArray<FStrategicCommand>& StrategicCommands,
	const TArray<ETacticalAction>& TacticalActions
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
	TArray<float> InputTensor = EncodeInput(CurrentState, StrategicCommands, TacticalActions);

	// Run inference (currently using heuristic)
	FTeamStateDelta LearnedDelta = PredictHeuristic(CurrentState, StrategicCommands, TacticalActions);

	// Apply stochastic sampling if enabled
	if (Config.bStochasticSampling)
	{
		ApplyStochasticSampling(LearnedDelta);
	}

	// Blend with heuristic if configured
	if (Config.HeuristicBlendWeight > 0.0f)
	{
		LearnedDelta = BlendWithHeuristic(LearnedDelta, CurrentState, StrategicCommands, TacticalActions);
	}

	// Validate output
	if (!ValidateOutput(LearnedDelta))
	{
		UE_LOG(LogTemp, Warning, TEXT("[WorldModel] Invalid prediction output, using fallback"));
		LearnedDelta = PredictHeuristic(CurrentState, StrategicCommands, TacticalActions);
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
	const TArray<TArray<FStrategicCommand>>& CommandSequence,
	const TArray<TArray<ETacticalAction>>& ActionSequence,
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
		// Get actions for this step
		const TArray<FStrategicCommand>& Commands = (Step < CommandSequence.Num()) ? CommandSequence[Step] : TArray<FStrategicCommand>();
		const TArray<ETacticalAction>& Actions = (Step < ActionSequence.Num()) ? ActionSequence[Step] : TArray<ETacticalAction>();

		// Predict next state
		FWorldModelPrediction Prediction = PredictNextState(CurrentState, Commands, Actions);
		Rollout.Add(Prediction);

		// Apply delta to current state for next iteration
		// TODO: Implement FTeamObservation::ApplyDelta()
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
	const TArray<FStrategicCommand>& Commands,
	const TArray<ETacticalAction>& Actions
)
{
	TArray<float> Input;

	// Flatten team observation
	TArray<float> StateFlat = State.Flatten();
	Input.Append(StateFlat);

	// Encode strategic commands
	for (const FStrategicCommand& Command : Commands)
	{
		FActionEncoding Encoding = FActionEncoding::EncodeStrategicCommand(Command);
		Input.Append(Encoding.Flatten());
	}

	// Encode tactical actions
	for (ETacticalAction Action : Actions)
	{
		TArray<float> ActionOneHot = FActionEncoding::EncodeTacticalAction(Action);
		Input.Append(ActionOneHot);
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

FTeamStateDelta UWorldModel::BlendWithHeuristic(
	const FTeamStateDelta& LearnedDelta,
	const FTeamObservation& State,
	const TArray<FStrategicCommand>& Commands,
	const TArray<ETacticalAction>& Actions
)
{
	FTeamStateDelta HeuristicDelta = PredictHeuristic(State, Commands, Actions);

	FTeamStateDelta BlendedDelta;
	const float LearnedWeight = 1.0f - Config.HeuristicBlendWeight;
	const float HeuristicWeight = Config.HeuristicBlendWeight;

	BlendedDelta.TeamHealthDelta = LearnedDelta.TeamHealthDelta * LearnedWeight + HeuristicDelta.TeamHealthDelta * HeuristicWeight;
	BlendedDelta.AliveCountDelta = FMath::RoundToInt(LearnedDelta.AliveCountDelta * LearnedWeight + HeuristicDelta.AliveCountDelta * HeuristicWeight);
	BlendedDelta.TeamCohesionDelta = LearnedDelta.TeamCohesionDelta * LearnedWeight + HeuristicDelta.TeamCohesionDelta * HeuristicWeight;
	BlendedDelta.PredictedKills = FMath::RoundToInt(LearnedDelta.PredictedKills * LearnedWeight + HeuristicDelta.PredictedKills * HeuristicWeight);
	BlendedDelta.PredictedDeaths = FMath::RoundToInt(LearnedDelta.PredictedDeaths * LearnedWeight + HeuristicDelta.PredictedDeaths * HeuristicWeight);
	BlendedDelta.PredictedDamageDealt = LearnedDelta.PredictedDamageDealt * LearnedWeight + HeuristicDelta.PredictedDamageDealt * HeuristicWeight;
	BlendedDelta.PredictedDamageTaken = LearnedDelta.PredictedDamageTaken * LearnedWeight + HeuristicDelta.PredictedDamageTaken * HeuristicWeight;
	BlendedDelta.DeltaTime = Config.TimeStepSeconds;
	BlendedDelta.Confidence = LearnedDelta.Confidence * LearnedWeight + HeuristicDelta.Confidence * HeuristicWeight;

	return BlendedDelta;
}

FTeamStateDelta UWorldModel::PredictHeuristic(
	const FTeamObservation& State,
	const TArray<FStrategicCommand>& Commands,
	const TArray<ETacticalAction>& Actions
)
{
	FTeamStateDelta Delta;

	// Heuristic prediction based on command types and current state
	int32 AssaultCount = 0;
	int32 DefendCount = 0;
	int32 RetreatCount = 0;

	for (const FStrategicCommand& Command : Commands)
	{
		switch (Command.CommandType)
		{
		case EStrategicCommandType::Assault:
			AssaultCount++;
			break;
		case EStrategicCommandType::Defend:
			DefendCount++;
			break;
		case EStrategicCommandType::Retreat:
			RetreatCount++;
			break;
		default:
			break;
		}
	}

	// Estimate combat outcome
	const float AssaultRatio = Commands.Num() > 0 ? (float)AssaultCount / Commands.Num() : 0.0f;
	const float DefendRatio = Commands.Num() > 0 ? (float)DefendCount / Commands.Num() : 0.0f;
	const float RetreatRatio = Commands.Num() > 0 ? (float)RetreatCount / Commands.Num() : 0.0f;

	// Predict damage dealt/taken based on aggression
	Delta.PredictedDamageDealt = AssaultRatio * 20.0f + DefendRatio * 5.0f;
	Delta.PredictedDamageTaken = AssaultRatio * 15.0f + DefendRatio * 5.0f + RetreatRatio * 2.0f;

	// Predict kills/deaths (simplified)
	const float DamageRatio = Delta.PredictedDamageDealt / FMath::Max(Delta.PredictedDamageTaken, 1.0f);
	Delta.PredictedKills = DamageRatio > 1.5f ? FMath::RandRange(0, 1) : 0;
	Delta.PredictedDeaths = DamageRatio < 0.7f ? FMath::RandRange(0, 1) : 0;

	// Predict team health change
	Delta.TeamHealthDelta = Delta.PredictedDamageDealt - Delta.PredictedDamageTaken;
	Delta.AliveCountDelta = Delta.PredictedKills - Delta.PredictedDeaths;

	// Predict cohesion change
	Delta.TeamCohesionDelta = DefendRatio * 0.1f - RetreatRatio * 0.2f;

	Delta.DeltaTime = Config.TimeStepSeconds;
	Delta.Confidence = 0.5f; // Heuristic has medium confidence

	return Delta;
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

// FActionEncoding implementations
FActionEncoding FActionEncoding::EncodeStrategicCommand(const FStrategicCommand& Command)
{
	FActionEncoding Encoding;

	// One-hot encode command type (5 types)
	Encoding.CommandTypeOneHot.Init(0.0f, 5);
	Encoding.CommandTypeOneHot[(int32)Command.CommandType] = 1.0f;

	// Normalize target location (assume world bounds ±5000)
	Encoding.TargetLocationNormalized = Command.TargetLocation / 5000.0f;
	Encoding.bHasTargetActor = Command.TargetActor != nullptr;

	return Encoding;
}

TArray<float> FActionEncoding::EncodeTacticalAction(ETacticalAction Action)
{
	// One-hot encode tactical action (16 actions)
	TArray<float> OneHot;
	OneHot.Init(0.0f, 16);
	OneHot[(int32)Action] = 1.0f;
	return OneHot;
}

TArray<float> FActionEncoding::Flatten() const
{
	TArray<float> Flat;

	// Command type one-hot (5)
	Flat.Append(CommandTypeOneHot);

	// Target location (3)
	Flat.Add(TargetLocationNormalized.X);
	Flat.Add(TargetLocationNormalized.Y);
	Flat.Add(TargetLocationNormalized.Z);

	// Has target actor (1)
	Flat.Add(bHasTargetActor ? 1.0f : 0.0f);

	// Tactical action one-hot (16)
	Flat.Append(TacticalActionOneHot);

	return Flat;
}
