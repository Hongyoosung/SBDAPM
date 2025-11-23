// Copyright (c) 2024 Advanced Micro Devices, Inc. All Rights Reserved.

#include "Misc/AutomationTest.h"
#include "Tests/AutomationEditorCommon.h"

#include "Common/AbstractInteractor.h"
#include "Actuators/DebugActuators.h"
#include "Agent/AgentComponents/ActuatorComponent.h"

#define TestEqualExactFloat(TestMessage, Actual, Expected) TestEqual(TestMessage, (float)Actual, (float)Expected, 0.0001f)

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FAbstractInteractorGetOwnerInComponentTest, "Schola.AbstractInteractor.Get Owner From Component", EAutomationTestFlags::EditorContext | EAutomationTestFlags::ProductFilter)

bool FAbstractInteractorGetOwnerInComponentTest::RunTest(const FString& Parameters)
{
	UWorld* World = FAutomationEditorCommonUtils::CreateNewMap();

	UAbstractInteractor* Interactor = NewObject<UDebugBoxActuator>();
	TestNotNull("Interactor is not null", Interactor);

	AActor* Actor = World->SpawnActor<AActor>();
	TestNotNull("Actor is not null", Actor);

	UActorComponent* ComponentPtr = Actor->AddComponentByClass(UActuatorComponent::StaticClass(), false, FTransform(), false);
	TestNotNull("Component is not null", ComponentPtr);

	Actor->AddInstanceComponent(ComponentPtr);
	Interactor->Rename(nullptr, ComponentPtr);

	AActor* Owner = Interactor->TryGetOwner();
	TestNotNull("Owner is not null", Owner);
	TestEqual("Owner is Spawned Actor", Owner, Actor);

	return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FAbstractInteractorGetOwnerInPawnTest, "Schola.AbstractInteractor.Get Owner From Pawn", EAutomationTestFlags::EditorContext | EAutomationTestFlags::ProductFilter)

bool FAbstractInteractorGetOwnerInPawnTest::RunTest(const FString& Parameters)
{
	// Do the setup
	UWorld*				 World = FAutomationEditorCommonUtils::CreateNewMap();
	UAbstractInteractor* Interactor = NewObject<UDebugBoxActuator>();
	AActor*				 Actor = World->SpawnActor<AActor>();
	Interactor->Rename(nullptr, Actor);

	// Run the function
	AActor* Owner = Interactor->TryGetOwner();
	TestNotNull("Owner is not null", Owner);
	TestEqual("Owner is Spawned Actor", Owner, Actor);

	return true;
}