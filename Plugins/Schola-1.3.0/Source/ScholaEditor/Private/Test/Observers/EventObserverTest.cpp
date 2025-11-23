// Copyright (c) 2024 Advanced Micro Devices, Inc. All Rights Reserved.

#include "Misc/AutomationTest.h"
#include "Tests/AutomationEditorCommon.h"

#include "Observers/EventObserver.h"

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FEventObserverTest, "Schola.Observers.EventObserver.Initial Values Test", EAutomationTestFlags::EditorContext | EAutomationTestFlags::ProductFilter)

bool FEventObserverTest::RunTest(const FString& Parameters)
{
    UEventObserver* EventObserver = NewObject<UEventObserver>();
    TestNotNull("EventObserver is not null", EventObserver);

    // Test default values
    TestFalse("EventObserver bEventTriggered is false", EventObserver->bEventTriggered);
    TestTrue("EventObserver bAutoClearEventFlag is true", EventObserver->bAutoClearEventFlag);

    // Test GetObservationSpace
    FBinarySpace ObservationSpace = EventObserver->GetObservationSpace();
	TestEqual("EventObserver GetObservationSpace size is 1", ObservationSpace.GetFlattenedSize(), 1);

    return true;
}

//Test Clearing the Event

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FEventObserverClearEventTest, "Schola.Observers.EventObserver.ClearEvent Test", EAutomationTestFlags::EditorContext | EAutomationTestFlags::ProductFilter)

bool FEventObserverClearEventTest::RunTest(const FString& Parameters)
{
    UEventObserver* EventObserver = NewObject<UEventObserver>();
    TestNotNull("EventObserver is not null", EventObserver);

    EventObserver->TriggerEvent();
    TestTrue("EventObserver bEventTriggered is true after TriggerEvent", EventObserver->bEventTriggered);

    EventObserver->ClearEvent();
    TestFalse("EventObserver bEventTriggered is false after ClearEvent", EventObserver->bEventTriggered);

    return true;
}

//Test Collect Observations

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FEventObserverCollectObservationsTest, "Schola.Observers.EventObserver.CollectObservations Test", EAutomationTestFlags::EditorContext | EAutomationTestFlags::ProductFilter)

bool FEventObserverCollectObservationsTest::RunTest(const FString& Parameters)
{
    UEventObserver* EventObserver = NewObject<UEventObserver>();
    TestNotNull("EventObserver is not null", EventObserver);

    EventObserver->TriggerEvent();

    // Test CollectObservations
    FBinaryPoint ObservationOne;
    EventObserver->CollectObservations(ObservationOne);
    TestTrue("EventObserver CollectObservations is True", ObservationOne.Values[0]);

    EventObserver->ClearEvent();
    FBinaryPoint ObservationTwo;
    EventObserver->CollectObservations(ObservationTwo);
    TestFalse("EventObserver CollectObservations is False", ObservationTwo.Values[0]);

    return true;
}

//Test autoclearevent flag

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FEventObserverAutoClearTest, "Schola.Observers.EventObserver.AutoClear Test", EAutomationTestFlags::EditorContext | EAutomationTestFlags::ProductFilter)

bool FEventObserverAutoClearTest::RunTest(const FString& Parameters)
{
    UEventObserver* EventObserver = NewObject<UEventObserver>();
    TestNotNull("EventObserver is not null", EventObserver);

    EventObserver->bAutoClearEventFlag = true;

    //Trigger the event
    EventObserver->TriggerEvent();
    
    FBinaryPoint ObservationOne;
    EventObserver->CollectObservations(ObservationOne);
    TestTrue("EventObserver CollectObservations before AutoReset is True", ObservationOne.Values[0]);

    FBinaryPoint ObservationTwo;
    EventObserver->CollectObservations(ObservationTwo);
    TestFalse("EventObserver CollectObservations after AutoReset is False", ObservationTwo.Values[0]);

    return true;
}

