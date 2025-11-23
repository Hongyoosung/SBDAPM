// Copyright (c) 2024 Advanced Micro Devices, Inc. All Rights Reserved.

#include "Misc/AutomationTest.h"
#include "Tests/AutomationEditorCommon.h"

#include "Observers/ObserverWrappers/HardNormalizer.h"

#define TestEqualExactFloat(TestMessage, Actual, Expected) TestEqual(TestMessage, (float)Actual, (float)Expected, 0.0001f)

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FHardNormalizerWrapBoxObservationTest, "Schola.ObservationWrappers.HardNormalizer.WrapBoxObservation Test ", EAutomationTestFlags::EditorContext | EAutomationTestFlags::ProductFilter)

bool FHardNormalizerWrapBoxObservationTest::RunTest(const FString& Parameters)
{
    UHardNormalizer* Normalizer = NewObject<UHardNormalizer>();
    FBoxPoint Point ={1.5,1.0,5.0};
	FBoxSpace		 Space = { { 0.0, 1.0, 2.0 }, { 3.0, 4.0, 5.0 } };
	Normalizer->WrapBoxObservationSpace(Space);

    FBoxPoint WrappedPoint = Normalizer->WrapBoxObservation(Point);

    TestEqualExactFloat(TEXT("WrappedPoint[0] == 0.5"), WrappedPoint[0], 0.5);
    TestEqualExactFloat(TEXT("WrappedPoint[1] == 0.0"), WrappedPoint[1], 0.0);
    TestEqualExactFloat(TEXT("WrappedPoint[2] == 1.0"), WrappedPoint[2], 1.0);

    return true;
}

