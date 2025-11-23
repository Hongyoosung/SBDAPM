// Copyright (c) 2024 Advanced Micro Devices, Inc. All Rights Reserved.

#include "Misc/AutomationTest.h"
#include "Tests/AutomationEditorCommon.h"

#include "Observers/ObserverWrappers/FrameStacker.h"

#define TestEqualExactFloat(TestMessage, Actual, Expected) TestEqual(TestMessage, (float)Actual, (float)Expected, 0.0001f)


IMPLEMENT_SIMPLE_AUTOMATION_TEST(FFrameStackerWrapBoxObservationSpaceTest, "Schola.ObservationWrappers.FrameStacker.WrapBoxObservationSpace Test ", EAutomationTestFlags::EditorContext | EAutomationTestFlags::ProductFilter)

bool FFrameStackerWrapBoxObservationSpaceTest::RunTest(const FString& Parameters)
{
	UFrameStacker* Stacker = NewObject<UFrameStacker>();
	Stacker->MemorySize = 4;
	FBoxSpace	Space = { { 1.0, 2.0, 3.0 }, { 3.0, 4.0, 5.0 } };
    FBoxSpace WrappedSpace = Stacker->WrapBoxObservationSpace(Space);

	// Check that the Wrapped Space is 4 times the size of the original space
	TestEqual(TEXT("WrappedSpace.GetFlattenedSize() == Space.GetFlattenedSize() * 4"), WrappedSpace.GetFlattenedSize(), Space.GetFlattenedSize() * 4);

	// Check that Wrapped Space is Space repeated 4 times
	for (int i = 0; i < 4; i++)
	{
		for (int j = 0; j < 3; j++)
		{
			TestEqualExactFloat(FString::Printf(TEXT("WrappedSpace.Dimensions[%d].Low == Space.Dimensions[%d].Low"), i * 3 + j, j), WrappedSpace.Dimensions[i * 3 + j].Low, Space.Dimensions[j].Low);
			TestEqualExactFloat(FString::Printf(TEXT("WrappedSpace.Dimensions[%d].High == Space.Dimensions[%d].High"), i * 3 + j, j), WrappedSpace.Dimensions[i * 3 + j].High, Space.Dimensions[j].High);
		}
	}

	return true;
};

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FFrameStackerWrapBoxObservationTest, "Schola.ObservationWrappers.FrameStacker.WrapBoxObservation Test ", EAutomationTestFlags::EditorContext | EAutomationTestFlags::ProductFilter)

bool FFrameStackerWrapBoxObservationTest::RunTest(const FString& Parameters)
{
    UFrameStacker* Stacker = NewObject<UFrameStacker>();
    Stacker->MemorySize = 3;
    FBoxSpace Space = { {0.0}, {1.0} };
    Stacker->WrapBoxObservationSpace(Space);
    FBoxPoint OutputPoint;
    //Pass point 1 and check output
    OutputPoint = Stacker->WrapBoxObservation(FBoxPoint{1.0f});

    TestEqualExactFloat(TEXT("OutputPoint[0] == 0.0"), OutputPoint[0], 0.0);
    TestEqualExactFloat(TEXT("OutputPoint[1] == 0.0"), OutputPoint[1], 0.0);
    TestEqualExactFloat(TEXT("OutputPoint[2] == 1.0"), OutputPoint[2], 1.0);

    //Pass Point 2 and check Output
    OutputPoint = Stacker->WrapBoxObservation({0.75f});

    TestEqualExactFloat(TEXT("OutputPoint[0] == 0.0"), OutputPoint[0], 0.0);
    TestEqualExactFloat(TEXT("OutputPoint[1] == 1.0"), OutputPoint[1], 1.0);
    TestEqualExactFloat(TEXT("OutputPoint[2] == 0.75"), OutputPoint[2], 0.75);

    //Pass Point 3 and check Output
    OutputPoint = Stacker->WrapBoxObservation({0.5f});

    TestEqualExactFloat(TEXT("OutputPoint[0] == 1.0"), OutputPoint[0], 1.0);
    TestEqualExactFloat(TEXT("OutputPoint[1] == 0.75"), OutputPoint[1], 0.75);
    TestEqualExactFloat(TEXT("OutputPoint[2] == 0.5"), OutputPoint[2], 0.5);

    return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FFrameStackerResetTest, "Schola.ObservationWrappers.FrameStacker.Reset Test ", EAutomationTestFlags::EditorContext | EAutomationTestFlags::ProductFilter)

bool FFrameStackerResetTest::RunTest(const FString& Parameters)
{
    UFrameStacker* Stacker = NewObject<UFrameStacker>();
    Stacker->MemorySize = 3;
    FBoxSpace Space = FBoxSpace({0.0}, {1.0});
    Stacker->WrapBoxObservationSpace(Space);

    //Fill the FrameBuffer
    Stacker->WrapBoxObservation(FBoxPoint{1.0});
    Stacker->WrapBoxObservation(FBoxPoint{0.75});
    Stacker->WrapBoxObservation(FBoxPoint{0.5});

    //Reset the FrameStacker
    Stacker->Reset();
    //Send a new point and see if the history has been cleared
    FBoxPoint OutputPoint = Stacker->WrapBoxObservation(FBoxPoint{0.5});

    TestEqualExactFloat(TEXT("OutputPoint[0] == 0.0"), OutputPoint[0], 0.0);
    TestEqualExactFloat(TEXT("OutputPoint[1] == 0.0"), OutputPoint[1], 0.0);
    TestEqualExactFloat(TEXT("OutputPoint[2] == 0.5"), OutputPoint[2], 0.5);

    return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FFrameStackerGetBufferSizeTest, "Schola.ObservationWrappers.FrameStacker.GetBufferSize Test ", EAutomationTestFlags::EditorContext | EAutomationTestFlags::ProductFilter)

bool FFrameStackerGetBufferSizeTest::RunTest(const FString& Parameters)
{
    UFrameStacker* Stacker = NewObject<UFrameStacker>();
    Stacker->MemorySize = 3;
    FBoxSpace Space = FBoxSpace({0.0,2.0}, {1.0, 3.0});
    Stacker->WrapBoxObservationSpace(Space);

    TestEqual(TEXT("Stacker->GetBufferSize() == 6"), Stacker->GetBufferSize(), 6);

    return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FFrameStackerFillValueTest, "Schola.ObservationWrappers.FrameStacker.Fill Value Test", EAutomationTestFlags::EditorContext | EAutomationTestFlags::ProductFilter)

bool FFrameStackerFillValueTest::RunTest(const FString& Parameters)
{
    UFrameStacker* Stacker = NewObject<UFrameStacker>();
    Stacker->MemorySize = 3;
    Stacker->FillValue = 1.0;
    FBoxSpace Space = FBoxSpace({0.0}, {1.0});
    Stacker->WrapBoxObservationSpace(Space);

    FBoxPoint OutputPoint = Stacker->WrapBoxObservation(FBoxPoint({0.5}));

    TestEqualExactFloat(TEXT("OutputPoint[0] == 1.0"), OutputPoint[0], 1.0);
    TestEqualExactFloat(TEXT("OutputPoint[1] == 1.0"), OutputPoint[1], 1.0);
    TestEqualExactFloat(TEXT("OutputPoint[2] == 0.5"), OutputPoint[2], 0.5);

    return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FFrameStackerIndividualSpaceSizeTest, "Schola.ObservationWrappers.FrameStacker.Individual Space Size Test", EAutomationTestFlags::EditorContext | EAutomationTestFlags::ProductFilter)

bool FFrameStackerIndividualSpaceSizeTest::RunTest(const FString& Parameters)
{
    UFrameStacker* Stacker = NewObject<UFrameStacker>();
    Stacker->MemorySize = 3;
    FBoxSpace Space = FBoxSpace({0.0, 2.0}, {1.0, 3.0});
    Stacker->WrapBoxObservationSpace(Space);

    TestEqual(TEXT("Stacker->IndividualSpaceSize == 2"), Stacker->IndividualSpaceSize, 2);

    return true;
}