// Copyright (c) 2024 Advanced Micro Devices, Inc. All Rights Reserved.

#include "Misc/AutomationTest.h"
#include "Tests/AutomationEditorCommon.h"

#include "Observers/ObserverWrappers/ObservationClipper.h"

#define TestEqualExactFloat(TestMessage, Actual, Expected) TestEqual(TestMessage, (float)Actual, (float)Expected, 0.0001f)

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FObservationClipperWrapBoxObservationSpaceTest, "Schola.ObservationWrappers.ObservationClipper.Wrap Space Test ", EAutomationTestFlags::EditorContext | EAutomationTestFlags::ProductFilter)

bool FObservationClipperWrapBoxObservationSpaceTest::RunTest(const FString& Parameters)
{
    UObservationClipper* Clipper = NewObject<UObservationClipper>();
    FBoxSpace Space = FBoxSpace({1.0,2.0,3.0}, {3.0, 4.0, 5.0});
    FBoxSpace WrappedSpace = Clipper->WrapBoxObservationSpace(Space);

    //Dimension 0
    TestEqualExactFloat(TEXT("WrappedSpace.Dimensions[0].Low == Space.Dimensions[0].Low"), WrappedSpace.Dimensions[0].Low, Space.Dimensions[0].Low);
	TestEqualExactFloat(TEXT("WrappedSpace.Dimensions[0].High == Space.Dimensions[0].High"), WrappedSpace.Dimensions[0].High, Space.Dimensions[0].High);

    //Dimension 1
    TestEqualExactFloat(TEXT("WrappedSpace.Dimensions[1].Low == Space.Dimensions[1].Low"), WrappedSpace.Dimensions[1].Low, Space.Dimensions[1].Low);
    TestEqualExactFloat(TEXT("WrappedSpace.Dimensions[1].High == Space.Dimensions[1].High"), WrappedSpace.Dimensions[1].High, Space.Dimensions[1].High);

    //Dimension 2
    TestEqualExactFloat(TEXT("WrappedSpace.Dimensions[2].Low == Space.Dimensions[2].Low"), WrappedSpace.Dimensions[2].Low, Space.Dimensions[2].Low);
    TestEqualExactFloat(TEXT("WrappedSpace.Dimensions[2].High == Space.Dimensions[2].High"), WrappedSpace.Dimensions[2].High, Space.Dimensions[2].High);

    return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FObservationClipperWrapValuesInSpaceTest, "Schola.ObservationWrappers.ObservationClipper.Wrap Values In Space Test ", EAutomationTestFlags::EditorContext | EAutomationTestFlags::ProductFilter)

bool FObservationClipperWrapValuesInSpaceTest::RunTest(const FString& Parameters)
{
    UObservationClipper* Clipper = NewObject<UObservationClipper>();
    FBoxPoint Point = FBoxPoint({1.0,2.0,3.0});
    FBoxSpace Space = FBoxSpace({0.0,1.0,2.0}, {3.0, 4.0, 5.0});
    Clipper->WrapBoxObservationSpace(Space);
    FBoxPoint WrappedPoint = Clipper->WrapBoxObservation(Point);

    TestEqualExactFloat(TEXT("WrappedPoint.Values[0] == 1.0"), WrappedPoint.Values[0], 1.0);
	TestEqualExactFloat(TEXT("WrappedPoint.Values[1] == 2.0"), WrappedPoint.Values[1], 2.0);
	TestEqualExactFloat(TEXT("WrappedPoint.Values[2] == 3.0"), WrappedPoint.Values[2], 3.0);

    return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FObservationClipperWrapValuesMixedSpaceTest, "Schola.ObservationWrappers.ObservationClipper.Wrap Values Mixed Space Test ", EAutomationTestFlags::EditorContext | EAutomationTestFlags::ProductFilter)

bool FObservationClipperWrapValuesMixedSpaceTest::RunTest(const FString& Parameters)
{
    UObservationClipper* Clipper = NewObject<UObservationClipper>();
    FBoxPoint Point = FBoxPoint({-1.0,2.0,6.0});
    FBoxSpace Space = FBoxSpace({0.0,1.0,2.0}, {3.0, 4.0, 5.0});
    Clipper->WrapBoxObservationSpace(Space);

    FBoxPoint WrappedPoint = Clipper->WrapBoxObservation(Point);

    TestEqualExactFloat(TEXT("Clipped Lower Bound: WrappedPoint.Values[0] == 0.0"), WrappedPoint.Values[0], 0.0);
	TestEqualExactFloat(TEXT("In Range: WrappedPoint.Values[1] == 2.0"), WrappedPoint.Values[1], 2.0);
	TestEqualExactFloat(TEXT("Clipped Upper Bound: WrappedPoint.Values[2] == 5.0"), WrappedPoint.Values[2], 5.0);

    return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FObservationClipperWrapValuesOutOfSpaceTest, "Schola.ObservationWrappers.ObservationClipper.Wrap Values Out Of Space Test", EAutomationTestFlags::EditorContext | EAutomationTestFlags::ProductFilter)

bool FObservationClipperWrapValuesOutOfSpaceTest::RunTest(const FString& Parameters)
{
    UObservationClipper* Clipper = NewObject<UObservationClipper>();
    FBoxPoint Point = FBoxPoint({-1.0,5.0,6.0});
    FBoxSpace Space = FBoxSpace({0.0,1.0,2.0}, {3.0, 4.0, 5.0});
    Clipper->WrapBoxObservationSpace(Space);

    FBoxPoint WrappedPoint = Clipper->WrapBoxObservation(Point);

    TestEqualExactFloat(TEXT("WrappedPoint.Values[0] == 0.0"), WrappedPoint.Values[0], 0.0);
	TestEqualExactFloat(TEXT("WrappedPoint.Values[1] == 4.0"), WrappedPoint.Values[1], 4.0);
	TestEqualExactFloat(TEXT("WrappedPoint.Values[2] == 5.0"), WrappedPoint.Values[2], 5.0);

    return true;
}

