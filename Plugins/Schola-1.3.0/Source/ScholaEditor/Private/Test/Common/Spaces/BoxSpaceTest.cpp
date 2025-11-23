// Copyright (c) 2024 Advanced Micro Devices, Inc. All Rights Reserved.

#include "Misc/AutomationTest.h"
#include "Tests/AutomationEditorCommon.h"

#include "Common/Spaces/BoxSpace.h"

#define TestEqualExactFloat(TestMessage, Actual, Expected) TestEqual(TestMessage, (float)Actual, (float)Expected, 0.0001f)

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FBoxSpaceDefaultTest, "Schola.Spaces.BoxSpace.Default Creation Test ", EAutomationTestFlags::EditorContext | EAutomationTestFlags::ProductFilter)

bool FBoxSpaceDefaultTest::RunTest(const FString& Parameters)
{
    FBoxSpace BoxSpace = FBoxSpace();

    TestEqualExactFloat(TEXT("BoxSpace.Dimensions.Num() == 0"), BoxSpace.Dimensions.Num(), 0);

    return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FBoxSpaceLowHighArrayCreationTest, "Schola.Spaces.BoxSpace.From Low High Array Test ", EAutomationTestFlags::EditorContext | EAutomationTestFlags::ProductFilter)

bool FBoxSpaceLowHighArrayCreationTest::RunTest(const FString& Parameters)
{
    TArray<float> Low = { -1.0, -2.0, -3.0 };
    TArray<float> High = { 1.0, 2.0, 3.0 };

    FBoxSpace BoxSpace = FBoxSpace(Low, High);

    TestEqual(TEXT("BoxSpace.Dimensions.Num() == 3"), BoxSpace.Dimensions.Num(), 3);

    //Check that the Low Values Match
    TestEqualExactFloat(TEXT("BoxSpace.Dimensions[0].Low == -1.0"), BoxSpace.Dimensions[0].Low, -1.0);
    TestEqualExactFloat(TEXT("BoxSpace.Dimensions[1].Low == -2.0"), BoxSpace.Dimensions[1].Low, -2.0);
    TestEqualExactFloat(TEXT("BoxSpace.Dimensions[2].Low == -3.0"), BoxSpace.Dimensions[2].Low, -3.0);

    //Check that the Hight Values Match
    TestEqualExactFloat(TEXT("BoxSpace.Dimensions[0].High == 1.0"), BoxSpace.Dimensions[0].High, 1.0);
    TestEqualExactFloat(TEXT("BoxSpace.Dimensions[1].High == 2.0"), BoxSpace.Dimensions[1].High, 2.0);
    TestEqualExactFloat(TEXT("BoxSpace.Dimensions[2].High == 3.0"), BoxSpace.Dimensions[2].High, 3.0);

    return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FBoxSpaceDimensionArrayCreationTest, "Schola.Spaces.BoxSpace.From Dimension Array Test ", EAutomationTestFlags::EditorContext | EAutomationTestFlags::ProductFilter)

bool FBoxSpaceDimensionArrayCreationTest::RunTest(const FString& Parameters)
{
    TArray<FBoxSpaceDimension> Dimensions = { FBoxSpaceDimension(-1.0, 1.0), FBoxSpaceDimension(-2.0, 2.0), FBoxSpaceDimension(-3.0, 3.0) };

    FBoxSpace BoxSpace = FBoxSpace(Dimensions);

    TestEqual(TEXT("BoxSpace.Dimensions.Num() == 3"), BoxSpace.Dimensions.Num(), 3);

    //Check that the Low Values Match
    TestEqualExactFloat(TEXT("BoxSpace.Dimensions[0].Low == -1.0"), BoxSpace.Dimensions[0].Low, -1.0);
    TestEqualExactFloat(TEXT("BoxSpace.Dimensions[1].Low == -2.0"), BoxSpace.Dimensions[1].Low, -2.0);
    TestEqualExactFloat(TEXT("BoxSpace.Dimensions[2].Low == -3.0"), BoxSpace.Dimensions[2].Low, -3.0);

    //Check that the Hight Values Match
    TestEqualExactFloat(TEXT("BoxSpace.Dimensions[0].High == 1.0"), BoxSpace.Dimensions[0].High, 1.0);
    TestEqualExactFloat(TEXT("BoxSpace.Dimensions[1].High == 2.0"), BoxSpace.Dimensions[1].High, 2.0);
    TestEqualExactFloat(TEXT("BoxSpace.Dimensions[2].High == 3.0"), BoxSpace.Dimensions[2].High, 3.0);

    return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FBoxSpaceAddDimensionTest, "Schola.Spaces.BoxSpace.Add Dimension Test ", EAutomationTestFlags::EditorContext | EAutomationTestFlags::ProductFilter)

bool FBoxSpaceAddDimensionTest::RunTest(const FString& Parameters)
{
    FBoxSpace BoxSpace = FBoxSpace();

    BoxSpace.Add(-1.0, 1.0);
    BoxSpace.Add(FBoxSpaceDimension(-2.0, 2.0));
    BoxSpace.Add(-3.0, 3.0);

    TestEqual(TEXT("BoxSpace.Dimensions.Num() == 3"), BoxSpace.Dimensions.Num(), 3);

    //Check that the Low Values Match
    TestEqualExactFloat(TEXT("BoxSpace.Dimensions[0].Low == -1.0"), BoxSpace.Dimensions[0].Low, -1.0);
    TestEqualExactFloat(TEXT("BoxSpace.Dimensions[1].Low == -2.0"), BoxSpace.Dimensions[1].Low, -2.0);
    TestEqualExactFloat(TEXT("BoxSpace.Dimensions[2].Low == -3.0"), BoxSpace.Dimensions[2].Low, -3.0);

    //Check that the Hight Values Match
    TestEqualExactFloat(TEXT("BoxSpace.Dimensions[0].High == 1.0"), BoxSpace.Dimensions[0].High, 1.0);
    TestEqualExactFloat(TEXT("BoxSpace.Dimensions[1].High == 2.0"), BoxSpace.Dimensions[1].High, 2.0);
    TestEqualExactFloat(TEXT("BoxSpace.Dimensions[2].High == 3.0"), BoxSpace.Dimensions[2].High, 3.0);

    return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FBoxSpaceNormalizedObservationSpaceTest, "Schola.Spaces.BoxSpace.Get Normalized Observation Space Test ", EAutomationTestFlags::EditorContext | EAutomationTestFlags::ProductFilter)

bool FBoxSpaceNormalizedObservationSpaceTest::RunTest(const FString& Parameters)
{
    FBoxSpace BoxSpace = FBoxSpace();
    BoxSpace.Add(-1.0, 1.0);
    BoxSpace.Add(-2.0, 2.0);
    BoxSpace.Add(-3.0, 3.0);

    FBoxSpace NormalizedBoxSpace = BoxSpace.GetNormalizedObservationSpace();

    TestEqual(TEXT("NormalizedBoxSpace.Dimensions.Num() == 3"), NormalizedBoxSpace.Dimensions.Num(), 3);

    //Check that the Low Values Match
    TestEqualExactFloat(TEXT("NormalizedBoxSpace.Dimensions[0].Low == 0.0"), NormalizedBoxSpace.Dimensions[0].Low, 0.0);
    TestEqualExactFloat(TEXT("NormalizedBoxSpace.Dimensions[1].Low == 0.0"), NormalizedBoxSpace.Dimensions[1].Low, 0.0);
    TestEqualExactFloat(TEXT("NormalizedBoxSpace.Dimensions[2].Low == 0.0"), NormalizedBoxSpace.Dimensions[2].Low, 0.0);

    //Check that the Hight Values Match
    TestEqualExactFloat(TEXT("NormalizedBoxSpace.Dimensions[0].High == 1.0"), NormalizedBoxSpace.Dimensions[0].High, 1.0);
    TestEqualExactFloat(TEXT("NormalizedBoxSpace.Dimensions[1].High == 1.0"), NormalizedBoxSpace.Dimensions[1].High, 1.0);
    TestEqualExactFloat(TEXT("NormalizedBoxSpace.Dimensions[2].High == 1.0"), NormalizedBoxSpace.Dimensions[2].High, 1.0);

    return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FBoxSpaceFlattenedSizeTest, "Schola.Spaces.BoxSpace.Get Flattened Size Test ", EAutomationTestFlags::EditorContext | EAutomationTestFlags::ProductFilter)

bool FBoxSpaceFlattenedSizeTest::RunTest(const FString& Parameters)
{
    FBoxSpace BoxSpace = FBoxSpace();
    BoxSpace.Add(-1.0, 1.0);
    BoxSpace.Add(-2.0, 2.0);
    BoxSpace.Add(-3.0, 3.0);

    TestEqual(TEXT("BoxSpace.GetFlattenedSize() == 3"), BoxSpace.GetFlattenedSize(), 3);

    return true;
}


IMPLEMENT_SIMPLE_AUTOMATION_TEST(FBoxSpaceSuccesfulValidatePointTest, "Schola.Spaces.BoxSpace.Succesful Validate Point Test ", EAutomationTestFlags::EditorContext | EAutomationTestFlags::ProductFilter)

bool FBoxSpaceSuccesfulValidatePointTest::RunTest(const FString& Parameters)
{
    FBoxSpace BoxSpace = FBoxSpace();
    BoxSpace.Add(-1.0, 1.0);
    BoxSpace.Add(-2.0, 2.0);
    BoxSpace.Add(-3.0, 3.0);

    TPoint Point = TPoint(TInPlaceType<FBoxPoint>());
    Point.Get<FBoxPoint>().Add(0.0);
    Point.Get<FBoxPoint>().Add(0.0);
    Point.Get<FBoxPoint>().Add(0.0);

    ESpaceValidationResult Result = BoxSpace.Validate(Point);

    TestEqual(TEXT("BoxSpace.Validate(Point) == ESpaceValidationResult::Success"), Result, ESpaceValidationResult::Success);

    return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FBoxSpaceValidateWrongDimensionPointTest, "Schola.Spaces.BoxSpace.Validate Wrong Dimension Point Test ", EAutomationTestFlags::EditorContext | EAutomationTestFlags::ProductFilter)

bool FBoxSpaceValidateWrongDimensionPointTest::RunTest(const FString& Parameters)
{
    FBoxSpace BoxSpace = FBoxSpace();
    BoxSpace.Add(-1.0, 1.0);
    BoxSpace.Add(-2.0, 2.0);
    BoxSpace.Add(-3.0, 3.0);

    TPoint Point = TPoint(TInPlaceType<FBoxPoint>());
    Point.Get<FBoxPoint>().Add(0.0);
    Point.Get<FBoxPoint>().Add(0.0);

    ESpaceValidationResult Result = BoxSpace.Validate(Point);

    TestEqual(TEXT("BoxSpace.Validate(Point) == ESpaceValidationResult::Invalid"), Result, ESpaceValidationResult::WrongDimensions);

    return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FBoxSpaceValidateOutOfBoundsPointTest, "Schola.Spaces.BoxSpace.Validate Out Of Bounds Point Test ", EAutomationTestFlags::EditorContext | EAutomationTestFlags::ProductFilter)

bool FBoxSpaceValidateOutOfBoundsPointTest::RunTest(const FString& Parameters)
{
    FBoxSpace BoxSpace = FBoxSpace();
    BoxSpace.Add(-1.0, 1.0);
    BoxSpace.Add(-2.0, 2.0);
    BoxSpace.Add(-3.0, 3.0);

    TPoint Point = BoxSpace.MakeTPoint();
    Point.Get<FBoxPoint>().Add(0.0);
    Point.Get<FBoxPoint>().Add(0.0);
    Point.Get<FBoxPoint>().Add(4.0);

    ESpaceValidationResult Result = BoxSpace.Validate(Point);

    TestEqual(TEXT("BoxSpace.Validate(Point) == ESpaceValidationResult::OutOfBounds"), Result, ESpaceValidationResult::OutOfBounds);

    return true;
}
IMPLEMENT_SIMPLE_AUTOMATION_TEST(FBoxSpaceValidateWrongDataTypeTest, "Schola.Spaces.BoxSpace.Validate Wrong Type of Point Test ", EAutomationTestFlags::EditorContext | EAutomationTestFlags::ProductFilter)

bool FBoxSpaceValidateWrongDataTypeTest::RunTest(const FString& Parameters)
{
    FBoxSpace BoxSpace = FBoxSpace();
    BoxSpace.Add(-1.0, 1.0);
    BoxSpace.Add(-2.0, 2.0);
    BoxSpace.Add(-3.0, 3.0);

    TPoint Point = TPoint(TInPlaceType<FBinaryPoint>());

    ESpaceValidationResult Result = BoxSpace.Validate(Point);

    TestEqual(TEXT("BoxSpace.Validate(Point) == ESpaceValidationResult::WrongDataType"), Result, ESpaceValidationResult::WrongDataType);

    return true;
}



IMPLEMENT_SIMPLE_AUTOMATION_TEST(FBoxSpaceNormalizeObservationTest, "Schola.Spaces.BoxSpace.Normalize Observation Test ", EAutomationTestFlags::EditorContext | EAutomationTestFlags::ProductFilter)

bool FBoxSpaceNormalizeObservationTest::RunTest(const FString& Parameters)
{
    FBoxSpace BoxSpace = FBoxSpace();
    BoxSpace.Add(0, 1.0);
    BoxSpace.Add(-2.0, 2.0);
    BoxSpace.Add(-3.0, 3.0);

    FBoxPoint BoxPoint = FBoxPoint();
    BoxPoint.Add(0.0);
    BoxPoint.Add(0.0);
    BoxPoint.Add(0.0);

    FBoxPoint NormalizedBoxPoint = BoxSpace.NormalizeObservation(BoxPoint);

    TestEqualExactFloat(TEXT("NormalizedBoxPoint[0] == 0.0"), NormalizedBoxPoint[0], 0.0);
    TestEqualExactFloat(TEXT("NormalizedBoxPoint[1] == 0.5"), NormalizedBoxPoint[1], 0.5);
    TestEqualExactFloat(TEXT("NormalizedBoxPoint[2] == 3.0"), NormalizedBoxPoint[2], 0.5);

    return true;
}