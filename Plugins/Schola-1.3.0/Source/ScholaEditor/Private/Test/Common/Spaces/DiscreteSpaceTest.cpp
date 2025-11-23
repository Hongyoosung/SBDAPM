// Copyright (c) 2024 Advanced Micro Devices, Inc. All Rights Reserved.

#include "Misc/AutomationTest.h"
#include "Tests/AutomationEditorCommon.h"

#include "Common/Spaces/DiscreteSpace.h"

#define TestEqualExactFloat(TestMessage, Actual, Expected) TestEqual(TestMessage, (float)Actual, (float)Expected, 0.0001f)

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FDiscreteSpaceDefaultTest, "Schola.Spaces.DiscreteSpace.Default Creation Test ", EAutomationTestFlags::EditorContext | EAutomationTestFlags::ProductFilter)

bool FDiscreteSpaceDefaultTest::RunTest(const FString& Parameters)
{
    FDiscreteSpace DiscreteSpace = FDiscreteSpace();

    TestEqualExactFloat(TEXT("DiscreteSpace.High.Num() == 0"), DiscreteSpace.High.Num(), 0);

    return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FDiscreteSpaceHighArrayCreationTest, "Schola.Spaces.DiscreteSpace.From High Array Test ", EAutomationTestFlags::EditorContext | EAutomationTestFlags::ProductFilter)

bool FDiscreteSpaceHighArrayCreationTest::RunTest(const FString& Parameters)
{
    TArray<int> High = { 1, 2, 3 };

    FDiscreteSpace DiscreteSpace = FDiscreteSpace(High);

    TestEqual(TEXT("DiscreteSpace.High.Num() == 3"), DiscreteSpace.High.Num(), 3);

    //Check that the High Values Match
	TestEqualExactFloat(TEXT("DiscreteSpace.High[0] == 1"), DiscreteSpace.High[0], 1);
	TestEqualExactFloat(TEXT("DiscreteSpace.High[1] == 2"), DiscreteSpace.High[1], 2);
	TestEqualExactFloat(TEXT("DiscreteSpace.High[2] == 3"), DiscreteSpace.High[2], 3);

    return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FDiscreteSpaceAddTest, "Schola.Spaces.DiscreteSpace.Add Test ", EAutomationTestFlags::EditorContext | EAutomationTestFlags::ProductFilter)

bool FDiscreteSpaceAddTest::RunTest(const FString& Parameters)
{
    FDiscreteSpace DiscreteSpace = FDiscreteSpace();
    DiscreteSpace.Add(1);
    DiscreteSpace.Add(2);

    TestEqualExactFloat(TEXT("DiscreteSpace.High.Num() == 2"), DiscreteSpace.High.Num(), 2);

    //Check that the High Values Match
	TestEqualExactFloat(TEXT("DiscreteSpace.High[0] == 1"), DiscreteSpace.High[0], 1);
	TestEqualExactFloat(TEXT("DiscreteSpace.High[1] == 2"), DiscreteSpace.High[1], 2);

    return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FDiscreteSpaceMergeTest, "Schola.Spaces.DiscreteSpace.Merge Test ", EAutomationTestFlags::EditorContext | EAutomationTestFlags::ProductFilter)

bool FDiscreteSpaceMergeTest::RunTest(const FString& Parameters)
{
    FDiscreteSpace DiscreteSpace = FDiscreteSpace();
    DiscreteSpace.Add(1);
    DiscreteSpace.Add(2);

    FDiscreteSpace OtherDiscreteSpace = FDiscreteSpace();
    OtherDiscreteSpace.Add(3);
    OtherDiscreteSpace.Add(4);

    DiscreteSpace.Merge(OtherDiscreteSpace);

    TestEqualExactFloat(TEXT("DiscreteSpace.High.Num() == 4"), DiscreteSpace.High.Num(), 4);

    //Check that the High Values Match
	TestEqualExactFloat(TEXT("DiscreteSpace.High[0] == 1"), DiscreteSpace.High[0], 1);
    TestEqualExactFloat(TEXT("DiscreteSpace.High[1] == 2"), DiscreteSpace.High[1], 2);
    TestEqualExactFloat(TEXT("DiscreteSpace.High[2] == 3"), DiscreteSpace.High[2], 3);
    TestEqualExactFloat(TEXT("DiscreteSpace.High[3] == 4"), DiscreteSpace.High[3], 4);

    return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FDiscreteSpaceFlattenedSizeTest, "Schola.Spaces.DiscreteSpace.FlattenedSize Test ", EAutomationTestFlags::EditorContext | EAutomationTestFlags::ProductFilter)

bool FDiscreteSpaceFlattenedSizeTest::RunTest(const FString& Parameters)
{
    FDiscreteSpace DiscreteSpace = FDiscreteSpace();
    DiscreteSpace.Add(1);
    DiscreteSpace.Add(2);

    TestEqualExactFloat(TEXT("DiscreteSpace.GetFlattenedSize() == 3"), DiscreteSpace.GetFlattenedSize(), 3);

    return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FDiscreteSpaceIsEmptyTrueTest, "Schola.Spaces.DiscreteSpace.Is Empty True Test ", EAutomationTestFlags::EditorContext | EAutomationTestFlags::ProductFilter)

bool FDiscreteSpaceIsEmptyTrueTest::RunTest(const FString& Parameters)
{
    FDiscreteSpace DiscreteSpace = FDiscreteSpace();

    TestEqual(TEXT("DiscreteSpace.IsEmpty() == true"), DiscreteSpace.IsEmpty(), true);

    return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FDiscreteSpaceIsEmptyFalseTest, "Schola.Spaces.DiscreteSpace.Is Empty False Test ", EAutomationTestFlags::EditorContext | EAutomationTestFlags::ProductFilter)

bool FDiscreteSpaceIsEmptyFalseTest::RunTest(const FString& Parameters)
{
    FDiscreteSpace DiscreteSpace = FDiscreteSpace();
    DiscreteSpace.Add(1);

    TestEqual(TEXT("DiscreteSpace.IsEmpty() == false"), DiscreteSpace.IsEmpty(), false);

    return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FDiscreteSpaceGetNumDimensionsTest, "Schola.Spaces.DiscreteSpace.Get Num Dimensions Test ", EAutomationTestFlags::EditorContext | EAutomationTestFlags::ProductFilter)

bool FDiscreteSpaceGetNumDimensionsTest::RunTest(const FString& Parameters)
{
    FDiscreteSpace DiscreteSpace = FDiscreteSpace();
    DiscreteSpace.Add(1);
    DiscreteSpace.Add(2);

    TestEqualExactFloat(TEXT("DiscreteSpace.GetNumDimensions() == 2"), DiscreteSpace.GetNumDimensions(), 2);

    return true;
}

//Validation Tests

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FDiscreteSpaceValidateSuccessTest, "Schola.Spaces.DiscreteSpace.Validate Success Test ", EAutomationTestFlags::EditorContext | EAutomationTestFlags::ProductFilter)

bool FDiscreteSpaceValidateSuccessTest::RunTest(const FString& Parameters)
{
    FDiscreteSpace DiscreteSpace = FDiscreteSpace();
    DiscreteSpace.Add(1);
    DiscreteSpace.Add(2);

    TPoint Point = DiscreteSpace.MakeTPoint();
    FDiscretePoint& TypedPoint = Point.Get<FDiscretePoint>();
    TypedPoint.Values = { 0, 1 };

    TestEqual(TEXT("DiscreteSpace.Validate(Point) == ESpaceValidationResult::Success"), DiscreteSpace.Validate(Point), ESpaceValidationResult::Success);

    return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FDiscreteSpaceValidateOutOfBoundsTest, "Schola.Spaces.DiscreteSpace.Validate Out Of Bounds Test ", EAutomationTestFlags::EditorContext | EAutomationTestFlags::ProductFilter)

bool FDiscreteSpaceValidateOutOfBoundsTest::RunTest(const FString& Parameters)
{
    FDiscreteSpace DiscreteSpace = FDiscreteSpace();
    DiscreteSpace.Add(1);
    DiscreteSpace.Add(2);

    TPoint Point = DiscreteSpace.MakeTPoint();
    FDiscretePoint& TypedPoint = Point.Get<FDiscretePoint>();
    TypedPoint.Values = { 0, 2 };

    TestEqual(TEXT("DiscreteSpace.Validate(Point) == ESpaceValidationResult::OutOfBounds"), DiscreteSpace.Validate(Point), ESpaceValidationResult::OutOfBounds);

    return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FDiscreteSpaceValidateWrongDataTypeTest, "Schola.Spaces.DiscreteSpace.Validate Wrong Data Type Test ", EAutomationTestFlags::EditorContext | EAutomationTestFlags::ProductFilter)

bool FDiscreteSpaceValidateWrongDataTypeTest::RunTest(const FString& Parameters)
{
    FDiscreteSpace DiscreteSpace = FDiscreteSpace();
    DiscreteSpace.Add(1);
    DiscreteSpace.Add(2);

    TPoint Point = TPoint(TInPlaceType<FBinaryPoint>());

    TestEqual(TEXT("DiscreteSpace.Validate(Point) == ESpaceValidationResult::WrongDataType"), DiscreteSpace.Validate(Point), ESpaceValidationResult::WrongDataType);

    return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FDiscreteSpaceGetMaxValueTest, "Schola.Spaces.DiscreteSpace.Get Max Value Test ", EAutomationTestFlags::EditorContext | EAutomationTestFlags::ProductFilter)

bool FDiscreteSpaceGetMaxValueTest::RunTest(const FString& Parameters)
{
    FDiscreteSpace DiscreteSpace = FDiscreteSpace();

    TArray<float> Vector = { 0.0, 1.0, 3.0 };

    TestEqualExactFloat(TEXT("DiscreteSpace.GetMaxValue(Vector) == 2"), DiscreteSpace.GetMaxValue(Vector), 2);

    return true;
}

//Unflatten Action

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FDiscreteSpaceUnflattenActionTest, "Schola.Spaces.DiscreteSpace.Unflatten Action Test ", EAutomationTestFlags::EditorContext | EAutomationTestFlags::ProductFilter)

bool FDiscreteSpaceUnflattenActionTest::RunTest(const FString& Parameters)
{
    FDiscreteSpace DiscreteSpace = FDiscreteSpace();
    DiscreteSpace.Add(2);
    DiscreteSpace.Add(3);

    TArray<float> Data = { 0.0, 1.0 , 0.0 , 0.0 , 1.0 };

    TPoint Point = DiscreteSpace.UnflattenAction(Data);

    FDiscretePoint& TypedPoint = Point.Get<FDiscretePoint>();

    TestEqualExactFloat(TEXT("TypedPoint.Values.Num() == 2"), TypedPoint.Values.Num(), 2);
    TestEqualExactFloat(TEXT("TypedPoint.Values[0] == 0"), TypedPoint.Values[0], 1);
    TestEqualExactFloat(TEXT("TypedPoint.Values[1] == 1"), TypedPoint.Values[1], 2);

    return true;
}

//Flatten Point

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FDiscreteSpaceFlattenPointTest, "Schola.Spaces.DiscreteSpace.Flatten Point Test ", EAutomationTestFlags::EditorContext | EAutomationTestFlags::ProductFilter)

bool FDiscreteSpaceFlattenPointTest::RunTest(const FString& Parameters)
{
    FDiscreteSpace DiscreteSpace = FDiscreteSpace();
    DiscreteSpace.Add(2);
    DiscreteSpace.Add(3);

    TPoint Point = DiscreteSpace.MakeTPoint();
    FDiscretePoint& TypedPoint = Point.Get<FDiscretePoint>();
    TypedPoint.Values = { 1, 2 };

    TArray<float> Buffer = {0.0, 0.0, 0.0, 0.0, 0.0};
	
    DiscreteSpace.FlattenPoint(Buffer, Point);

    //Only indices 1 and 4 should be set to 1

    TestEqualExactFloat(TEXT("Buffer[0] == 0.0"), Buffer[0], 0.0);
    TestEqualExactFloat(TEXT("Buffer[1] == 1.0"), Buffer[1], 1.0);
    TestEqualExactFloat(TEXT("Buffer[2] == 0.0"), Buffer[2], 0.0);
    TestEqualExactFloat(TEXT("Buffer[3] == 0.0"), Buffer[3], 0.0);
    TestEqualExactFloat(TEXT("Buffer[4] == 1.0"), Buffer[4], 1.0);

    return true;
}

