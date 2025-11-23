// Copyright (c) 2023-2024 Advanced Micro Devices, Inc. All Rights Reserved.

#pragma once

#include "Common/Spaces/Space.h"
#include "../../../Generated/Spaces.pb.h"
#include "Common/Points/PointVariant.h"
#include "DiscreteSpace.generated.h"


using Schola::DiscreteSpace;

/**
 * @brief A struct representing a Discrete space (e.g. Vector of integers) of possible observations or actions.
 */
USTRUCT(BlueprintType)
struct SCHOLA_API FDiscreteSpace : public FSpace
{
	GENERATED_BODY()
public:
	/** The maximum value on each dimension of this DiscreteSpace. The Lower bound is always 0. e.g. High=2 gives actions {0,1} */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Definition", meta = (DisplayName = "Maximum Values"))
	TArray<int> High = TArray<int>();

	/**
	 * @brief Construct an empty DiscreteSpace
	 */
	FDiscreteSpace();

	FDiscreteSpace(TArray<int>& High);

	
	/**
	 * @brief Copy construct a DiscreteSpace
	 * @param[in] Other The DiscreteSpace to copy
	 */
	void Copy(const FDiscreteSpace& Other);

	/**
	 * @brief Merge another DiscreteSpace into this one
	 * @param[in] Other The DiscreteSpace to merge
	 */
	void Merge(const FDiscreteSpace& Other);
	
	/**
	 * @brief Add a dimension to this DiscreteSpace
	 * @param[in] DimSize The maximum value of the dimension
	 */
	void Add(int DimSize);

	virtual ~FDiscreteSpace();

	/**
	 * @brief fill a protobuf message with the data from this DiscreteSpace
	 * @param[in] Msg A ptr to the protobuf message to fill
	 */
	void FillProtobuf(DiscreteSpace* Msg) const;

	/**
	 * @brief fill a protobuf message with the data from this DiscreteSpace
	 * @param[in] Msg A ref to the protobuf message to fill
	 */
	void FillProtobuf(DiscreteSpace& Msg) const;

	/**
	 * @brief Get the3 index of the maximum value in an Array of Values
	 * @param[in] Vector The vector to get the maximum values index from
	 * @return The index of the maximum value in the vector
	 */
	int GetMaxValue(const TArray<float>& Vector) const;

	// FSpace API

	void FillProtobuf(FundamentalSpace* Msg) const override;

	int GetNumDimensions() const override;

	ESpaceValidationResult Validate(TPoint& Observation) const override;

	int GetFlattenedSize() const override;

	bool IsEmpty() const override;

	TPoint MakeTPoint() const override;

	TPoint UnflattenAction(const TArray<float>& Data, int Offset = 0) const override;

	void FlattenPoint(TArrayView<float> Buffer, const TPoint& Point) const override;

};