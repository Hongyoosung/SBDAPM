// Copyright (c) 2023-2025 Advanced Micro Devices, Inc. All Rights Reserved.

#pragma once

#include "Common/Spaces/Space.h"
#include "../../../Generated/Spaces.pb.h"
#include "BinarySpace.generated.h"

using Schola::BinarySpace;

/**
 * @brief A struct representing a Binary space (e.g. boolean vector) of possible observations or actions.
 */
USTRUCT(BlueprintType)
struct SCHOLA_API FBinarySpace : public FSpace
{
	GENERATED_BODY()
public:
	/** The number of dimensions in this BinarySpace*/
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Definition")
	int Shape = 0;

	/**
	 * @brief Construct an empty BinarySpace
	 */
	FBinarySpace();

	/**
	 * @brief Construct a BinarySpace with the given number of dimensions
	 * @param[in] Shape The number of dimensions in this BinarySpace
	 */
	FBinarySpace(int Shape);

	/**
	 * @brief Merge another BinarySpace into this one
	 * @param[in] Other The BinarySpace to merge
	 */
	void Merge(const FBinarySpace& Other);

	/**
	 * @brief Copy constructor
	 * @param[in] Other The BinarySpace to copy
	 */
	void Copy(const FBinarySpace& Other);

	/**
	 * @brief fill a protobuf message with the data from this BinarySpace
	 * @param[in] Msg A ptr to the protobuf message to fill
	 */
	void FillProtobuf(BinarySpace* Msg) const;

	/**
	 * @brief fill a protobuf message with the data from this BinarySpace
	 * @param[in] Msg A ref to the protobuf message to fill
	 */
	void FillProtobuf(BinarySpace& Msg) const;

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