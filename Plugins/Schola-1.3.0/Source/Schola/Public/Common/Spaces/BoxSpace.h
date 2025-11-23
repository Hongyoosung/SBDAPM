// Copyright (c) 2023-2025 Advanced Micro Devices, Inc. All Rights Reserved.

#pragma once

#include "Common/Spaces/Space.h"
#include "../../../Generated/Spaces.pb.h"
#include "Common/Spaces/BoxSpaceDimension.h"
#include "BoxSpace.generated.h"


using Schola::BoxSpace;

/**
 * @brief A struct representing a box(continuous) space of possible observations or actions.
 * @details A BoxSpace is a Cartesian product of BoxSpaceDimensions. Each dimension is a continuous space.
 */
USTRUCT(BlueprintType)
struct SCHOLA_API FBoxSpace : public FSpace
{
	GENERATED_BODY()

public:
	
	/** The dimensions of this BoxSpace */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Definition", meta = (TitleProperty = "[{Low}, {High}]"))
	TArray<FBoxSpaceDimension> Dimensions = TArray<FBoxSpaceDimension>();
	
	/** The shape of the Box Space. If empty uses a 1d-array for the Dimensions */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Definition", meta = (TitleProperty = "Shape"))
	TArray<int> Shape = TArray<int>();
	
	/**
	 * @brief Construct an empty BoxSpace
	 */
	FBoxSpace();

	/**
	 * @brief Construct a BoxSpace with the given bounds
	 * @param[in] Low An array representing the the lower bound of each dimension
	 * @param[in] High An array representing the the upper bound of each dimension
	 * @note Low and High must have the same length
	 */
	FBoxSpace(TArray<float>& Low, TArray<float>& High, const TArray<int>& Shape = TArray<int>());

	/**
	 * @brief Construct a BoxSpace with the given bounds
	 * @param[in] Dimensions An array of BoxSpaceDimensions
	 */
	FBoxSpace(TArray<FBoxSpaceDimension>& Dimensions, const TArray<int>& Shape = TArray<int>());

	/**
	 * @brief Construct a BoxSpace from an initializer list of Low, and an initializer list of High values
	 * @param[in] Low The initializer list of lower bounds
	 * @param[in] High The initializer list of upper bounds
	 */
	FBoxSpace(std::initializer_list<float> Low, std::initializer_list<float> High, std::initializer_list<int> Shape = std::initializer_list<int>());

	/**
	 * @brief Construct an empty BoxSpace with a preallocated number of uninitialized dimensions
	 * @param[in] Shape The shape of the BoxSpace, preallocated with uninitialized dimensions
	 */
	FBoxSpace(const TArray<int>& Shape);

	/**
	 * @brief Copy constructor
	 * @param[in] Other The BoxSpace to copy
	 */
	void Copy(const FBoxSpace& Other);

	/**
	 * @brief Get the normalized version of this BoxSpace
	 * @return A BoxSpace with all dimensions normalized to [0, 1]
	 */
	FBoxSpace GetNormalizedObservationSpace() const;

	virtual ~FBoxSpace();

	/**
	 * @brief Fill a protobuf message with the data from this BoxSpace
	 * @param[in] Msg A ptr to the protobuf message to fill
	 */
	void FillProtobuf(BoxSpace* Msg) const;

	/**
	 * @brief Fill a protobuf message with the data from this BoxSpace
	 * @param[in] Msg A ref to the protobuf message to fill
	 */
	void FillProtobuf(BoxSpace& Msg) const;

	/**
	 * @brief Add a dimension to this BoxSpace
	 * @param[in] Low The lower bound of the dimension
	 * @param[in] High The upper bound of the dimension
	 */
	void Add(float Low, float High);

	/**
	 * @brief Add a dimension to this BoxSpace
	 * @param[in] Dimension The BoxSpaceDimension to add
	 */
	void Add(const FBoxSpaceDimension& Dimension);

	//FSpace API

	void FillProtobuf(FundamentalSpace* Msg) const override;

	int GetNumDimensions() const override;

	ESpaceValidationResult Validate(TPoint& Observation) const override;

	int GetFlattenedSize() const override;

	bool IsEmpty() const override;

	TPoint MakeTPoint() const override;

	/**
	 * @brief Convert an observation in this space to one in the normalized equivalent space 
	 * @param[in,out] Observation The observation to normalize
	 * @return A Box point in the normalized space.
	 */
	FBoxPoint NormalizeObservation(const FBoxPoint& Observation) const;

	TPoint UnflattenAction(const TArray<float>& Data, int Offset = 0) const override;

	void FlattenPoint(TArrayView<float> Buffer, const TPoint& Point) const override;
};

