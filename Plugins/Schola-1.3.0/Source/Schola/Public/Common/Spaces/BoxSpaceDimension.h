// Copyright (c) 2023-2025 Advanced Micro Devices, Inc. All Rights Reserved.

#pragma once
#include "../../../Generated/Spaces.pb.h"
#include "BoxSpaceDimension.generated.h"

/**
 * @brief A struct representing a dimension of a box(continuous) space of possible observations or actions.
 */
USTRUCT(BlueprintType)
struct SCHOLA_API FBoxSpaceDimension 
{
	GENERATED_BODY()

	/** The upper bound on this dimension */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Definition")
	float High = 1.0;

	/** The lower bound on this dimension */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Definition")
	float Low = -1.0;

	/** 
	 * @brief Construct a BoxSpaceDimension with default values
	 */
	FBoxSpaceDimension();

	/**
	 * @brief Construct a BoxSpaceDimension with the given bounds
	 * @param[in] Low The lower bound
	 * @param[in] High The upper bound
	 */
	FBoxSpaceDimension(float Low, float High);

	void FillProtobuf(Schola::BoxSpace::BoxSpaceDimension* Dimension) const;

	/**
	 * @brief Get a unit sized BoxSpaceDimension centered at 0.5
	 * @return A BoxSpaceDimension with bounds [0, 1]
	 */
	static inline FBoxSpaceDimension ZeroOneUnitDimension() { return FBoxSpaceDimension(0, 1); };

	/**
	 * @brief Get a unit sized BoxSpaceDimension centered at 0
	 * @return A BoxSpaceDimension with bounds [-0.5, 0.5]
	 */
	static inline FBoxSpaceDimension CenteredUnitDimension() { return FBoxSpaceDimension(-0.5, 0.5); };

	/**
	 * @brief Rescale a normalized value to be within this space
	 * @param[in] Value The value to rescale
	 * @return The rescaled value
	 */
	float RescaleValue(float Value) const;

	/**
	 * @brief Rescale from an another box space dimension to be within this space
	 * @param[in] Value The value to rescale
	 * @param[in] OldHigh The upper bound of the input space
	 * @param[in] OldLow The lower bound of the input space
	 * @return The rescaled value
	 */
	float RescaleValue(float Value, float OldHigh, float OldLow) const;

	/**
	 * @brief Normalize a value from this space to be in the range [0, 1]
	 * @param[in] Value The value to normalize
	 * @return The normalized value
	 */
	float NormalizeValue(float Value) const;

	/**
	 * @brief Check if two BoxSpaces are equal
	 * @param[in] Other The BoxSpace to compare to
	 * @return True if the BoxSpaces are equal
	 */
	bool operator==(const FBoxSpaceDimension& Other) const;
};