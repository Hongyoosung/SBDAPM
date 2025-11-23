// Copyright (c) 2023-2025 Advanced Micro Devices, Inc. All Rights Reserved.

#pragma once

#include "Common/Spaces/SpaceVariant.h"
#include "Common/Points/DictPoint.h"
#include "../../../Generated/Spaces.pb.h"
#include "DictSpace.generated.h"

using Schola::DictSpace;

/**
 * @brief A struct representing a dictionary of possible observations or actions.
 */
USTRUCT()
struct SCHOLA_API FDictSpace
{
	GENERATED_BODY()
	// We store the dictionary as 2 arrays here. This is because we need to be able to look up spaces by label, but we also need to be able to iterate over them in order

	/** The labels of the spaces in this dictionary, used as keys for lookups */
	UPROPERTY(VisibleAnywhere, Category = "Space")
	TArray<FString> Labels;

	/** The spaces in this dictionary */
	TArray<TSpace>	Spaces;

	/**
	 * @brief Construct an empty DictSpace
	 */
	FDictSpace();
	
	//Utility Methods
	/**
	 * @brief Get the number of spaces in this dictionary
	 * @return The number of spaces in this dictionary
	 */
	int						   Num();

	/**
	 * @brief Get the number of dimensions, of all spaces in this dictionary after flattening
	 * @return The number of dimensions
	 */
	int						   GetFlattenedSize() const;

	//Methods for working with Points in the Space
	/** 
	 * @brief Validate a point in this space, by checking if all of it's dimensions pass validation
	 * @param[in] PointMap The point to validate
	 * @return An enum indicating the result of the validation
	 */
	ESpaceValidationResult	   Validate(FDictPoint& PointMap) const;

	/**
	* @brief Empty this Dict Space, by deleting all the keys and values
	*/
	void					   Reset();
	
	// Methods for adding things to the Space. Note that items are added in order here

	/**
	 * @brief Create a new empty space in place in this dictionary
	 * @param[in] Key The label of the space
	 * @return A reference to the newly added space
	 */
	TSpace& Add(const FString& Key);
	/**
	 * @brief Add a preallocated space from a reference to this dictionary
	 * @param[in] Key The label of the space
	 * @param[in] Value The BoxSpace to add
	 */
	void	Add(const FString& Key, TSpace& Value);
	/**
	 * @brief Add a BoxSpace to this dictionary from a reference
	 * @param[in] Key The label of the space
	 * @param[in] Value The BoxSpace to add
	 */
	void	Add(const FString& Key, FBoxSpace& Value);
	/**
	 * @brief Add a DiscreteSpace to this dictionary from a reference
	 * @param[in] Key The label of the space
	 * @param[in] Value The DiscreteSpace to add
	 */
	void	Add(const FString& Key, FDiscreteSpace& Value);
	/**
	 * @brief Add a BinarySpace to this dictionary from a reference
	 * @param[in] Key The label of the space
	 * @param[in] Value The BinarySpace to add
	 */
	void	Add(const FString& Key, FBinarySpace& Value);

	/**
	 * @brief Append another DictSpace to this one
	 * @param[in] Other The DictSpace to append
	 */
	void	Append(const FDictSpace& Other);

	// Protobuf Support
	/**
	 * @brief Convert this DictSpace to a protobuf message
	 * @return A protobuf message representing this DictSpace
	 */
	DictSpace* ToProtobuf() const;

	/**
	 * @brief Fill a protobuf message with the data from this DictSpace
	 * @param[in] Msg The protobuf message to fill
	 */
	void	   FillProtobuf(DictSpace* Msg) const;
	/**
	 * @brief Configure an empty DictPoint with the correct entries corresponding to this space
	 * @param[in,out] EmptyPoint The point to initialize
	 */ 
	void InitializeEmptyDictPoint(FDictPoint& EmptyPoint);

	/**
	 * @brief Create an empty DictPoint from a flattened point
	 * @param[in] FlattenedPoint The flattened point buffer to unflatten
	 * @return The unflattened point
	 */
	FDictPoint				   UnflattenPoint(TArray<float>& FlattenedPoint);

	/**
	 * @brief Get a subspace from this DictSpace, from an Index
	 * @param[in] Index The index of the subspace
	 * @return The subspace
	 */
	TSpace& operator[](int Index)
	{
		return this->Spaces[Index];
	};


	/**
	 * @brief Get a subspace from this DictSpace, from a Label
	 * @param[in] Label The label of the subspace
	 * @return The subspace
	 */
	TSpace& operator[](const FString& Label)
	{
		return this->Spaces[this->Labels.IndexOfByKey(Label)];
	};


};