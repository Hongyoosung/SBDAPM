// Copyright (c) 2023 Advanced Micro Devices, Inc. All Rights Reserved.

#pragma once

#include "CoreMinimal.h"
#include <google/protobuf/message.h>
#include "Async/Future.h"
#include "Common/CommonInterfaces.h"
#include "Communicator/ProtobufDeserializer.h"

/**
 * @brief An abstracted communication backend that can send string/byte messages and can either be polled for responses or do exchanges when it sends messages.
 * @note This isn't a UE interface because those don't have typical Object Oriented interface functionality which we want here
 */
class IComBackendInterface
{
public:
	/**
	 * @brief Empty Desctructor explicitly defined to avoid C4265
	 */
	virtual ~IComBackendInterface(){};

	/**
	 * Perform any setup that happens prior to establishing any external connection.
	 * this could include opening a socket and waiting for a connection.
	 */
	virtual void Initialize() = 0;

	/**
	 * @brief Perform any setup that involves handshakes with the external communication (e.g. setting up serialization).
	 * Use after Initialize.
	 */
	virtual void Establish() = 0;

	/**
	 * @brief Close the external connection
	 */
	virtual void Shutdown() = 0;

	/**
	 * @brief Reset the Communication backend
	 */
	virtual void Restart() = 0;

	virtual void Start() = 0;
};

/**
 * @brief A Generic Interface for any service that can be used to receive messages of type T asynchronously via polling
 * @tparam T The type of message that will be output by this interface
 * 
 */
template <typename T>
class IPollingBackendInterface : public IComBackendInterface
{
public:
	
	/**
	 * @brief Poll the Backend for a message from the client
	 * @return An Empty optional if No messages have been received, a fulfilled one otherwise
	 * @note This function should be non-blocking
	 */
	virtual TOptional<const T*> Poll() = 0;

	/**
	 * @brief Poll the Backend for a message from the client and deserialize it into the specified type
	 * @tparam UnrealType The type to deserialize the message into
	 * @return An Empty optional if No messages have been received, a fulfilled one otherwise
	 */
	template <typename UnrealType>
	TOptional<UnrealType*> PollAndDeserialize()
	{
		TOptional<const T*>	   SerializedMessage = this->Poll();
		TOptional<UnrealType*> DeserializedMessage;
		if (SerializedMessage.IsSet())
		{
			DeserializedMessage.Emplace(new UnrealType(*SerializedMessage.GetValue()));
		}
		return DeserializedMessage;
	}
};

/**
 * @brief A Generic Interface for any service that can be used to send messages of type T asynchronously
 * @tparam T The type of message that will be input to this interface
 */
template <typename T>
class IProducerBackendInterface : public IComBackendInterface
{
public:

	/**
	 * @brief Send a message to the client
	 * @param[in] Msg The message to send to the client
	 */
	virtual void SendProtobufMessage(T* Msg) = 0;

	/**
	 * @brief Send a serializable unreal type to the client, converting it into `T` in the process.
	 * @tparam UnrealType The type to serialize into the message
	 */
	template <typename UnrealType>
	void SendSerializeableMessage(UnrealType* UnrealObject)
	{
		this->SendProtobufMessage(UnrealObject->ToProto());
	}
};

/**
 * @brief A Generic Interface for any service that can be used to exchange messages of type In and Out asynchronously
 * @tparam In The type of message that will be input to this interface
 * @tparam Out The type of message that will be output by this interface
 */
template <typename In, typename Out>
class IExchangeBackendInterface : public IComBackendInterface
{
public:
	/**
	 * @brief Initiate an Exchange with the Client.
	 * @return A future that will be fulfilled with the result of the exchange. The value ptr from the future is valid until the next time exchange is called.
	 */
	virtual TFuture<const In*> Receive() = 0;

	/**
	 * @brief Respond to a message from the client
	 * @param[in] Response The message to send to the client
	 */
	virtual void Respond(Out* Response) = 0;

	/**
	 * @brief Do an exchange before converting the protomessage into the specified type
	 * @tparam T The type to deserialize the message into
	 * @return A proto message that has been deserialized from `In` into type `T`
	 */
	template <typename T>
	TFuture<T*> ReceiveAndDeserialize()
	{
		// If T is not in our list of Deserializable Types this will cause a compilation error if this is the case
		TFuture<const In*> SerializedFuture = this->Receive();
		TPromise<T*>*	   DeserializedActionPromise = new TPromise<T*>();
		TFuture<T*>		   FutureDeserializedAction = DeserializedActionPromise->GetFuture();

		SerializedFuture.Next(
			[DeserializedActionPromise](const In* Request) {
				// This will ensure that the type is deserializeable
				DeserializedActionPromise->SetValue(ProtobufDeserializer::Deserialize<In,T>(*Request));
				delete DeserializedActionPromise;
			});

		return FutureDeserializedAction;
	}
};