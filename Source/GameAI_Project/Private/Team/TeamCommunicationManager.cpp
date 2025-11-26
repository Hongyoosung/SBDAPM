#include "Team/TeamCommunicationManager.h"
#include "Team/TeamLeaderComponent.h"
#include "Team/FollowerAgentComponent.h"
#include "Engine/World.h"
#include "Kismet/GameplayStatics.h"

UTeamCommunicationManager::UTeamCommunicationManager()
{
	// Default configuration
	bEnableMessageQueue = false;  // Deliver immediately by default
	MaxQueueSize = 100;
	bEnablePeerToPeer = true;
	bEnableMessageLogging = true;
}

//------------------------------------------------------------------------------
// LEADER → FOLLOWER MESSAGING
//------------------------------------------------------------------------------

void UTeamCommunicationManager::SendFormationUpdate(
	UTeamLeaderComponent* Leader,
	AActor* Follower,
	FVector FormationPosition,
	FRotator FormationRotation)
{
	if (!Leader || !Follower) return;

	FTeamMessage Message;
	Message.MessageType = ETeamMessageType::FormationUpdate;
	Message.Sender = Leader->GetOwner();
	Message.Recipient = Follower;
	Message.Priority = 4;
	Message.MessageData.Add(TEXT("Position"), FormationPosition.ToString());
	Message.MessageData.Add(TEXT("Rotation"), FormationRotation.ToString());

	if (bEnableMessageQueue)
	{
		QueueMessage(Message);
	}
	else
	{
		DeliverMessage(Message);
	}

	TotalMessagesSent++;
	OnMessageSent.Broadcast(Message);

	if (bEnableMessageLogging)
	{
		LogMessage(Message, true);
	}
}

void UTeamCommunicationManager::SendAcknowledgement(
	UTeamLeaderComponent* Leader,
	AActor* Follower,
	const FString& AcknowledgementMessage)
{
	if (!Leader || !Follower) return;

	FTeamMessage Message;
	Message.MessageType = ETeamMessageType::Acknowledgement;
	Message.Sender = Leader->GetOwner();
	Message.Recipient = Follower;
	Message.Priority = 3;
	Message.MessageData.Add(TEXT("Message"), AcknowledgementMessage);

	if (bEnableMessageQueue)
	{
		QueueMessage(Message);
	}
	else
	{
		DeliverMessage(Message);
	}

	TotalMessagesSent++;
}

//------------------------------------------------------------------------------
// FOLLOWER → LEADER MESSAGING
//------------------------------------------------------------------------------

void UTeamCommunicationManager::SendEventToLeader(
	UFollowerAgentComponent* Follower,
	UTeamLeaderComponent* Leader,
	const FStrategicEventContext& EventContext)
{
	if (!Follower || !Leader) return;

	FTeamMessage Message;
	Message.MessageType = ETeamMessageType::EventSignal;
	Message.Sender = Follower->GetOwner();
	Message.Recipient = Leader->GetOwner();
	Message.Priority = EventContext.Priority;
	Message.EventContext = EventContext;

	// Events are typically high-priority, deliver immediately
	DeliverMessage(Message);

	TotalMessagesSent++;
	OnMessageSent.Broadcast(Message);

	if (bEnableMessageLogging)
	{
		LogMessage(Message, true);
	}
}

void UTeamCommunicationManager::SendStatusReport(
	UFollowerAgentComponent* Follower,
	UTeamLeaderComponent* Leader,
	float Health,
	float Ammo,
	const FString& StatusMessage)
{
	if (!Follower || !Leader) return;

	FTeamMessage Message;
	Message.MessageType = ETeamMessageType::StatusReport;
	Message.Sender = Follower->GetOwner();
	Message.Recipient = Leader->GetOwner();
	Message.Priority = 2;
	Message.MessageData.Add(TEXT("Health"), FString::SanitizeFloat(Health));
	Message.MessageData.Add(TEXT("Ammo"), FString::SanitizeFloat(Ammo));
	Message.MessageData.Add(TEXT("Status"), StatusMessage);

	if (bEnableMessageQueue)
	{
		QueueMessage(Message);
	}
	else
	{
		DeliverMessage(Message);
	}

	TotalMessagesSent++;
}

void UTeamCommunicationManager::SendAssistanceRequest(
	UFollowerAgentComponent* Follower,
	UTeamLeaderComponent* Leader,
	int32 Priority)
{
	if (!Follower || !Leader) return;

	FTeamMessage Message;
	Message.MessageType = ETeamMessageType::RequestAssistance;
	Message.Sender = Follower->GetOwner();
	Message.Recipient = Leader->GetOwner();
	Message.Priority = Priority;

	// Assistance requests are urgent, deliver immediately
	DeliverMessage(Message);

	TotalMessagesSent++;
	OnMessageSent.Broadcast(Message);

	if (bEnableMessageLogging)
	{
		UE_LOG(LogTemp, Warning, TEXT("TeamComm: Follower %s requests assistance (Priority: %d)"),
			*Follower->GetOwner()->GetName(), Priority);
	}
}

//------------------------------------------------------------------------------
// PEER-TO-PEER MESSAGING
//------------------------------------------------------------------------------

void UTeamCommunicationManager::SendPeerMessage(
	AActor* Sender,
	AActor* Recipient,
	const FString& Message,
	int32 Priority)
{
	if (!bEnablePeerToPeer)
	{
		UE_LOG(LogTemp, Warning, TEXT("TeamComm: Peer-to-peer messaging is disabled"));
		return;
	}

	if (!Sender || !Recipient) return;

	FTeamMessage TeamMessage;
	TeamMessage.MessageType = ETeamMessageType::PeerMessage;
	TeamMessage.Sender = Sender;
	TeamMessage.Recipient = Recipient;
	TeamMessage.Priority = Priority;
	TeamMessage.MessageData.Add(TEXT("Message"), Message);

	DeliverMessage(TeamMessage);

	TotalMessagesSent++;
	OnMessageSent.Broadcast(TeamMessage);
}

void UTeamCommunicationManager::BroadcastToNearby(
	AActor* Sender,
	float Radius,
	const FString& Message,
	int32 Priority)
{
	if (!bEnablePeerToPeer) return;
	if (!Sender) return;

	// Find nearby actors with FollowerAgentComponent
	TArray<AActor*> FoundActors;
	UGameplayStatics::GetAllActorsWithTag(Sender->GetWorld(), FName("Follower"), FoundActors);

	FVector SenderLocation = Sender->GetActorLocation();

	for (AActor* Actor : FoundActors)
	{
		if (Actor == Sender) continue;

		float Distance = FVector::Dist(SenderLocation, Actor->GetActorLocation());
		if (Distance <= Radius)
		{
			SendPeerMessage(Sender, Actor, Message, Priority);
		}
	}

	if (bEnableMessageLogging)
	{
		UE_LOG(LogTemp, Verbose, TEXT("TeamComm: %s broadcast message to nearby (Radius: %.1fm)"),
			*Sender->GetName(), Radius);
	}
}

//------------------------------------------------------------------------------
// MESSAGE QUEUE MANAGEMENT
//------------------------------------------------------------------------------

void UTeamCommunicationManager::QueueMessage(const FTeamMessage& Message)
{
	// Check queue size limit
	if (MessageQueue.Num() >= MaxQueueSize)
	{
		UE_LOG(LogTemp, Warning, TEXT("TeamComm: Message queue full (%d), dropping oldest message"), MaxQueueSize);
		MessageQueue.RemoveAt(0);
	}

	// Add to queue
	MessageQueue.Add(Message);

	// Sort by priority (highest first)
	MessageQueue.Sort([](const FTeamMessage& A, const FTeamMessage& B) {
		return A.Priority > B.Priority;
	});
}

void UTeamCommunicationManager::ProcessMessageQueue()
{
	if (MessageQueue.Num() == 0) return;

	// Process top message
	FTeamMessage TopMessage = MessageQueue[0];
	MessageQueue.RemoveAt(0);

	DeliverMessage(TopMessage);
}

void UTeamCommunicationManager::ClearMessageQueue()
{
	int32 NumCleared = MessageQueue.Num();
	MessageQueue.Empty();

	if (bEnableMessageLogging && NumCleared > 0)
	{
		UE_LOG(LogTemp, Warning, TEXT("TeamComm: Cleared %d messages from queue"), NumCleared);
	}
}

//------------------------------------------------------------------------------
// MESSAGE DELIVERY
//------------------------------------------------------------------------------

void UTeamCommunicationManager::DeliverMessage(const FTeamMessage& Message)
{
	if (!Message.Recipient)
	{
		UE_LOG(LogTemp, Warning, TEXT("TeamComm: Cannot deliver message, no recipient"));
		return;
	}

	TotalMessagesReceived++;
	OnMessageReceived.Broadcast(Message);

	// Route message based on type
	switch (Message.MessageType)
	{
		case ETeamMessageType::EventSignal:
		{
			// Deliver event to leader
			if (UTeamLeaderComponent* Leader = Message.Recipient->FindComponentByClass<UTeamLeaderComponent>())
			{
				Leader->ProcessStrategicEventWithContext(Message.EventContext);
			}
			break;
		}

		case ETeamMessageType::FormationUpdate:
		{
			// Update follower formation position
			// TODO: Implement formation system
			break;
		}

		case ETeamMessageType::StatusReport:
		case ETeamMessageType::RequestAssistance:
		{
			// These are informational, no specific action needed
			// Could trigger leader re-evaluation if needed
			break;
		}

		case ETeamMessageType::PeerMessage:
		case ETeamMessageType::BroadcastMessage:
		{
			// Peer-to-peer messages
			// Could be used for coordination, callouts, etc.
			break;
		}

		default:
			UE_LOG(LogTemp, Warning, TEXT("TeamComm: Unknown message type %d"),
				static_cast<int32>(Message.MessageType));
			break;
	}

	if (bEnableMessageLogging)
	{
		LogMessage(Message, false);
	}
}

//------------------------------------------------------------------------------
// LOGGING
//------------------------------------------------------------------------------

void UTeamCommunicationManager::LogMessage(const FTeamMessage& Message, bool bSending)
{
	FString Direction = bSending ? TEXT("SEND") : TEXT("RECV");
	FString MessageTypeName = UEnum::GetValueAsString(Message.MessageType);

	FString SenderName = Message.Sender ? Message.Sender->GetName() : TEXT("None");
	FString RecipientName = Message.Recipient ? Message.Recipient->GetName() : TEXT("None");

	UE_LOG(LogTemp, Verbose, TEXT("TeamComm [%s]: %s | %s → %s | Priority: %d"),
		*Direction,
		*MessageTypeName,
		*SenderName,
		*RecipientName,
		Message.Priority);

	// Log event details if applicable
	if (Message.MessageType == ETeamMessageType::EventSignal)
	{
		UE_LOG(LogTemp, Verbose, TEXT("  Event: %s"),
			*UEnum::GetValueAsString(Message.EventContext.EventType));
	}
}
