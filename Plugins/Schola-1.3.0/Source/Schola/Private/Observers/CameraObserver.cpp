// Copyright (c) 2023 Advanced Micro Devices, Inc. All Rights Reserved.

#include "Observers/CameraObserver.h"
#include "RHICommandList.h"


void UCameraObserver::InitializeObserver()
{	
	UObject* Outer = this->GetOuter();
	AActor*	 OuterActor;
	UActorComponent* OuterComponent = Cast<UActorComponent>(Outer);

	if (OuterComponent)
	{
		OuterActor = OuterComponent->GetOwner();
	}
	else
	{
		OuterActor = Cast<AActor>(Outer);
	}

    USceneCaptureComponent2D* CapComponent = Cast<USceneCaptureComponent2D>(this->SceneCaptureCompRef.GetComponent(OuterActor));
	
	if(!CapComponent)
	{
		UE_LOG(LogTemp, Warning, TEXT("CameraObserver: SceneCaptureComponent2D not found"));
		return;
	}
	if(!RenderTarget)
	{
		if(CapComponent->TextureTarget)
		{
			RenderTarget = CapComponent->TextureTarget;
		}
		else
		{
			UE_LOG(LogTemp, Warning, TEXT("CameraObserver: TextureTarget not found. Creating new TextureTarget."));
			RenderTarget = NewObject<UTextureRenderTarget2D>();
			RenderTarget->bGPUSharedFlag = 1;
			RenderTarget->InitAutoFormat(128, 128);
		}
	}

	RenderTarget->bNoFastClear = 0;
	RenderTarget->bHDR_DEPRECATED = 0;
	Width = RenderTarget->GetSurfaceWidth();
	Height = RenderTarget->GetSurfaceHeight();
	CapComponent->TextureTarget = RenderTarget;
	CapComponent->TextureTarget->InitAutoFormat(Width, Height);
	UpdateChannelBooleans();
    
}
FBoxSpace UCameraObserver::GetObservationSpace() const
{
    FBoxSpace SpaceDefinition;
    
	SpaceDefinition.Dimensions.Init(FBoxSpaceDimension(0.0, 1.0), Width * Height * (bObserveChannelR + bObserveChannelG + bObserveChannelB + bObserveChannelA));
	SpaceDefinition.Shape = { (bObserveChannelR + bObserveChannelG + bObserveChannelB + bObserveChannelA) , Height, Width};
	return SpaceDefinition;
}

void UCameraObserver::CollectObservations(FBoxPoint& OutObservations)
{
	if (!RenderTarget)
	{
		UE_LOG(LogSchola, Error, TEXT("CameraObserver: RenderTarget not found. Not collecting Observations."));
		return;
	}
	TArray<FColor> Bitmap;
	int			   NumChannels = (bObserveChannelR + bObserveChannelG + bObserveChannelB + bObserveChannelA);
    OutObservations.Values.AddUninitialized(Width * Height * NumChannels);
	RenderTarget->GameThread_GetRenderTargetResource()->ReadPixels(Bitmap);
	
	for (int i = 0; i < Width*Height; i++)
    {
		int Index = i;
		if (bObserveChannelR)
		{
			OutObservations.Values[Index] = ((float)Bitmap[i].R / 255.0); // R
			Index += Width*Height;
		}
		
		if (bObserveChannelG)
		{
			OutObservations.Values[Index] = ((float)Bitmap[i].G / 255.0); // G
			Index += Width * Height;
		}

		if (bObserveChannelB)
		{
			OutObservations.Values[Index] = ((float)Bitmap[i].B / 255.0); // B
			Index += Width * Height;
		}

		if(bObserveChannelA)
		{
			OutObservations.Values[Index] = ((float)Bitmap[i].A / 255.0); // A
		}
	}
}

bool UCameraObserver::IsChannelUsed(USceneCaptureComponent2D* CapComponent, FName ChannelName) const{
	bool bLocalObserveChannelA = true, bLocalObserveChannelR = true, bLocalObserveChannelG = true, bLocalObserveChannelB = true;
	if(!CapComponent)
	{
		return true;
	}
	// Set channels based on the CaptureSource
	switch (CapComponent->CaptureSource)
	{
		case ESceneCaptureSource::SCS_SceneColorHDRNoAlpha:
		case ESceneCaptureSource::SCS_FinalColorLDR:
		case ESceneCaptureSource::SCS_DeviceDepth:
		case ESceneCaptureSource::SCS_Normal:
		case ESceneCaptureSource::SCS_BaseColor:
		case ESceneCaptureSource::SCS_FinalColorHDR:
		case ESceneCaptureSource::SCS_FinalToneCurveHDR:
			bLocalObserveChannelA = false;
			break;

		case ESceneCaptureSource::SCS_SceneColorSceneDepth:
			break;

		case ESceneCaptureSource::SCS_SceneDepth:
			bLocalObserveChannelG = false;
			bLocalObserveChannelB = false;
			bLocalObserveChannelA = false;
			break;

		default:
			break;
	}

	// Set channels based on the RenderTarget
	if(CapComponent->TextureTarget)
	{
		switch (CapComponent->TextureTarget->RenderTargetFormat)
		{
			case ETextureRenderTargetFormat::RTF_RGBA8:
			case ETextureRenderTargetFormat::RTF_RGBA8_SRGB:

			case ETextureRenderTargetFormat::RTF_RGB10A2:
			case ETextureRenderTargetFormat::RTF_RGBA16f:
			case ETextureRenderTargetFormat::RTF_RGBA32f:
				break;
			case ETextureRenderTargetFormat::RTF_RG32f:
			case ETextureRenderTargetFormat::RTF_RG16f:
			case ETextureRenderTargetFormat::RTF_RG8:
				bLocalObserveChannelB = false;
				bLocalObserveChannelA = false;
				break;
			case ETextureRenderTargetFormat::RTF_R32f:
			case ETextureRenderTargetFormat::RTF_R16f:
			case ETextureRenderTargetFormat::RTF_R8:
				bLocalObserveChannelG = false;
				bLocalObserveChannelB = false;
				bLocalObserveChannelA = false;
				break;
			default:
				break;
		}
	}
	return 	
		ChannelName == GET_MEMBER_NAME_CHECKED(UCameraObserver, bObserveChannelR) ? bLocalObserveChannelR : 
		ChannelName == GET_MEMBER_NAME_CHECKED(UCameraObserver, bObserveChannelG) ? bLocalObserveChannelG : 
		ChannelName == GET_MEMBER_NAME_CHECKED(UCameraObserver, bObserveChannelB) ? bLocalObserveChannelB : 
		ChannelName == GET_MEMBER_NAME_CHECKED(UCameraObserver, bObserveChannelA) ? bLocalObserveChannelA : 
		true;
}

#if WITH_EDITOR
bool UCameraObserver::CanEditChange(const FProperty* InProperty) const
{
	const bool bParentVal = Super::CanEditChange(InProperty);

	if(InProperty->GetFName() == GET_MEMBER_NAME_CHECKED(UCameraObserver, bObserveChannelR) || InProperty->GetFName() == GET_MEMBER_NAME_CHECKED(UCameraObserver, bObserveChannelG) || InProperty->GetFName() == GET_MEMBER_NAME_CHECKED(UCameraObserver, bObserveChannelB) || InProperty->GetFName() == GET_MEMBER_NAME_CHECKED(UCameraObserver, bObserveChannelA) ){
		// Check if the SceneCaptureCompRef is valid
		USceneCaptureComponent2D* CapComponent = Cast<USceneCaptureComponent2D>(SceneCaptureCompRef.GetComponent(Cast<UActorComponent>(GetOuter())->GetOwner()));
		return IsChannelUsed(CapComponent, InProperty->GetFName());
	}
	return true;
}
#endif

void UCameraObserver::UpdateChannelBooleans()
{
	// Check if the SceneCaptureCompRef is valid
	USceneCaptureComponent2D* CapComponent = Cast<USceneCaptureComponent2D>(SceneCaptureCompRef.GetComponent(Cast<UActorComponent>(GetOuter())->GetOwner()));
	bObserveChannelR = bObserveChannelR && IsChannelUsed(CapComponent, GET_MEMBER_NAME_CHECKED(UCameraObserver, bObserveChannelR));
	bObserveChannelG = bObserveChannelG && IsChannelUsed(CapComponent, GET_MEMBER_NAME_CHECKED(UCameraObserver, bObserveChannelG));
	bObserveChannelB = bObserveChannelB && IsChannelUsed(CapComponent, GET_MEMBER_NAME_CHECKED(UCameraObserver, bObserveChannelB));
	bObserveChannelA = bObserveChannelA && IsChannelUsed(CapComponent, GET_MEMBER_NAME_CHECKED(UCameraObserver, bObserveChannelA));
}

FString UCameraObserver::GenerateId() const
{	
	USceneCaptureComponent2D* CapComponent = Cast<USceneCaptureComponent2D>(SceneCaptureCompRef.GetComponent(Cast<UActorComponent>(GetOuter())->GetOwner()));
	FString					  Output = FString("Camera"); 

	//Add CaptureSource Enum to Id
	if (CapComponent)
	{
		Output = Output.Append("_").Append(UEnum::GetValueAsString<ESceneCaptureSource>(CapComponent->CaptureSource));

		// Add Render Target Enum to Id
		if (CapComponent->TextureTarget)
		{
			Output = Output.Append("_").Append(UEnum::GetValueAsString<ETextureRenderTargetFormat>(CapComponent->TextureTarget->RenderTargetFormat));
		}
	}
	Output.Append("_");
	//Add channels to Id
	if(bObserveChannelR)
	{
		Output = Output.Append("R");
	}
	
	if(bObserveChannelG)
	{
		Output = Output.Append("G");
	}

	if(bObserveChannelB)
	{
		Output = Output.Append("B");
	}

	if(bObserveChannelA)
	{
		Output = Output.Append("A");
	}

	//Add width and height
	Output = Output.Appendf(TEXT("_W%d_H%d"), Width, Height); // Width and Height
	return Output;
}
