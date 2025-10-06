from typing import List, Tuple, Dict, Optional, Set
import torch, torch.nn as nn
import torchaudio
import torchvision.models as models
from dataloader import ensure_mono, pad_or_trim

def process_audio(target_len, target_sr):
        def _process_audio(wav_path):
            
            wav,sr = torchaudio.load(wav_path)
            wav = ensure_mono(wav)
            wav = torchaudio.functional.resample(wav, sr,target_sr)
            wav = pad_or_trim(wav, target_len * target_sr)
            return wav
        return _process_audio


def _resolve_efficientnet_weights(weights):
    try:
        return getattr(models, "EfficientNet_B0_Weights").IMAGENET1K_V1 if weights=="IMAGENET1K_V1" else weights
    except Exception:
        return weights

class Extract3channels(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super(Extract3channels, self).__init__()
        self.mel_spectorgram = torchaudio.transforms.MelSpectrogram(*args, **kwargs)

    def forward(self, x):
        x = self.mel_spectorgram(x)
        x = torch.log10(x+1)
        delta = torchaudio.functional.compute_deltas(x)
        delta_delta = torchaudio.functional.compute_deltas(delta)
        return torch.stack([x, delta, delta_delta], dim=1)

class Compose(torch.nn.Module):
    def __init__(self, transforms):
        super(Compose, self).__init__()
        self.transforms = transforms
        
    def forward(self, x):
        for transform in self.transforms:
            x = transform(x)
        return x
   
class NormalizePerUtterance(nn.Module):
        def __init__(self):
            super().__init__()
        def forward(self,x):
            print(x.shape)
            mean = x.mean(dim=(1,2))[:,None,None]
            std = x.std(dim=(1,2))[:,None,None]
            x_normalized =(x- mean)/std
            return x_normalized    

class EfficientNetB0(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 6,
        weights="IMAGENET1K_V1",
        freeze_backbone: bool = False,
        freeze_until: Optional[int] = None,
        dropout_rate: float = 0.3,
    ):
        super().__init__()
        weights = _resolve_efficientnet_weights(weights)
        self.efficientnet = models.efficientnet_b0(weights=weights)

        if in_channels != 3:
            self.efficientnet.features[0][0] = nn.Conv2d(
                in_channels, 32, kernel_size=3, stride=2, padding=1, bias=False
            )

        in_features = self.efficientnet.classifier[1].in_features
        self.efficientnet.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features, num_classes)
        )

        if freeze_backbone:
            for p in self.efficientnet.features.parameters(): p.requires_grad = False
        elif freeze_until is not None:
            for i, block in enumerate(self.efficientnet.features):
                if i < freeze_until:
                    for p in block.parameters(): p.requires_grad = False

    def forward(self, x): return self.efficientnet(x)
    def get_trainable_params(self): return [p for p in self.parameters() if p.requires_grad]



def inference(model,transforms,id_to_label) :
    n_mels=128,
    n_fft=512,                    
    hop_length=160, 
    def _inference(wave):
        wave = transforms(wave)
        wave = model(wave)
        return id_to_label[int(wave.to("cpu").softmax(0).argmax())]

    return _inference


model = torch.load("best_model.pth", weights_only=False, map_location="cuda" if torch.cuda.is_available() else "cpu")


transform = [
    process_audio(3,16000),
    Extract3channels(n_mels=128, n_fft=512, hop_length=160),
    NormalizePerUtterance()
]
transform = Compose(transform)

## to do later id_to_label 
id2label = [
"ANG",  # Angry
"SAD",  # Sad
"HAP",  # Happy
"FEA",  # Fear
"DIS",  # Disgust
"NEU"   # Neutral
]



