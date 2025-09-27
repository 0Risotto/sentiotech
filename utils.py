import torchaudio
import torch

class Compose(torch.nn.Module):
    def __init__(self, transforms):
        self.transforms = transforms
        
    def forward(self, x):
        for transform in self.transforms:
            x = transform(x)
        return x
    
class Extract3channels(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        self.mel_spectorgram = torchaudio.transforms.MelSpectrogram(*args, **kwargs)

    def forward(self, x):
        x = self.mel_spectorgram(x)
        delta = torchaudio.functional.compute_deltas(x)
        delta_delta = torchaudio.functional.compute_deltas(delta)
        return torch.stack([x, delta, delta_delta], dim=1)

def train(model, dataloader, loss_fn, eval_metrics, optim, num_classes):
    for inputs, labels in dataloader:
        
        inputs = Compose(inputs)
        labels = torch.nn.functional.one_hot(labels.long(), num_classes)