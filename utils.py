from math import ceil
import torchaudio
import torch
from tqdm import tqdm

class Compose(torch.nn.Module):
    def __init__(self, transforms):
        super(Compose, self).__init__()
        self.transforms = transforms
        
    def forward(self, x):
        for transform in self.transforms:
            x = transform(x)
        return x
    
class Extract3channels(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super(Extract3channels, self).__init__()
        self.mel_spectorgram = torchaudio.transforms.MelSpectrogram(*args, **kwargs)

    def forward(self, x):
        x = self.mel_spectorgram(x)
        delta = torchaudio.functional.compute_deltas(x)
        delta_delta = torchaudio.functional.compute_deltas(delta)
        return torch.stack([x, delta, delta_delta], dim=1)

def training_iterator(epochs, testing_freq):
    testing_count = ceil(epochs / testing_freq)
    train_counter = 0
    test_counter = 0
    for i in range(epochs + testing_count):
        if train_counter > 0 and train_counter % testing_freq == 0:
            test_counter += 1
            yield "testing", test_counter

        else:
            train_counter += 1
            yield "training", train_counter

def train(model, dataloader, transforms, loss_fn, eval_metrics, optim):
    model.train()
    outs = []
    total_loss = 0
    for inputs, labels in tqdm(dataloader):
        model.zero_grad()
        
        inputs = transforms(inputs)
        
        out = model(inputs)
        batch_size = out.shape[0]
        out = out.reshape(batch_size, -1)

        loss = loss_fn(out, labels)
        total_loss += loss.item()
        
        outs.append(out)
        outs.append(labels)

        loss.backward()
        optim.step()
        
    scores = {}
    for metric, fn in eval_metrics.items():
        assert isinstance(outs, list)
        
        if "logits" in metric:
            score = fn(torch.stack(outs), torch.stack(labels))
        else:
            outs = torch.stack(outs).softmax(1).argmax(1)
            score = fn(outs, torch.stack(labels))
        
        scores[metric] = score
    scores["loss"] = total_loss / len(dataloader)
    
    return scores


@torch.no_grad()
def test(model, dataloader, transforms, loss_fn, eval_metrics):
    model.eval()
    outs = []
    total_loss = 0
    for inputs, labels in tqdm(dataloader):

        inputs = transforms(inputs)
        
        out = model(inputs)
        batch_size = out.shape[0]
        out = out.reshape(batch_size, -1)

        loss = loss_fn(out, labels)
        total_loss += loss.item()

        outs.append(out)
        outs.append(labels)
        
    scores = {}
    for metric, fn in eval_metrics.items():
        assert isinstance(outs, list)
        
        if "logits" in metric:
            score = fn(torch.stack(outs), torch.stack(labels))
        else:
            outs = torch.stack(outs).softmax(1).argmax(1)
            score = fn(outs, torch.stack(labels))
        
        scores[metric] = score
    scores["loss"] = total_loss / len(dataloader)
    return scores