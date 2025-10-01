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
    test_counter = 0
    for train_counter in range(1, epochs + 1):

        yield "training", train_counter

        if train_counter > 1 and train_counter % testing_freq == 0:
            test_counter += 1
            yield "testing", test_counter

def train(model, dataloader, transforms, loss_fn, eval_metrics, optim):
    model.train()
    outs = []
    labels = []
    total_loss = 0
    for inputs, label in tqdm(dataloader):
        model.zero_grad()
        
        inputs = transforms(inputs)
        
        out = model(inputs)
        batch_size = out.shape[0]
        out = out.reshape(batch_size, -1)

        loss = loss_fn(out, label)
        total_loss += loss.item()
        
        outs.append(out)
        labels.append(label)

        loss.backward()
        optim.step()
        
    scores = {}
    outs = torch.cat(outs, dim=0)
    labels = torch.cat(labels, dim=0)
    for metric, fn in eval_metrics.items():

        if "logits" in metric:
            score = fn(outs, labels)
        else:
            score = fn(outs.softmax(1).argmax(1), labels.argmax(1))

        scores[metric] = score
        
    for metric, score in scores.items():
        print(f"{metric}: {round(score, 3) if isinstance(score, float) else score}", end="||")
    return scores


@torch.no_grad()
def test(model, dataloader, transforms, loss_fn, eval_metrics):
    model.eval()
    outs = []
    labels = []
    total_loss = 0
    for inputs, label in tqdm(dataloader):

        inputs = transforms(inputs)
        
        out = model(inputs)
        batch_size = out.shape[0]
        out = out.reshape(batch_size, -1)

        loss = loss_fn(out, label)
        total_loss += loss.item()

        outs.append(out)
        labels.append(label)
        
    scores = {}
    outs = torch.cat(outs, dim=0)
    labels = torch.cat(labels, dim=0)
    for metric, fn in eval_metrics.items():

        if "logits" in metric:
            score = fn(outs, labels)
        else:
            score = fn(outs.softmax(1).argmax(1), labels.argmax(1))
        
        scores[metric] = score
        
    for metric, score in scores.items():
        print(f"{metric}: {round(score, 3) if isinstance(score, float) else score}", end="||")
        
    scores["loss"] = total_loss / len(dataloader)
    return scores