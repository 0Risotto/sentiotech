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
    """
    Returns [B, 3, n_mels, T'] = [mel, delta, delta-delta]
    You MUST pass sample_rate, n_fft, hop_length, n_mels in the ctor.
    """

    def __init__(self, *args, **kwargs):
        super(Extract3channels, self).__init__()
        self.mel_spectrogram = torchaudio.transforms.MelSpectrogram(*args, **kwargs)
        self.to_db = torchaudio.transforms.AmplitudeToDB(stype="power")

    def forward(self, x):  # x: [B, T] (CPU recommended)
        m = self.mel_spectrogram(x)  # [B, n_mels, T']
        m = self.to_db(m)  # log-mel
        d1 = torchaudio.functional.compute_deltas(m)
        d2 = torchaudio.functional.compute_deltas(d1)
        return torch.stack([m, d1, d2], dim=1)  # [B, 3, n_mels, T']


def training_iterator(epochs, testing_freq):
    # test at EACH epoch (including 1) if divisible by testing_freq
    for ep in range(1, epochs + 1):
        yield "training", ep
        if ep % testing_freq == 0:
            yield "testing", ep


def _labels_to_indices(y: torch.Tensor) -> torch.Tensor:
    # Accept either indices or one-hot; return indices
    return y if y.ndim == 1 else y.argmax(1)


def _flatten_logits_if_needed(logits: torch.Tensor) -> torch.Tensor:
    # Expect [B, C]. If [B, C, 1, 1] etc., flatten safely.
    if logits.ndim > 2:
        logits = torch.flatten(logits, start_dim=1)
    return logits


def train(model, dataloader, transforms, loss_fn, eval_metrics, optim, device="cpu"):
    model.train()
    preds_all, labels_all = [], []
    total_loss = 0.0

    for inputs, labels in tqdm(dataloader):
        optim.zero_grad()

        # Keep transforms on CPU to avoid CUDA-only issues in torchaudio
        inputs = inputs.float().cpu()  # [B, T]
        feats = transforms(inputs)  # [B, 3, n_mels, T']
        feats = feats.to(device)

        logits = model(feats)  # [B, C] expected
        logits = _flatten_logits_if_needed(logits)

        labels_idx = _labels_to_indices(labels).long().to(device)

        loss = loss_fn(logits, labels_idx)
        total_loss += loss.item()
        loss.backward()
        optim.step()

        preds_all.append(logits.detach().cpu().argmax(1))
        labels_all.append(labels_idx.detach().cpu())

    preds_all = torch.cat(preds_all).numpy()
    labels_all = torch.cat(labels_all).numpy()

    scores = {name: fn(labels_all, preds_all) for name, fn in eval_metrics.items()}
    scores["loss"] = total_loss / max(1, len(dataloader))
    print(" | ".join(f"{k}:{scores[k]:.3f}" for k in scores))
    return scores


@torch.no_grad()
def test(model, dataloader, transforms, loss_fn, eval_metrics, device="cpu"):
    model.eval()
    preds_all, labels_all = [], []
    total_loss = 0.0

    for inputs, labels in tqdm(dataloader):
        inputs = inputs.float().cpu()
        feats = transforms(inputs).to(device)

        logits = model(feats)
        logits = _flatten_logits_if_needed(logits)

        labels_idx = _labels_to_indices(labels).long().to(device)

        total_loss += loss_fn(logits, labels_idx).item()
        preds_all.append(logits.detach().cpu().argmax(1))
        labels_all.append(labels_idx.detach().cpu())

    preds_all = torch.cat(preds_all).numpy()
    labels_all = torch.cat(labels_all).numpy()

    scores = {name: fn(labels_all, preds_all) for name, fn in eval_metrics.items()}
    scores["loss"] = total_loss / max(1, len(dataloader))
    print(" | ".join(f"{k}:{scores[k]:.3f}" for k in scores))
    return scores
