import os
from pathlib import Path
from typing import List, Tuple, Dict
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torchaudio
import soundfile as sf

TARGET_SR = 16000
CLIP_SECONDS = 2.0


def load_with_soundfile(path: str) -> Tuple[np.ndarray, int]:
    wav, sr = sf.read(path, dtype="float32")
    if np.isscalar(wav):
        wav = np.array([wav], dtype=np.float32)
    return wav, sr


def ensure_mono(wav: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
    if wav.ndim == 2:
        if isinstance(wav, torch.Tensor):
            wav = wav.mean(dim=0)  # [T]
        else:
            wav = wav.mean(axis=1)  # [T]
    return wav


def resample_if_needed(wav: np.ndarray, sr: int, target_sr: int) -> np.ndarray:
    if sr == target_sr:
        return wav
    wav_t = torch.from_numpy(wav)
    wav_rs = torchaudio.functional.resample(wav_t, sr, target_sr)
    return wav_rs.numpy()


def pad_or_trim(
    wav: np.ndarray | torch.Tensor, target_len: int
) -> np.ndarray | torch.Tensor:
    n = wav.shape[0]
    if n > target_len:
        return wav[:target_len]
    if n < target_len:
        if isinstance(wav, np.ndarray):
            return np.pad(wav, (0, target_len - n), mode="constant")
        else:
            return torch.cat(
                [wav, torch.zeros(target_len - n, dtype=wav.dtype, device=wav.device)]
            )
    return wav


class AudioFolderDataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        target_sr: int = TARGET_SR,
        clip_seconds: float = CLIP_SECONDS,
        extensions: Tuple[str, ...] = (".wav", ".flac", ".mp3", ".ogg"),
        sort_files: bool = True,
    ):
        self.root = Path(root)
        self.target_sr = target_sr
        self.clip_seconds = clip_seconds
        self.target_len = int(self.clip_seconds * self.target_sr)
        self.extensions = tuple(e.lower() for e in extensions)

        class_dirs = [p for p in self.root.iterdir() if p.is_dir()]
        if not class_dirs:
            raise RuntimeError(f"No class folders found in {self.root}")
        class_dirs.sort(key=lambda p: p.name)

        self.classes: List[str] = [p.name for p in class_dirs]
        self.label2id: Dict[str, int] = {c: i for i, c in enumerate(self.classes)}
        self.id2label: Dict[int, str] = {i: c for c, i in self.label2id.items()}

        items: List[Tuple[Path, int]] = []
        for cdir in class_dirs:
            label_id = self.label2id[cdir.name]
            files = [
                f
                for f in cdir.glob("*")
                if f.is_file() and f.suffix.lower() in self.extensions
            ]
            if sort_files:
                files.sort()
            items.extend((f, label_id) for f in files)

        if not items:
            raise RuntimeError(
                f"No audio files with {self.extensions} found in {self.root}"
            )
        self.items: List[Tuple[Path, int]] = items

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        path, label = self.items[idx]
        wav, sr = load_with_soundfile(str(path))
        wav = ensure_mono(wav)
        wav = resample_if_needed(wav, sr, self.target_sr)
        wav = pad_or_trim(wav, int(self.clip_seconds * self.target_sr))
        return torch.from_numpy(wav.astype(np.float32)), label


class AudioDataset(Dataset):
    """
    DataFrame with two columns: [path, label_index].
    Returns (wave[T], label_index).
    """

    def __init__(self, df, target_sr, clip_seconds, num_classes, device="cpu"):
        self._df = df
        self.target_sr = target_sr
        self.clip_seconds = clip_seconds  # seconds
        self.target_len = int(self.clip_seconds * self.target_sr)  # samples
        self.num_classes = num_classes
        self.device = device

    def __len__(self):
        return len(self._df)

    def __getitem__(self, idx):
        wav_path, label = self._df.iloc[idx]
        label = int(label)

        wav, sr = torchaudio.load(wav_path)  # [C, T] or [1, T]
        wav = wav.mean(dim=0) if wav.ndim == 2 else wav  # [T]
        if sr != self.target_sr:
            wav = torchaudio.functional.resample(wav, sr, self.target_sr)
        wav = pad_or_trim(wav, self.target_len)  # [T]

        # Keep waveform CPU to play nice with CPU-only torchaudio transforms
        return wav.contiguous(), torch.tensor(label, dtype=torch.long)

    def to(self, device):
        self.device = device

    def data(self):
        return self._df


# Collates unchanged
def collate_fixed_length(
    batch: List[Tuple[torch.Tensor, int]],
) -> Tuple[torch.Tensor, torch.Tensor]:
    waves, labels = zip(*batch)
    waves = torch.stack(waves, dim=0)  # [B, T]
    labels = torch.tensor(labels, dtype=torch.long)  # [B]
    return waves, labels


def collate_variable_length(
    batch: List[Tuple[torch.Tensor, int]],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    waves, labels = zip(*batch)
    lengths = torch.tensor([w.shape[0] for w in waves], dtype=torch.long)
    waves_padded = torch.nn.utils.rnn.pad_sequence(waves, batch_first=True)
    labels = torch.tensor(labels, dtype=torch.long)
    return waves_padded, lengths, labels


def make_splits(
    ds: Dataset,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> Tuple[Dataset, Dataset, Dataset]:
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6
    n = len(ds)
    g = torch.Generator().manual_seed(seed)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    n_test = n - n_train - n_val
    return random_split(ds, [n_train, n_val, n_test], generator=g)


def make_dataloaders(
    root: str | Path,
    batch_size: int = 32,
    num_workers: int = os.cpu_count() or 4,
    pin_memory: bool = True,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
    target_sr: int = TARGET_SR,
    clip_seconds: float = CLIP_SECONDS,
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[int, str], Dict[str, int]]:
    ds = AudioFolderDataset(root=root, target_sr=target_sr, clip_seconds=clip_seconds)
    train_ds, val_ds, test_ds = make_splits(
        ds, train_ratio, val_ratio, test_ratio, seed=seed
    )

    collate = collate_fixed_length
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate,
    )

    return train_loader, val_loader, test_loader, ds.id2label, ds.label2id
