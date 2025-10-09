from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import kagglehub
import torch
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedShuffleSplit
from extract_classes import iter_wavs, detect_and_parse
from dataloader import AudioDataset, collate_fixed_length
from utils import Compose, Extract3channels, training_iterator, train, test
from models import create_model
from sklearn.metrics import accuracy_score, f1_score

import warnings

warnings.filterwarnings("ignore")

# ========== HYPERPARAMETERS ==========

SAMPLING_RATE = 16000
CLIP_LENGTH = 3
WINDOW_SIZE = 400
HOP_LENGTH = 160
N_MELS = 128

# Training
BATCH_SIZE = 16
EPOCHS = 20
TESTING_FREQUENCY = 1
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BACKBONE = "resnet50"  # or "resnet18", "resnet34", "vgg11", "vgg16"

# Metrics
eval_metrics = {
    "accuracy": accuracy_score,
    "f1": lambda y_true, y_pred: f1_score(y_true, y_pred, average="macro"),
}

# Features pipeline
transforms = Compose(
    [
        Extract3channels(
            sample_rate=SAMPLING_RATE,
            n_fft=WINDOW_SIZE,
            hop_length=HOP_LENGTH,
            n_mels=N_MELS,
        )
    ]
)

# ========== DATA ==========
path = kagglehub.dataset_download("dmitrybabko/speech-emotion-recognition-en")

dataset_df = (
    pd.DataFrame(
        [(wav_path, detect_and_parse(wav_path)) for wav_path in iter_wavs(Path(path))],
        columns=["paths", "emotion"],
    )
    .dropna()
    .sample(frac=1, random_state=42)
    .reset_index(drop=True)
)

# Encode classes
classes = {c: i for i, c in enumerate(dataset_df["emotion"].unique())}
dataset_df["emotion"] = dataset_df["emotion"].map(classes).astype(int)
NUM_CLASSES = len(classes)

# Build dataset (labels as indices)
dataset = AudioDataset(
    dataset_df, SAMPLING_RATE, CLIP_LENGTH, NUM_CLASSES, device="cpu"
)  # keep CPU for transforms

# Stratified 80/10/10 split
labels = dataset_df["emotion"].values
sss1 = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, temp_idx = next(sss1.split(np.zeros(len(labels)), labels))
temp_labels = labels[temp_idx]
sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=42)
val_rel, test_rel = next(sss2.split(np.zeros(len(temp_labels)), temp_labels))
val_idx = temp_idx[val_rel]
test_idx = temp_idx[test_rel]

from torch.utils.data import Subset

train_dataset = Subset(dataset, train_idx)
val_dataset = Subset(dataset, val_idx)
test_dataset = Subset(dataset, test_idx)

train_dataloader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fixed_length
)
val_dataloader = DataLoader(
    val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fixed_length
)
test_dataloader = DataLoader(
    test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fixed_length
)

# Insights
print("original dataset distribution")
print(dataset_df["emotion"].value_counts())
print("train dataset distribution")
print(dataset_df.iloc[train_idx]["emotion"].value_counts())
print("val dataset distribution")
print(dataset_df.iloc[val_idx]["emotion"].value_counts())
print("test dataset distribution")
print(dataset_df.iloc[test_idx]["emotion"].value_counts())
print(classes)

# ========== MODEL / LOSS / OPT ==========
model = create_model(
    BACKBONE, in_channels=3, num_classes=NUM_CLASSES, pretrained=True
).to(DEVICE)

# Class weights (inverse frequency) to help macro-F1
counts = (
    dataset_df.iloc[train_idx]["emotion"]
    .value_counts()
    .reindex(range(NUM_CLASSES), fill_value=0)
    .values
)
weights = counts.sum() / np.maximum(1, counts)
weights = torch.tensor(weights, dtype=torch.float32, device=DEVICE)

loss_fn = CrossEntropyLoss(weight=weights)
optim = AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

# ========== TRAIN / EVAL LOOP ==========
summary = []
start = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

for mode, epoch in training_iterator(EPOCHS, TESTING_FREQUENCY):
    if mode == "training":
        scores = train(
            model,
            train_dataloader,
            transforms,
            loss_fn,
            eval_metrics,
            optim,
            device=DEVICE,
        )
    else:
        # validate during training
        scores = test(
            model, val_dataloader, transforms, loss_fn, eval_metrics, device=DEVICE
        )
    summary.append({"mode": mode, "epoch": epoch, **scores})

# Final test once
final_scores = test(
    model, test_dataloader, transforms, loss_fn, eval_metrics, device=DEVICE
)
summary.append({"mode": "final_test", "epoch": EPOCHS, **final_scores})

end = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
pd.DataFrame(summary).to_csv(start + " -- " + end + ".csv", index=False)
