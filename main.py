from pathlib import Path
from datetime import datetime
import pandas as pd
import kagglehub
import torch
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from extract_classes import iter_wavs, detect_and_parse
from dataloader import AudioDataset
from utils import Compose, Extract3channels, training_iterator, train, test
from models import TestModel
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score

import warnings
warnings.filterwarnings("ignore")
# HYPERPARAMETERS == START ==
SAMPLING_RATE = 48000
CLIP_LENGTH = 3
WINDOW_SIZE = 400
HOP_LENGTH = 128
N_MELS = 128
# preprocessing hyperparameters

SPLITS = [0.01, 0.01, 0.98]
BATCH_SIZE = 8
EPOCHS = 1
TESTING_FREQUENCY = 1
DEVICE = "cpu"
# traininig hyperparameter

transforms = Compose([ # transformations
    Extract3channels(n_fft=WINDOW_SIZE, hop_length=HOP_LENGTH, n_mels=N_MELS)
    # torchaudio.transforms.MelSpectrogram()
])

eval_metrics = {
    "accuracy" : accuracy_score,
    "f1" : lambda x, y: f1_score(x, y, average="macro"),
    # "f1" : f1_score,
    # "recall" : recall_score,
    # "precision" : precision_score
}

# HYPERPARAMETERS == END ==

path = kagglehub.dataset_download("dmitrybabko/speech-emotion-recognition-en")

dataset_df = pd.DataFrame([
        (wav_path, detect_and_parse(wav_path))
        for wav_path in iter_wavs(Path(path))
    ], columns=["paths", "emotion"]).dropna().sample(frac=1).reset_index(drop=True)

CLASSES = {col:i for i, col in enumerate(dataset_df["emotion"].unique())}
dataset_df["emotion"] = dataset_df["emotion"].replace({col:i for i, col in enumerate(dataset_df["emotion"].unique())}) # enocding classes
NUM_CLASSES = len(CLASSES)

dataset = AudioDataset(dataset_df, SAMPLING_RATE, CLIP_LENGTH, NUM_CLASSES, DEVICE)
train_dataset, test_dataset, val_dataset = random_split(dataset, SPLITS)
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True)

#insights about the dataset
print("original dataset distribution")
print(dataset_df["emotion"].value_counts()) # shows the frequency of each class

print("train dataset distribution")
print(dataset_df.iloc[train_dataset.indices]["emotion"].value_counts()) # shows the frequency of each class

print("test dataset distribution")
print(dataset_df.iloc[train_dataset.indices]["emotion"].value_counts()) # shows the frequency of each class

print(CLASSES) # shows each class with it encoded number

#model parameters == start

model = TestModel(3, NUM_CLASSES)#...
model = model.to(DEVICE)
optim = AdamW(model.parameters(), lr=1e-3) #...
loss_fn = CrossEntropyLoss()#...
# model parameters == end

summary = []
start = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

for mode, epoch in training_iterator(EPOCHS, TESTING_FREQUENCY):
    if mode == "training":
        scores = train(model, train_dataloader, transforms, loss_fn, eval_metrics, optim)
    if mode == "testing":
        scores = test(model, test_dataloader, transforms, loss_fn, eval_metrics)
    
    summary.append({"mode":mode, "epoch":epoch, **scores})

end = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

pd.DataFrame(summary).to_csv(start + " -- " + end + ".csv", index=False)

torch.save(model.state_dict(), "emotion_model.pth")
