from pathlib import Path
from datetime import datetime
import pandas as pd
import kagglehub
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from extract_classes import iter_wavs, detect_and_parse
from dataloader import AudioDataset
from utils import Compose, Extract3channels, training_iterator, train, test
from models import TestModel

# HYPERPARAMETERS == START ==
SAMPLING_RATE = 48000
CLIP_LENGTH = 3
WINDOW_SIZE = 400
HOP_LENGTH = 128
N_MELS = 128
# preprocessing hyperparameters

SPLITS = [0.8, 0.1, 0.1]
BATCH_SIZE = 8
EPOCHS = 1
TESTING_FREQUENCY = 1
# traininig hyperparameter

transforms = Compose([ # transformations
    Extract3channels(n_fft=WINDOW_SIZE, hop_length=HOP_LENGTH, n_mels=N_MELS)
    # torchaudio.transforms.MelSpectrogram()
])

eval_metrics = {
    
}

# HYPERPARAMETERS == END ==

path = kagglehub.dataset_download("dmitrybabko/speech-emotion-recognition-en")

dataset_df = pd.DataFrame([
        (wav_path, detect_and_parse(wav_path))
        for wav_path in iter_wavs(Path(path))
    ], columns=["paths", "emotion"]).dropna()
dataset_df["emotion"] = dataset_df["emotion"].replace({col:i for i, col in enumerate(dataset_df["emotion"].unique())}) # enocding classes
NUM_CLASSES = len(dataset_df["emotion"].unique())

dataset = AudioDataset(dataset_df, SAMPLING_RATE, CLIP_LENGTH, NUM_CLASSES)
train_dataset, test_dataset, val_dataset = random_split(dataset, SPLITS)
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True)


# model parameters == start
model = TestModel(3, NUM_CLASSES)#...
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