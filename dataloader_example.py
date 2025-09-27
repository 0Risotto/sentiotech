from dataloader import make_dataloaders


import kagglehub

path = kagglehub.dataset_download("dmitrybabko/speech-emotion-recognition-en")

DATA_ROOT = path


train_dl, val_dl, test_dl, id2label, label2id = make_dataloaders(
    root=DATA_ROOT,
    batch_size=4,
    num_workers=0,
)

print("Label mapping:", id2label)
batch = next(iter(train_dl))
waves, labels = batch

print("Waves shape:", waves.shape)
print("Labels:", labels)


for i, lab in enumerate(labels.tolist()):
    print(f"Sample {i}: class={id2label[lab]}, waveform shape={waves[i].shape}")
