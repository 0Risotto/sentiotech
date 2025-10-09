import torch
import torchaudio
from utils import Compose, Extract3channels
from models import create_model
import tempfile
import os

# Model configuration (should match training configuration)
SAMPLING_RATE = 16000
CLIP_LENGTH = 3
WINDOW_SIZE = 400
HOP_LENGTH = 160
N_MELS = 128
BACKBONE = "resnet50"
NUM_CLASSES = 6

# Emotion class mapping
id2label = {0: "Angry", 1: "Disgust", 2: "Fear", 3: "Happy", 4: "Neutral", 5: "Sad"}

# Initialize transforms
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

# Initialize model
model = create_model(BACKBONE, in_channels=3, num_classes=NUM_CLASSES, pretrained=False)

# Load trained model weights (you'll need to update this path)
# model.load_state_dict(torch.load('path_to_your_trained_model.pth', map_location='cpu'))
model.eval()

# For testing purposes, we'll create a mock inference function
def mock_inference(model, transforms, id2label):
    """Mock inference function for testing without trained weights."""
    
    def predict_emotion(audio_path):
        """Mock predict emotion from audio file path."""
        try:
            # Load audio to verify it's valid
            waveform, sample_rate = torchaudio.load(audio_path)
            
            # Check if audio loaded successfully
            if waveform.shape[0] == 0 or waveform.shape[1] == 0:
                return {"error": "Audio file is empty or corrupted"}
            
            # Resample if necessary
            if sample_rate != SAMPLING_RATE:
                resampler = torchaudio.transforms.Resample(sample_rate, SAMPLING_RATE)
                waveform = resampler(waveform)
            
            # Take only the first channel if stereo
            if waveform.shape[0] > 1:
                waveform = waveform[0:1]
            
            # Pad or truncate to CLIP_LENGTH
            target_length = SAMPLING_RATE * CLIP_LENGTH
            if waveform.shape[1] > target_length:
                waveform = waveform[:, :target_length]
            else:
                padding = target_length - waveform.shape[1]
                waveform = torch.nn.functional.pad(waveform, (0, padding))
            
            # Extract features
            features = transforms(waveform)
            
            # Mock prediction (since we don't have trained weights)
            import random
            emotions = list(id2label.values())
            predicted_emotion = random.choice(emotions)
            
            # Create mock probabilities
            mock_probs = {}
            for emotion in emotions:
                if emotion == predicted_emotion:
                    mock_probs[emotion] = round(random.uniform(60, 90), 2)
                else:
                    mock_probs[emotion] = round(random.uniform(1, 15), 2)
            
            # Clean up temporary file
            if os.path.exists(audio_path):
                os.unlink(audio_path)
            
            return {
                "emotion": predicted_emotion,
                "confidence": mock_probs[predicted_emotion],
                "all_probabilities": mock_probs,
                "note": "This is a mock prediction for testing. Load trained model weights for real predictions."
            }
            
        except Exception as e:
            # Clean up temporary file in case of error
            if os.path.exists(audio_path):
                os.unlink(audio_path)
            return {"error": f"Error processing audio: {str(e)}"}
    
    return predict_emotion


def inference(model, transforms, id2label):
    """Create an inference function that processes audio files."""

    def predict_emotion(audio_path):
        """Predict emotion from audio file path."""
        try:
            # Load audio
            waveform, sample_rate = torchaudio.load(audio_path)

            # Resample if necessary
            if sample_rate != SAMPLING_RATE:
                resampler = torchaudio.transforms.Resample(sample_rate, SAMPLING_RATE)
                waveform = resampler(waveform)

            # Take only the first channel if stereo
            if waveform.shape[0] > 1:
                waveform = waveform[0:1]

            # Pad or truncate to CLIP_LENGTH
            target_length = SAMPLING_RATE * CLIP_LENGTH
            if waveform.shape[1] > target_length:
                waveform = waveform[:, :target_length]
            else:
                padding = target_length - waveform.shape[1]
                waveform = torch.nn.functional.pad(waveform, (0, padding))

            # Extract features
            features = transforms(waveform)

            # Make prediction
            with torch.no_grad():
                logits = model(features)
                probabilities = torch.softmax(logits, dim=1)
                predicted_class = torch.argmax(probabilities, dim=1).item()
                confidence = probabilities[0][predicted_class].item()

            emotion = id2label[predicted_class]

            # Clean up temporary file
            if os.path.exists(audio_path):
                os.unlink(audio_path)

            return {
                "emotion": emotion,
                "confidence": round(confidence * 100, 2),
                "all_probabilities": {
                    id2label[i]: round(probabilities[0][i].item() * 100, 2)
                    for i in range(len(id2label))
                },
            }

        except Exception as e:
            # Clean up temporary file in case of error
            if os.path.exists(audio_path):
                os.unlink(audio_path)
            return {"error": f"Error processing audio: {str(e)}"}

    return predict_emotion
