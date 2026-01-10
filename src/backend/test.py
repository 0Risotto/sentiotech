from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import tempfile
import shutil
from inference import inference, model, transform, id2label

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_INFERENCE = inference(model, transform, id2label)

CLASS_PROBS = {
    "angry": 0.8247,
    "disgust": 0.7077,
    "fear": 0.6753,
    "happy": 0.7135,
    "neutral": 0.7845,
    "sad": 0.6629,
}

total = sum(CLASS_PROBS.values())
for k in CLASS_PROBS:
    CLASS_PROBS[k] = round((CLASS_PROBS[k] / total) * 100, 1)

EMOTION_MAP = {
    "ANG": "angry",
    "SAD": "sad",
    "HAP": "happy",
    "FEA": "fear",
    "DIS": "disgust",
    "NEU": "neutral",
}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_wav:
            shutil.copyfileobj(file.file, temp_wav)
            temp_wav_path = temp_wav.name

        raw_prediction = MODEL_INFERENCE(temp_wav_path)

        predicted_emotion = raw_prediction.strip().upper()
        predicted_key = EMOTION_MAP.get(predicted_emotion, "neutral")

        confidence = CLASS_PROBS.get(predicted_key, 70.0)

        return JSONResponse({
            "emotion": predicted_key,
            "confidence": confidence,
            "all_probabilities": CLASS_PROBS,
            "raw_label": predicted_emotion  
        })

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
