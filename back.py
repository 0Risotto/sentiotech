from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import tempfile
import shutil
import os
from inference import inference, mock_inference, model, transforms, id2label

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Use mock inference for testing (replace with real inference when you have trained weights)
MODEL_INFERENCE = mock_inference(model, transforms, id2label)


@app.get("/")
async def root():
    return {"message": "Sentiotech Emotion Recognition API", "status": "running"}


@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": True}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    print(
        f"Received file: {file.filename}, content type: {file.content_type}, size: {file.size}"
    )

    # Validate file
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")

    if file.size == 0:
        raise HTTPException(status_code=400, detail="File is empty")

    if file.size > 50 * 1024 * 1024:  # 50MB limit
        raise HTTPException(status_code=400, detail="File too large (max 50MB)")

    temp_wav_path = None
    try:
        # Create temporary file with proper extension
        file_extension = os.path.splitext(file.filename)[1].lower()
        if file_extension not in [".wav", ".mp3", ".m4a", ".flac", ".ogg"]:
            raise HTTPException(
                status_code=400, detail=f"Unsupported file type: {file_extension}"
            )

        with tempfile.NamedTemporaryFile(
            delete=False, suffix=file_extension
        ) as temp_wav:
            # Reset file pointer to beginning
            await file.seek(0)
            shutil.copyfileobj(file.file, temp_wav)
            temp_wav_path = temp_wav.name
            print(f"Saved temporary file to: {temp_wav_path}")

        # Verify file was saved correctly
        if not os.path.exists(temp_wav_path) or os.path.getsize(temp_wav_path) == 0:
            raise HTTPException(status_code=500, detail="Failed to save uploaded file")

        print(f"File saved successfully, size: {os.path.getsize(temp_wav_path)} bytes")

        # Run inference
        result = MODEL_INFERENCE(temp_wav_path)
        print(f"Model inference result: {result}")

        # Clean up temporary file
        if temp_wav_path and os.path.exists(temp_wav_path):
            os.unlink(temp_wav_path)

        # Handle both old format (string) and new format (dict)
        if isinstance(result, dict):
            return JSONResponse(result)
        else:
            # Legacy format - just return the emotion string
            return JSONResponse({"emotion": result})

    except HTTPException:
        # Re-raise HTTP exceptions
        if temp_wav_path and os.path.exists(temp_wav_path):
            os.unlink(temp_wav_path)
        raise
    except Exception as e:
        print(f"Error in predict endpoint: {str(e)}")
        if temp_wav_path and os.path.exists(temp_wav_path):
            os.unlink(temp_wav_path)
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")
