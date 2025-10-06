from inference import inference, model, transform, id2label

MODEL_INFERENCE = inference(model, transform, id2label)

wav_path = "2.wav"  
prediction = MODEL_INFERENCE(wav_path)
print(f"Predicted emotion: {prediction}")
