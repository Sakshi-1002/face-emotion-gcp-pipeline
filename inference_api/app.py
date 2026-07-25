from fastapi import FastAPI, UploadFile, File, HTTPException
from deepface import DeepFace
import numpy as np
import cv2
import uvicorn

# Initialize app
app = FastAPI(
    title="Face & Emotion Detection API",
    version="1.0",
    description="FastAPI service for real-time face and emotion detection"
)

@app.get("/")
def root():
    return {"status": "ok", "message": "Face & Emotion API is running 🚀"}

@app.post("/predict")
async def predict_emotion(file: UploadFile = File(...)):
    # Read uploaded image bytes into an OpenCV image
    contents = await file.read()
    npimg = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    if img is None:
        raise HTTPException(status_code=400, detail="Could not decode image. Please upload a valid image file.")

    try:
        result = DeepFace.analyze(
            img_path=img,
            actions=["emotion"],
            enforce_detection=True  # raises if no face is found, instead of guessing
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=f"No face detected in image: {str(e)}")

    # DeepFace returns a list (one entry per detected face); we return the first face
    face_result = result[0]
    emotions = {k: round(float(v), 2) for k, v in face_result["emotion"].items()}

    return {
        "emotion": face_result["dominant_emotion"],
        "confidence": round(float(emotions[face_result["dominant_emotion"]]), 2),
        "all_emotions": emotions,
        "face_region": face_result["region"]
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
