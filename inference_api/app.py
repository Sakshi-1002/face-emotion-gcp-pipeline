from fastapi import FastAPI
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
def predict_emotion():
    # Dummy response (for now)
    return {"emotion": "neutral", "confidence": 0.99}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
