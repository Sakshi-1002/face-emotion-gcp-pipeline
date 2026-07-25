# 🧠 Face & Emotion Detection API

A FastAPI service that detects faces in an uploaded image and classifies the emotion being expressed — powered by a pretrained deep learning model.

## What this actually does (tested and working)

Upload an image → the API detects the face → runs it through a pretrained CNN → returns a full probability breakdown across 7 emotions (angry, disgust, fear, happy, sad, surprise, neutral) plus the dominant one.

```bash
curl -X POST "http://localhost:8000/predict" -F "file=@your_photo.jpg"
```

```json
{
  "emotion": "happy",
  "confidence": 94.2,
  "all_emotions": {
    "angry": 0.3, "disgust": 0.0, "fear": 0.1,
    "happy": 94.2, "sad": 1.1, "surprise": 3.8, "neutral": 0.5
  },
  "face_region": {"x": 143, "y": 94, "w": 247, "h": 247}
}
```

If no face is detected, it returns a clean `422` error instead of guessing.

## Tech Stack

| Category | Tools |
|---|---|
| Backend / API | FastAPI, Uvicorn |
| Face detection + emotion classification | DeepFace (pretrained CNN, TensorFlow backend) |
| Image handling | OpenCV, NumPy |

## Getting Started

```bash
git clone https://github.com/Sakshi-1002/face-emotion-gcp-pipeline.git
cd face-emotion-gcp-pipeline
pip install -r requirements.txt
python inference_api/app.py
```

Then open `http://localhost:8000/docs` for an interactive interface to test it — upload any photo and see the live prediction.

## ✅ Built

- Working FastAPI service with `/` health check and `/predict` endpoints
- Real emotion inference using a pretrained model (no training required to get accurate results — the model was trained on large facial expression datasets and is used here for inference)
- Input validation and graceful error handling for invalid images or no-face-detected cases

## 🔜 Roadmap

- Custom model fine-tuned on a specific dataset, rather than relying solely on the pretrained model
- Data collection & preprocessing pipeline for custom training data
- Containerize with Docker and deploy to Google Cloud Run
- Add monitoring (Cloud Monitoring / Prometheus) for latency and prediction drift
- MLflow integration for experiment tracking once custom training begins

## What I learned

Building this taught me the practical difference between "using AI" and "building AI infrastructure" — wiring a pretrained model into a real API with proper input validation and error handling is its own skill, separate from the model itself. It also clarified for me what "production-ready" actually requires beyond just a working prediction function: validation, clear error states, and an honest accounting of what's inference-only versus what would need custom training.

---
*Built by [Sakshi Shirude](https://github.com/Sakshi-1002)*
