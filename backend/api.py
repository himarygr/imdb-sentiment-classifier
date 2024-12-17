from fastapi import FastAPI, Request
from pydantic import BaseModel
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import joblib
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from fastapi.middleware.cors import CORSMiddleware

# Initialize FastAPI app
app = FastAPI(title="IMDB Sentiment Analysis API")

# CORS settings
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"],  
)

# Load the trained model and vectorizer
MODEL_PATH = "model/sentiment_model.pkl"
VECTORIZER_PATH = "data/processed/tfidf_vectorizer.pkl"

model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)

# Serve static files (CSS, images, etc.)
#app.mount("/static", StaticFiles(directory="frontend/static"), name="static")

# Function to clean text
def clean_text(text: str):
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)  # Remove URLs
    text = re.sub(r'[^A-Za-z ]+', '', text)  # Remove special characters
    return text.lower()

# Input model schema
class ReviewRequest(BaseModel):
    review: str

# Route to serve the HTML page
@app.get("/", response_class=HTMLResponse)
def get_homepage():
    with open("frontend/index.html", "r") as file:
        return HTMLResponse(content=file.read())

@app.get("/test/")
def test_model():
    sample_text = ["The movie was great and amazing!"]  # Тестовый текст
    cleaned_text = clean_text(sample_text[0])
    tfidf_text = vectorizer.transform([cleaned_text])
    prediction = model.predict(tfidf_text)
    return {"cleaned_text": cleaned_text, "prediction": prediction[0]}

# Route for prediction
@app.post("/predict/")
def predict_sentiment(request: ReviewRequest):
    cleaned_review = clean_text(request.review)
    tfidf_review = vectorizer.transform([cleaned_review])
    prediction = model.predict(tfidf_review)[0]
    sentiment = "positive" if prediction == 1 else "negative"
    return {"sentiment": sentiment}
