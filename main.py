from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
import os
import sys
import gdown

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import rbf_kernel


# =========================
# 🔧 CUSTOM TRANSFORMERS
# =========================

class Cluster_similarity(BaseEstimator, TransformerMixin):
    def __init__(self, n_clusters=10, gamma=1.0, random_state=42):
        self.n_clusters = n_clusters
        self.gamma = gamma
        self.random_state = random_state

    def fit(self, X, y=None, sample_weight=None):
        self.Kmeans_ = KMeans(n_clusters=self.n_clusters, random_state=self.random_state)
        self.Kmeans_.fit(X, sample_weight=sample_weight)
        return self

    def transform(self, X):
        return rbf_kernel(X, self.Kmeans_.cluster_centers_, gamma=self.gamma)

    def get_feature_names_out(self, input_features=None):
        return [f"cluster_{i}_similarity" for i in range(self.n_clusters)]


def column_ratio(X):
    return X[:, [0]] / X[:, [1]]


def ratio_name(transformer, feature_names_in):
    return ["ratio"]


# =========================
# ⚠️ CRITICAL FIX (DO NOT REMOVE)
# =========================

sys.modules["__main__"].column_ratio = column_ratio
sys.modules["__main__"].ratio_name = ratio_name
sys.modules["__main__"].Cluster_similarity = Cluster_similarity


# =========================
# 🚀 FASTAPI SETUP
# =========================

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://frontend-california-housing-predict.vercel.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================
# 📦 MODEL DOWNLOAD + LOAD
# =========================

MODEL_URL = "https://drive.google.com/file/d/1xGpyvUo3WB48qCLi6byVEVLgayi4Y3Z2/view?usp=sharing"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "california_housing_predictor_model.pkl")

model = None


def download_model():
    if not os.path.exists(MODEL_PATH):
        print("📥 Downloading model from Google Drive...")
        gdown.download(MODEL_URL, MODEL_PATH, quiet=False, fuzzy=True)
        print("✅ Model downloaded successfully")
    else:
        print("📦 Model already exists, skipping download")


def load_model():
    global model
    try:
        download_model()
        model = joblib.load(MODEL_PATH)
        print("✅ Model loaded successfully")
    except Exception as e:
        print("❌ Error loading model:", str(e))
        model = None


# Load model on startup
load_model()


# =========================
# 📥 INPUT SCHEMA
# =========================

class HousingData(BaseModel):
    longitude: float
    latitude: float
    housingMedianAge: float
    totalRooms: float
    totalBedrooms: float
    population: float
    households: float
    medianIncome: float
    oceanProximity: str


# =========================
# 🔮 PREDICTION ENDPOINT
# =========================

@app.get("/")
def home():
    return {"message": "API is running 🚀"}


@app.post("/predict")
async def predict(data: HousingData):
    if model is None:
        return {"error": "Model not loaded"}

    df = pd.DataFrame([{
        "longitude": data.longitude,
        "latitude": data.latitude,
        "housing_median_age": data.housingMedianAge,
        "total_rooms": data.totalRooms,
        "total_bedrooms": data.totalBedrooms,
        "population": data.population,
        "households": data.households,
        "median_income": data.medianIncome,
        "ocean_proximity": data.oceanProximity,
    }])

    try:
        prediction = model.predict(df)
        return {
            "predictedValue": float(prediction[0]),
            "confidence": 0.95,
            "timestamp": pd.Timestamp.now().isoformat()
        }
    except Exception as e:
        return {"error": str(e)}
