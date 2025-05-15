import os
import joblib
from pathlib import Path
from train_model import ProductClassifier
from sklearn.model_selection import train_test_split
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_DIR = Path("./models")
REQUIRED_MODELS = {
    "tfidf_vectorizer.pkl": None,
    "linear_svc_model.pkl": None,
    "label_encoder.pkl": None
}

# Load models if they exist, otherwise train new ones
def load_or_train_models(credentials_path="../training/credentials.json"):
    # Check if all model files exist
    all_exist = all((MODEL_DIR / fname).exists() for fname in REQUIRED_MODELS.keys())

    if all_exist:
        logger.info("Loading existing models")
        try:
            for fname in REQUIRED_MODELS:
                REQUIRED_MODELS[fname] = joblib.load(MODEL_DIR / fname)
            return REQUIRED_MODELS
        except Exception as e:
            logger.warning(f"Model loading failed: {str(e)}")

    # Train new models if loading failed or files don't exist
    logger.info("Training new models")
    os.makedirs(MODEL_DIR, exist_ok=True)

    classifier = ProductClassifier()
    df = classifier.load_data(credentials_path)
    X, y = classifier.prepare_data(df)
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train_tfidf, _ = classifier.create_features(X_train, X_train)  # We only need training features
    classifier.train(X_train_tfidf, y_train)
    classifier.save_models(MODEL_DIR)

    # Load the newly trained models
    for fname in REQUIRED_MODELS:
        REQUIRED_MODELS[fname] = joblib.load(MODEL_DIR / fname)

    return REQUIRED_MODELS