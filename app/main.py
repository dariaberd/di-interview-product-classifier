from flask import Flask, request, jsonify
import spacy
import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import LabelEncoder

from model_loader import load_or_train_models

app = Flask(__name__)

# Load models and components at startup
nlp = spacy.load("en_core_web_sm")
models = load_or_train_models()
vectorizer = models["tfidf_vectorizer.pkl"]
model = models["linear_svc_model.pkl"]
label_encoder = models["label_encoder.pkl"]

# Denylist for keyword extraction
DENYLIST = {"product", "title", "containing"}

# Keyword extraction
def extract_keywords(text):
    doc = nlp(text)
    return " ".join([
        token.text for token in doc
        if (token.pos_ in ["NOUN", "PROPN"]) and (token.text.lower() not in DENYLIST)
    ])


def clean_text(text):
    return text.lower().strip()


@app.route('/classify', methods=['POST'])
def classify_product():

    try:
        # Get and validate input
        data = request.get_json()
        if not data or 'title' not in data:
            return jsonify({"error": "Missing 'title' in request"}), 400

        # Process text
        raw_text = data['title']
        keywords = extract_keywords(raw_text)
        cleaned_text = clean_text(keywords)

        # Vectorize and predict
        X = vectorizer.transform([cleaned_text])
        proba = model.predict_proba(X)[0]

        # Get top 3 predictions
        top3_idx = np.argsort(proba)[::-1][:3]
        top3_labels = label_encoder.inverse_transform(top3_idx)
        top3_scores = proba[top3_idx]

        # Build response
        response = {
            "title": raw_text,
            "top_3_results": [
                {
                    "product_type": str(label),
                    "score": round(float(score), 4)
                } for label, score in zip(top3_labels, top3_scores)
            ],
            "product_type": str(top3_labels[0])
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)