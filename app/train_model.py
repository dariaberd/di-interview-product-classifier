import re
import numpy as np
import pandas as pd
import joblib
from google.cloud import bigquery
from sklearn.svm import LinearSVC
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score,
                             f1_score,
                             top_k_accuracy_score,
                             classification_report)
import nltk
from nltk.corpus import stopwords
import logging

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Download stopwords
nltk.download('stopwords', quiet=True)


class ProductClassifier:
    def __init__(self):
        self.tfidf = None
        self.model = None
        self.label_encoder = None

    def load_data(self, credentials_path):
        logger.info("Loading data from BigQuery")
        client = bigquery.Client.from_service_account_json(credentials_path,
                                                           project='charged-dialect-824')
        sql = """
        SELECT *
        FROM RicardoInterview.product_detection_training_data
        """
        df = client.query(sql).to_dataframe()
        return df

    def preprocess_text(self, text):
        text = str(text).lower()
        text = re.sub(r"[^\w\s]", "", text)
        return text

    def prepare_data(self, df, min_samples=99):
        logger.info("Preparing training data")

        # Combine title and subtitle
        df['text'] = (df['title'].fillna('') + ' ' + df['subtitle'].fillna('')).apply(self.preprocess_text)

        # Handle class imbalance
        class_counts = df['productType'].value_counts()
        valid_classes = class_counts[class_counts >= min_samples].index
        df = df[df['productType'].isin(valid_classes)]

        # Encode labels
        self.label_encoder = LabelEncoder()
        y = self.label_encoder.fit_transform(df['productType'])
        logger.info(f"Final number of classes: {len(self.label_encoder.classes_)}")

        return df['text'], y

    def create_features(self, X_train, X_test):

        logger.info("Creating TF-IDF features")

        word_pipe = TfidfVectorizer(
            ngram_range=(1, 3),
            stop_words=stopwords.words("german"),
            min_df=3,
            max_df=0.85,
            max_features=10000
        )

        char_pipe = TfidfVectorizer(
            analyzer='char_wb',
            ngram_range=(3, 5),
            min_df=3
        )

        self.tfidf = FeatureUnion([
            ('word', word_pipe),
            ('char', char_pipe)
        ])

        X_train_tfidf = self.tfidf.fit_transform(X_train)
        X_test_tfidf = self.tfidf.transform(X_test)

        return X_train_tfidf, X_test_tfidf

    def train(self, X_train, y_train):

        logger.info("Training LinearSVC model")

        svm = LinearSVC(
            C=0.2158,
            class_weight='balanced',
            dual=False,
            max_iter=10000,
            random_state=42
        )

        self.model = CalibratedClassifierCV(svm, method='sigmoid')
        self.model.fit(X_train, y_train)
        return self.model

    def evaluate(self, X_test, y_test):

        logger.info("Evaluating model")

        y_pred = self.model.predict(X_test)
        y_proba = self.model.predict_proba(X_test)

        print("\n=== Model Evaluation ===")
        print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
        print(f"Top-3 Accuracy: {top_k_accuracy_score(y_test, y_proba, k=3):.4f}")
        print(f"Weighted F1: {f1_score(y_test, y_pred, average='weighted'):.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred,
                                    target_names=self.label_encoder.classes_))

    def save_models(self, output_dir="models"):

        logger.info(f"Saving models to {output_dir}")

        import os
        os.makedirs(output_dir, exist_ok=True)

        joblib.dump(self.tfidf, f'{output_dir}/tfidf_vectorizer.pkl')
        joblib.dump(self.model, f'{output_dir}/linear_svc_model.pkl')
        joblib.dump(self.label_encoder, f'{output_dir}/label_encoder.pkl')


def main():
    # Initialize classifier
    classifier = ProductClassifier()

    # Load and prepare data
    df = classifier.load_data("./training/credentials.json")
    X, y = classifier.prepare_data(df)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Create features
    X_train_tfidf, X_test_tfidf = classifier.create_features(X_train, X_test)

    # Train model
    classifier.train(X_train_tfidf, y_train)

    # Evaluate
    classifier.evaluate(X_test_tfidf, y_test)

    # Save models
    classifier.save_models()


if __name__ == "__main__":
    main()