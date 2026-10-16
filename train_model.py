"""
train_model.py
-----------------------------------------------------------------------------
SpamGuard AI - Model Training Script
-----------------------------------------------------------------------------
This script trains a Multinomial Naive Bayes spam classifier on the SMS
Spam Collection dataset (or an equivalent CSV with 'label' and 'message'
columns).

What this script does, step by step:
1. Loads the labeled dataset from data/spam.csv
2. Cleans the text (lowercasing, removing punctuation/numbers/extra spaces)
3. Removes duplicate rows and rows with missing values
4. Splits the data into a training set and a held-out test set
5. Fits a TF-IDF vectorizer ONLY on the training data (avoids data leakage)
6. Trains a Multinomial Naive Bayes classifier on the TF-IDF features
7. Evaluates the model on the test set (accuracy, precision, recall, F1,
   confusion matrix)
8. Saves the trained model, vectorizer, and evaluation metrics to the
   models/ folder using joblib, so the Streamlit app can load them without
   retraining every time.

Run this script any time you update the dataset:
    python train_model.py
-----------------------------------------------------------------------------
"""

import os
import re
import json
import string
import sys

import pandas as pd
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

# -----------------------------------------------------------------------
# Configuration / file paths
# -----------------------------------------------------------------------
DATA_PATH = os.path.join("data", "spam.csv")
MODELS_DIR = "models"
MODEL_PATH = os.path.join(MODELS_DIR, "spam_classifier.pkl")
VECTORIZER_PATH = os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl")
METRICS_PATH = os.path.join(MODELS_DIR, "metrics.json")

RANDOM_SEED = 42  # Keeps results reproducible every time we train


def clean_text(text: str) -> str:
    """
    Normalizes a raw text message so the model focuses on meaningful words:
    - Lowercases everything
    - Removes punctuation
    - Removes digits (spam often varies numbers, so we strip them)
    - Collapses multiple spaces into one
    """
    text = str(text).lower()
    text = re.sub(r"http\S+|www\.\S+", " urllink ", text)   # normalize URLs
    text = re.sub(r"\d+", " ", text)                         # remove numbers
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_dataset(path: str) -> pd.DataFrame:
    """
    Loads the dataset and validates that it has the expected columns.
    Supports the common SMS Spam Collection formats:
      - A simple CSV with columns: label, message
      - The raw UCI file with columns: v1 (label), v2 (message)
    """
    if not os.path.exists(path):
        print(f"\nERROR: Dataset not found at '{path}'.\n")
        print("How to fix this:")
        print("  1. Run 'python generate_dataset.py' to create a practice")
        print("     dataset automatically, OR")
        print("  2. Download the real SMS Spam Collection dataset from:")
        print("     https://archive.ics.uci.edu/dataset/228/sms+spam+collection")
        print("     (or search 'SMS Spam Collection Dataset' on Kaggle)")
        print("  3. Save it as 'data/spam.csv' with two columns named")
        print("     'label' (spam/ham) and 'message' (the text content).")
        sys.exit(1)

    df = pd.read_csv(path, encoding="latin-1")

    # Handle the raw UCI column names (v1/v2) if present
    if "v1" in df.columns and "v2" in df.columns:
        df = df.rename(columns={"v1": "label", "v2": "message"})

    df = df[[c for c in df.columns if c.lower() in ("label", "message")]]
    df.columns = [c.lower() for c in df.columns]

    if "label" not in df.columns or "message" not in df.columns:
        print("\nERROR: CSV must contain 'label' and 'message' columns.")
        print(f"Found columns: {list(df.columns)}")
        sys.exit(1)

    return df


def main():
    print("=" * 70)
    print("SpamGuard AI - Model Training")
    print("=" * 70)

    # 1. Load dataset
    print(f"\n[1/7] Loading dataset from '{DATA_PATH}' ...")
    df = load_dataset(DATA_PATH)
    print(f"      Loaded {len(df)} rows.")

    # 2. Clean labels and handle missing values / duplicates
    print("\n[2/7] Cleaning data (missing values, duplicates, label format) ...")
    df["label"] = df["label"].astype(str).str.strip().str.lower()
    df = df[df["label"].isin(["spam", "ham"])]
    df = df.dropna(subset=["message", "label"])
    before = len(df)
    df = df.drop_duplicates(subset=["message", "label"])
    print(f"      Removed {before - len(df)} duplicate/invalid rows. "
          f"{len(df)} rows remain.")

    if len(df) < 20:
        print("ERROR: Not enough data to train a reliable model.")
        sys.exit(1)

    # 3. Clean message text
    print("\n[3/7] Normalizing message text (lowercasing, removing noise) ...")
    df["clean_message"] = df["message"].apply(clean_text)
    df = df[df["clean_message"].str.len() > 0]

    # Encode labels: spam = 1, ham = 0
    df["target"] = (df["label"] == "spam").astype(int)

    print(f"      Class balance -> spam: {(df['target']==1).sum()}, "
          f"ham: {(df['target']==0).sum()}")

    # 4. Train/test split (reproducible, stratified so both sets keep the
    #    same spam/ham ratio)
    print("\n[4/7] Splitting into train/test sets (80/20, stratified) ...")
    X_train_text, X_test_text, y_train, y_test = train_test_split(
        df["clean_message"],
        df["target"],
        test_size=0.2,
        random_state=RANDOM_SEED,
        stratify=df["target"],
    )
    print(f"      Train: {len(X_train_text)} messages | "
          f"Test: {len(X_test_text)} messages")

    # 5. TF-IDF vectorization - IMPORTANT: fit only on training data to
    #    avoid data leakage from the test set.
    print("\n[5/7] Fitting TF-IDF vectorizer on TRAINING data only ...")
    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        max_df=0.95,
        min_df=1,
        sublinear_tf=True,
    )
    X_train = vectorizer.fit_transform(X_train_text)
    X_test = vectorizer.transform(X_test_text)  # only transform, never fit
    print(f"      Vocabulary size: {len(vectorizer.vocabulary_)} terms")

    # 6. Train Multinomial Naive Bayes classifier
    print("\n[6/7] Training Multinomial Naive Bayes classifier ...")
    model = MultinomialNB(alpha=0.3)
    model.fit(X_train, y_train)

    # 7. Evaluate on the held-out test set
    print("\n[7/7] Evaluating on the held-out test set ...")
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred).tolist()  # [[TN, FP], [FN, TP]]

    print("\n" + "-" * 70)
    print(f"Accuracy  : {accuracy:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1-score  : {f1:.4f}")
    print("\nConfusion Matrix [[TN, FP], [FN, TP]]:")
    print(cm)
    print("\nClassification report:")
    print(classification_report(y_test, y_pred, target_names=["ham", "spam"]))
    print("-" * 70)

    # Save everything the app needs: model, vectorizer, and metrics
    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)

    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "confusion_matrix": cm,  # rows/cols order: [ham, spam]
        "labels": ["ham", "spam"],
        "train_size": len(X_train_text),
        "test_size": len(X_test_text),
        "vocab_size": len(vectorizer.vocabulary_),
        "random_seed": RANDOM_SEED,
    }
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nSaved model      -> {MODEL_PATH}")
    print(f"Saved vectorizer -> {VECTORIZER_PATH}")
    print(f"Saved metrics    -> {METRICS_PATH}")
    print("\nDone! You can now run: streamlit run app.py")


if __name__ == "__main__":
    main()
