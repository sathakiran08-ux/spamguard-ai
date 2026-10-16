"""
app.py
-----------------------------------------------------------------------------
SpamGuard AI - Streamlit Dashboard
-----------------------------------------------------------------------------
Intelligent Spam Message Detection Using Machine Learning.

This file ONLY handles the user interface and loading the already-trained
model. The actual model training happens in train_model.py so the app
never retrains on every submission - it just loads the saved
TF-IDF vectorizer + Multinomial Naive Bayes classifier with joblib.

Run with:
    streamlit run app.py
-----------------------------------------------------------------------------
"""

import os
import re
import json
import string
from datetime import datetime

import joblib
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

# -----------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------
MODEL_PATH = os.path.join("models", "spam_classifier.pkl")
VECTORIZER_PATH = os.path.join("models", "tfidf_vectorizer.pkl")
METRICS_PATH = os.path.join("models", "metrics.json")

# -----------------------------------------------------------------------
# Page configuration
# -----------------------------------------------------------------------
st.set_page_config(
    page_title="SpamGuard AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------------------------------------------------
# Custom CSS - dark navy & blue theme
# -----------------------------------------------------------------------
st.markdown("""
<style>
    .stApp {
        background-color: #0b1220;
        color: #e6edf7;
    }
    section[data-testid="stSidebar"] {
        background-color: #0e1a2b;
        border-right: 1px solid #1c2b40;
    }
    h1, h2, h3, h4 {
        color: #eaf2ff !important;
    }
    .sg-title {
        font-size: 2.4rem;
        font-weight: 800;
        background: linear-gradient(90deg, #4f9dff, #7cd4ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0;
    }
    .sg-subtitle {
        color: #9db4d6;
        font-size: 1.05rem;
        margin-top: 0.2rem;
        margin-bottom: 1.5rem;
    }
    .sg-card {
        background-color: #121d30;
        border: 1px solid #1f3350;
        border-radius: 14px;
        padding: 1.3rem 1.5rem;
        margin-bottom: 1rem;
    }
    .sg-spam-box {
        background: linear-gradient(135deg, #3a0f14, #1f0a0d);
        border: 1px solid #8f2d36;
        border-radius: 14px;
        padding: 1.5rem;
        text-align: center;
    }
    .sg-ham-box {
        background: linear-gradient(135deg, #0f3a1f, #0a1f12);
        border: 1px solid #2d8f4e;
        border-radius: 14px;
        padding: 1.5rem;
        text-align: center;
    }
    .sg-spam-text {
        color: #ff6b6b;
        font-size: 1.6rem;
        font-weight: 800;
    }
    .sg-ham-text {
        color: #4ee28c;
        font-size: 1.6rem;
        font-weight: 800;
    }
    .sg-metric-box {
        background-color: #121d30;
        border: 1px solid #1f3350;
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
    }
    .stButton>button {
        border-radius: 10px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------
# Text cleaning - MUST match the cleaning done in train_model.py so the
# model sees text in the same format it was trained on.
# -----------------------------------------------------------------------
def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"http\S+|www\.\S+", " urllink ", text)
    text = re.sub(r"\d+", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text


# -----------------------------------------------------------------------
# Load the trained model, vectorizer, and metrics (cached so this only
# runs once per app session, not on every click).
# -----------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(VECTORIZER_PATH):
        return None, None, None

    # joblib.load is used here (not raw pickle.load on arbitrary input) and
    # it only ever loads files that WE generated locally via train_model.py.
    # Never point this at a model file you did not train yourself.
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)

    metrics = None
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH, "r") as f:
            metrics = json.load(f)

    return model, vectorizer, metrics


model, vectorizer, metrics = load_artifacts()
MODEL_READY = model is not None and vectorizer is not None


def predict_message(message: str):
    """
    Returns (label, spam_probability) for a single raw message string.
    label is 'spam' or 'ham'. spam_probability is a float 0-1, or None if
    the underlying model doesn't support predict_proba.
    """
    cleaned = clean_text(message)
    features = vectorizer.transform([cleaned])
    pred = model.predict(features)[0]
    label = "spam" if pred == 1 else "ham"

    proba = None
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(features)[0]
        # class order follows model.classes_ ([0, 1] = [ham, spam])
        spam_index = list(model.classes_).index(1)
        proba = float(probs[spam_index])

    return label, proba


# -----------------------------------------------------------------------
# Session state setup
# -----------------------------------------------------------------------
if "history" not in st.session_state:
    st.session_state.history = []  # list of dicts: time, message, label, proba

if "message_input" not in st.session_state:
    st.session_state.message_input = ""


def clear_message():
    st.session_state.message_input = ""


def set_example(text):
    st.session_state.message_input = text


EXAMPLES = {
    "📩 Example - Likely Spam": "WINNER!! You have been selected to receive a £1000 cash prize! "
                                "Call 08712460324 now to claim your reward before it expires.",
    "💬 Example - Likely Ham": "Hey, are we still on for lunch tomorrow? Let me know what time works for you.",
    "🎁 Example - Fake Prize": "Congratulations! You've been selected for a free iPhone. Claim now at http://claimnow.example.com",
    "👋 Example - Casual chat": "Can you pick up some milk on your way home? Thanks!",
}

# -----------------------------------------------------------------------
# Sidebar navigation
# -----------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🛡️ SpamGuard AI")
    page = st.radio(
        "Navigate",
        ["🏠 Home", "🔍 Message Detector", "📊 Model Information", "ℹ️ About Project"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    if MODEL_READY:
        st.success("Model loaded ✅")
    else:
        st.error("Model not found ❌")
    st.caption("Built with Python, scikit-learn & Streamlit")

# -----------------------------------------------------------------------
# Shared header
# -----------------------------------------------------------------------
st.markdown('<p class="sg-title">SpamGuard AI</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="sg-subtitle">Intelligent Spam Message Detection Using Machine Learning</p>',
    unsafe_allow_html=True,
)

# If model files are missing, show a clear error everywhere and stop early
# for pages that need the model.
if not MODEL_READY:
    st.error(
        "⚠️ Model files were not found in the `models/` folder.\n\n"
        "Please train the model first by running:\n\n"
        "```\npython train_model.py\n```\n\n"
        "This will create `models/spam_classifier.pkl` and "
        "`models/tfidf_vectorizer.pkl`."
    )

# =========================================================================
# PAGE: Home
# =========================================================================
if page == "🏠 Home":
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(
            '<div class="sg-card"><h3>🧠 Machine Learning</h3>'
            '<p>Trained with a real Multinomial Naive Bayes classifier on '
            'TF-IDF text features - not a hardcoded demo.</p></div>',
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            '<div class="sg-card"><h3>⚡ Instant Results</h3>'
            '<p>Paste any SMS, email, or text message and get a prediction '
            'in milliseconds, with an estimated spam probability.</p></div>',
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            '<div class="sg-card"><h3>🔒 Privacy First</h3>'
            '<p>Nothing is sent to external servers or saved permanently. '
            'History lives only in this browser session.</p></div>',
            unsafe_allow_html=True,
        )

    st.markdown("### How it works")
    st.markdown(
        """
        1. **You paste a message** into the Message Detector page.
        2. The text is **cleaned and normalized** the same way the training
           data was cleaned.
        3. A **TF-IDF vectorizer** converts the words into numerical
           features based on word importance.
        4. A **Multinomial Naive Bayes** classifier - trained on thousands
           of labeled spam/ham examples - predicts the most likely label.
        5. You see the result along with the model's **confidence
           (spam probability)**.
        """
    )

    if MODEL_READY and metrics:
        st.markdown("### Quick model snapshot")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Accuracy", f"{metrics['accuracy']*100:.1f}%")
        c2.metric("Precision", f"{metrics['precision']*100:.1f}%")
        c3.metric("Recall", f"{metrics['recall']*100:.1f}%")
        c4.metric("F1-score", f"{metrics['f1_score']*100:.1f}%")

    st.info(
        "👉 Head to **Message Detector** in the sidebar to try it out, or "
        "upload a CSV for batch detection."
    )

# =========================================================================
# PAGE: Message Detector
# =========================================================================
elif page == "🔍 Message Detector":
    st.markdown("## 🔍 Message Detector")

    # --- Example message buttons ---
    st.markdown("**Try an example message:**")
    ex_cols = st.columns(len(EXAMPLES))
    for i, (label, text) in enumerate(EXAMPLES.items()):
        with ex_cols[i]:
            if st.button(label, use_container_width=True, key=f"ex_{i}"):
                set_example(text)

    # --- Text input area ---
    message = st.text_area(
        "Enter or paste a message to analyze:",
        key="message_input",
        height=160,
        placeholder="Type or paste an SMS, email, or text message here...",
    )

    char_count = len(message)
    st.caption(f"Character count: {char_count}")

    col_a, col_b, _ = st.columns([1, 1, 4])
    with col_a:
        detect_clicked = st.button("🔎 Detect Spam", type="primary", use_container_width=True, disabled=not MODEL_READY)
    with col_b:
        st.button("🧹 Clear", on_click=clear_message, use_container_width=True)

    if detect_clicked:
        if not message or not message.strip():
            st.warning("⚠️ Please enter a message before clicking Detect Spam.")
        elif not MODEL_READY:
            st.error("Model is not loaded. Please run `python train_model.py` first.")
        else:
            label, proba = predict_message(message)

            # Save to session history (not persisted to disk/db)
            st.session_state.history.insert(0, {
                "time": datetime.now().strftime("%H:%M:%S"),
                "message": message if len(message) <= 80 else message[:77] + "...",
                "label": label,
                "proba": proba,
            })
            st.session_state.history = st.session_state.history[:15]  # keep last 15

            st.markdown("### Result")
            if label == "spam":
                st.markdown(
                    f"""
                    <div class="sg-spam-box">
                        <div class="sg-spam-text">⚠️ SPAM MESSAGE DETECTED</div>
                        <p style="color:#f3b3b3; margin-top:0.5rem;">
                            This message shows patterns commonly found in spam,
                            such as urgent language, prize claims, suspicious
                            links, or promotional offers.
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"""
                    <div class="sg-ham-box">
                        <div class="sg-ham-text">✅ THIS MESSAGE APPEARS TO BE NOT SPAM</div>
                        <p style="color:#a9e8c4; margin-top:0.5rem;">
                            This message looks like typical personal or
                            conversational text, without strong spam
                            indicators.
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            if proba is not None:
                st.markdown("#### Estimated spam probability")
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=proba * 100,
                    number={"suffix": "%"},
                    gauge={
                        "axis": {"range": [0, 100]},
                        "bar": {"color": "#ff6b6b" if label == "spam" else "#4ee28c"},
                        "steps": [
                            {"range": [0, 40], "color": "#0f3a1f"},
                            {"range": [40, 70], "color": "#3a3410"},
                            {"range": [70, 100], "color": "#3a0f14"},
                        ],
                    },
                ))
                fig.update_layout(
                    height=260,
                    margin=dict(l=20, r=20, t=30, b=10),
                    paper_bgcolor="rgba(0,0,0,0)",
                    font={"color": "#e6edf7"},
                )
                st.plotly_chart(fig, use_container_width=True)

            st.caption(
                "ℹ️ This is a machine-learning estimate based on patterns in "
                "training data. It may be incorrect, and a 'not spam' result "
                "is never a guarantee of safety."
            )

    # --- Prediction history ---
    if st.session_state.history:
        st.markdown("### 🕘 Recent prediction history (this session only)")
        hist_df = pd.DataFrame(st.session_state.history)
        hist_df["label"] = hist_df["label"].apply(
            lambda x: "⚠️ SPAM" if x == "spam" else "✅ NOT SPAM"
        )
        hist_df["proba"] = hist_df["proba"].apply(
            lambda p: f"{p*100:.1f}%" if p is not None else "-"
        )
        hist_df.columns = ["Time", "Message (preview)", "Prediction", "Spam probability"]
        st.dataframe(hist_df, use_container_width=True, hide_index=True)
        if st.button("🗑️ Clear history"):
            st.session_state.history = []
            st.rerun()

    st.markdown("---")
    st.markdown("### 📂 Batch Detection (CSV upload)")
    st.caption(
        "Upload a CSV file containing a column of messages. The column can "
        "be named 'message', 'text', or similar - we'll try to detect it "
        "automatically."
    )
    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

    if uploaded_file is not None and MODEL_READY:
        try:
            batch_df = pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"❌ Could not read the uploaded file as a CSV: {e}")
            batch_df = None

        if batch_df is not None:
            # Try to auto-detect the message column
            candidate_cols = [c for c in batch_df.columns
                               if c.strip().lower() in ("message", "text", "sms", "content", "body")]
            if not candidate_cols:
                st.error(
                    "❌ No message column found. Please make sure your CSV "
                    "has a column named 'message' (or 'text')."
                )
            else:
                msg_col = candidate_cols[0]
                batch_df = batch_df.dropna(subset=[msg_col])
                if batch_df.empty:
                    st.warning("⚠️ The uploaded CSV has no usable rows after removing empty messages.")
                else:
                    with st.spinner("Analyzing messages..."):
                        labels, probas = [], []
                        for msg in batch_df[msg_col].astype(str):
                            lbl, pr = predict_message(msg)
                            labels.append(lbl)
                            probas.append(round(pr, 4) if pr is not None else None)

                        batch_df["prediction"] = labels
                        batch_df["spam_probability"] = probas

                    st.success(f"✅ Analyzed {len(batch_df)} messages.")
                    st.dataframe(batch_df, use_container_width=True, hide_index=True)

                    spam_count = (batch_df["prediction"] == "spam").sum()
                    ham_count = (batch_df["prediction"] == "ham").sum()
                    fig = px.pie(
                        names=["Spam", "Not Spam"],
                        values=[spam_count, ham_count],
                        color=["Spam", "Not Spam"],
                        color_discrete_map={"Spam": "#ff6b6b", "Not Spam": "#4ee28c"},
                        hole=0.45,
                    )
                    fig.update_layout(
                        paper_bgcolor="rgba(0,0,0,0)",
                        font={"color": "#e6edf7"},
                        height=320,
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    csv_bytes = batch_df.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        "⬇️ Download predictions as CSV",
                        data=csv_bytes,
                        file_name="spamguard_predictions.csv",
                        mime="text/csv",
                    )

# =========================================================================
# PAGE: Model Information
# =========================================================================
elif page == "📊 Model Information":
    st.markdown("## 📊 Model Information")

    st.markdown("### How the prediction works (beginner-friendly)")
    st.markdown(
        """
        **1. TF-IDF (Term Frequency - Inverse Document Frequency)**

        TF-IDF turns each message into a list of numbers that represent how
        important each word is.
        - A word that appears **often in one message but rarely in others**
          (like "prize" or "winner" in a spam text) gets a **high score**.
        - A word that appears in almost every message (like "the" or "is")
          gets a **low score**, because it doesn't help tell messages apart.

        This lets the computer "read" text as numbers while still capturing
        which words really matter.

        **2. Multinomial Naive Bayes**

        This is a simple but surprisingly powerful classification algorithm.
        It looks at the training examples and learns things like:
        *"messages labeled spam tend to contain words like 'free', 'win',
        'urgent', 'claim' far more often than messages labeled ham."*

        When a new message comes in, it calculates the **probability** that
        the message is spam vs. ham, based on which words appear, and
        picks whichever is more likely. It's called "naive" because it
        assumes each word contributes independently to the result - a
        simplification that works remarkably well for text classification.

        **3. Putting it together**

        ```
        Raw message  →  Clean text  →  TF-IDF numbers  →  Naive Bayes  →  Spam / Ham
        ```
        """
    )

    if MODEL_READY and metrics:
        st.markdown("### 📈 Evaluation Dashboard (on held-out test data)")

        c1, c2, c3, c4 = st.columns(4)
        c1.markdown(f'<div class="sg-metric-box"><h2>{metrics["accuracy"]*100:.1f}%</h2>Accuracy</div>', unsafe_allow_html=True)
        c2.markdown(f'<div class="sg-metric-box"><h2>{metrics["precision"]*100:.1f}%</h2>Precision</div>', unsafe_allow_html=True)
        c3.markdown(f'<div class="sg-metric-box"><h2>{metrics["recall"]*100:.1f}%</h2>Recall</div>', unsafe_allow_html=True)
        c4.markdown(f'<div class="sg-metric-box"><h2>{metrics["f1_score"]*100:.1f}%</h2>F1-score</div>', unsafe_allow_html=True)

        st.markdown("")
        st.markdown(
            f"""
            - **Training examples:** {metrics['train_size']}
            - **Test examples (held out, never seen during training):** {metrics['test_size']}
            - **Vocabulary size (unique TF-IDF terms):** {metrics['vocab_size']}
            - **Random seed used for reproducibility:** {metrics['random_seed']}
            """
        )

        st.markdown("### Confusion Matrix")
        cm = metrics["confusion_matrix"]  # [[TN, FP], [FN, TP]]
        cm_labels = metrics["labels"]  # ["ham", "spam"]
        fig = px.imshow(
            cm,
            text_auto=True,
            x=[f"Predicted {l}" for l in cm_labels],
            y=[f"Actual {l}" for l in cm_labels],
            color_continuous_scale="Blues",
        )
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font={"color": "#e6edf7"},
            height=400,
        )
        st.plotly_chart(fig, use_container_width=True)

        with st.expander("What do precision, recall, and F1-score mean?"):
            st.markdown(
                """
                - **Precision**: Of all the messages the model called *spam*,
                  what fraction actually were spam? High precision means
                  few legitimate messages get wrongly flagged.
                - **Recall**: Of all the *actual* spam messages, what
                  fraction did the model correctly catch? High recall means
                  few spam messages slip through.
                - **F1-score**: The balance between precision and recall,
                  combined into a single number.
                - **Confusion matrix**: A table showing correct vs.
                  incorrect predictions, broken down by actual vs.
                  predicted label.
                """
            )
    else:
        st.warning("Train the model first (`python train_model.py`) to see evaluation metrics here.")

# =========================================================================
# PAGE: About Project
# =========================================================================
elif page == "ℹ️ About Project":
    st.markdown("## ℹ️ About SpamGuard AI")
    st.markdown(
        """
        **SpamGuard AI** is a beginner-friendly, fully functional spam
        message detector built with Python, scikit-learn, and Streamlit.

        ### Tech stack
        - **Language:** Python
        - **Web framework:** Streamlit
        - **Machine learning:** scikit-learn (TF-IDF + Multinomial Naive Bayes)
        - **Data processing:** pandas, NumPy
        - **Visualization:** Plotly

        ### Design principles
        - The model is **trained separately** (`train_model.py`) from the
          web app (`app.py`), so the app never retrains on every click -
          it simply loads the saved model with `joblib`.
        - The **TF-IDF vectorizer is fit only on training data**, never on
          test data, to avoid data leakage and give an honest accuracy
          estimate.
        - **Nothing you type is sent anywhere** or stored permanently.
          Prediction history lives only in your current browser session
          and disappears when you close the tab.
        - Predictions are **machine-learning estimates**. The model can be
          wrong, especially on messages very different from its training
          data - always use your own judgment for anything important.

        ### Project structure
        ```
        spamguard-ai/
        ├── app.py                   # Streamlit UI (this app)
        ├── train_model.py           # Model training script
        ├── generate_dataset.py      # Creates a practice dataset
        ├── requirements.txt
        ├── README.md
        ├── data/
        │   └── spam.csv
        └── models/
            ├── spam_classifier.pkl
            ├── tfidf_vectorizer.pkl
            └── metrics.json
        ```
        """
    )
