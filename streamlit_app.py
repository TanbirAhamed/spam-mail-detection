import streamlit as st
import pickle
import numpy as np
import re
import nltk
import json
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.exceptions import NotFittedError

st.set_page_config(page_title="Email Spam Detector", page_icon="📧", layout="wide")

@st.cache_resource
def download_nltk_data():
    try:
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk.download('punkt_tab', quiet=True)
    except:
        pass

download_nltk_data()

def pretty_model_name_from_filename(fname: str) -> str:
    """Return a consistent, human-friendly model key from filename."""
    fname = fname.lower()
    if 'naive' in fname or 'bayes' in fname:
        return 'Naive Bayes'
    if 'svm' in fname:
        return 'SVM'
    if 'random' in fname or 'forest' in fname:
        return 'Random Forest'
    if 'gradient' in fname or 'boost' in fname:
        return 'Gradient Boosting'
    return fname.split('.')[0].replace('_', ' ').title()

@st.cache_resource
def load_models():
    models = {}
    tokenizer = None
    tfidf = None
    scaler = None
    encoder = None

    config = {'max_len': 100, 'best_model': 'DNN', 'best_f1_score': 0.0}

    try:
        try:
            with open('models/tfidf_vectorizer.pkl', 'rb') as f:
                tfidf = pickle.load(f)
        except Exception as e:
            st.warning("models/tfidf_vectorizer.pkl not found or failed to load. Some models will not work.")
            tfidf = None

        try:
            with open('models/scaler.pkl', 'rb') as f:
                scaler = pickle.load(f)
        except Exception as e:
            st.warning("models/scaler.pkl not found or failed to load. Numeric features won't be available for some models.")
            scaler = None

        try:
            with open('models/label_encoder.pkl', 'rb') as f:
                encoder = pickle.load(f)
        except:
            encoder = None

        candidate_ml_files = [
            'models/model_svm.pkl',
            'models/model_naive_bayes.pkl',
            'models/model_random_forest.pkl',
            'models/model_gradient_boosting.pkl'
        ]

        for fname in candidate_ml_files:
            try:
                with open(fname, 'rb') as f:
                    model_obj = pickle.load(f)
                    key = pretty_model_name_from_filename(fname)
                    models[key] = model_obj
            except FileNotFoundError:
                continue
            except Exception as e:
                st.warning(f"Failed to load {fname}: {e}")
                continue
        try:
            with open('models/tokenizer.pkl', 'rb') as f:
                tokenizer = pickle.load(f)
        except:
            tokenizer = None

        try:
            models['DNN'] = load_model('models/model_dnn.h5', compile=False)
        except:
            
            pass

        try:
            models['LSTM'] = load_model('models/model_lstm.h5', compile=False)
        except:
            pass

        try:
            with open('models/model_config', 'r') as f:
                cfg = json.load(f)
                config.update(cfg)
        except:
            pass

        return models, tfidf, scaler, encoder, tokenizer, config

    except Exception as e:
        st.error(f"Critical error loading models: {e}")
        return {}, None, None, None, None, config

def preprocess_text(text: str) -> str:
    text = str(text).lower()
    
    try:
        tokens = word_tokenize(text)
    except Exception:

        tokens = text.split()

    tokens_cleaned = []
    for word in tokens:
        if any(char.isdigit() for char in word) or any(char in ['$', '£', '€', '¥'] for char in word):
            tokens_cleaned.append(word)
        else:
            cleaned = re.sub(r'[^\w\s]', '', word)
            if cleaned:
                tokens_cleaned.append(cleaned)

    try:
        stop_words = set(stopwords.words('english'))
    except:
        stop_words = set()
    spam_keywords = {'free', 'win', 'won', 'urgent', 'click', 'claim', 'prize',
                     'cash', 'text', 'call', 'now', 'limited', 'offer', 'congratulations'}
    stop_words -= spam_keywords

    tokens = [w for w in tokens_cleaned if (w not in stop_words) or (w in spam_keywords)]

    ps = PorterStemmer()
    tokens = [ps.stem(w) if len(w) > 3 else w for w in tokens]

    return ' '.join(tokens)

def predict_email(email_text: str, model_name: str, models: dict, tfidf, scaler, tokenizer, config):
    if model_name not in models:
        return "N/A", 0.0

    processed_text = preprocess_text(email_text)

    if model_name in ['DNN', 'LSTM'] and tokenizer is not None:
        try:
            sequence = tokenizer.texts_to_sequences([processed_text])
            padded = pad_sequences(sequence, maxlen=int(config.get('max_len', 100)), padding='post', truncating='post')
            model = models[model_name]
            pred = model.predict(padded, verbose=0).flatten()
            if pred.size == 0:
                return "N/A", 0.0
            pred_prob = float(pred[0])
            prediction = 1 if pred_prob > 0.5 else 0
            confidence = pred_prob if prediction == 1 else 1 - pred_prob
        except Exception as e:
            st.error(f"DL prediction error for {model_name}: {e}")
            return "N/A", 0.0

    else:
        if tfidf is None:
            st.error("TF-IDF vectorizer not loaded. Traditional ML models cannot run.")
            return "N/A", 0.0

        try:
            tfidf_features = tfidf.transform([processed_text]).toarray()
        except NotFittedError:
            st.error("TF-IDF vectorizer is not fitted.")
            return "N/A", 0.0
        except Exception as e:
            st.error(f"TF-IDF transform error: {e}")
            return "N/A", 0.0
.
        name_low = model_name.lower()
        is_naive = ('naive' in name_low) or ('bayes' in name_low)

        if is_naive:
            features = tfidf_features
        else:
            
            if scaler is None:
                st.warning("Scaler not loaded — cannot compute numeric features; trying TF-IDF only.")
                features = tfidf_features
            else:
                length = len(email_text)
                try:
                    num_words = len(word_tokenize(email_text))
                    num_sentences = len(sent_tokenize(email_text))
                except:

                    num_words = len(email_text.split())
                    num_sentences = email_text.count('.') + email_text.count('!') + email_text.count('?')
                
                try:
                    additional_features = scaler.transform([[length, num_words, num_sentences]])
                    features = np.hstack([tfidf_features, additional_features])
                except Exception as e:
                    st.warning(f"Failed to create additional numeric features: {e}. Using TF-IDF only.")
                    features = tfidf_features

        model = models[model_name]
        try:
            prediction = model.predict(features)[0]
        except Exception as e:
            st.error(f"Model prediction error for {model_name}: {e}")
            return "N/A", 0.0

        if hasattr(model, 'predict_proba'):
            try:
                proba = model.predict_proba(features)[0]
                if isinstance(prediction, (np.integer, int)):
                    idx = int(prediction)
                    confidence = float(proba[idx])
                else:
                    confidence = float(np.max(proba))
            except Exception:
                confidence = 1.0
        else:
            confidence = 1.0

    return 'Spam' if int(prediction) == 1 else 'Ham', float(confidence)

def main():

    st.title("📧 Email Spam Detection System")
    st.markdown("### AI-Powered Email Classification")
    st.markdown("---")

    models, tfidf, scaler, encoder, tokenizer, config = load_models()

    if not models:
        st.error("No models loaded successfully. Check your model files (4 .pkl and 2 .h5).")
        return

    with st.sidebar:
        st.header("⚙️ Model Settings")
        available_models = list(models.keys())
        preferred_order = ['Naive Bayes', 'SVM', 'Random Forest', 'Gradient Boosting', 'DNN', 'LSTM']
        ordered = [m for m in preferred_order if m in available_models] + \
                  [m for m in available_models if m not in preferred_order]
        available_models = ordered

        default_model = config.get('best_model', available_models[0]) if available_models else None
        if default_model not in available_models:
            default_model = available_models[0] if available_models else None

        model_name = st.selectbox("🤖 Select Model", available_models, index=available_models.index(default_model) if default_model else 0)
        st.markdown("---")
        st.subheader("📊 Model Info")
        st.write(f"Models loaded: **{len(available_models)}**")
        st.metric("Best Model (config)", config.get('best_model', 'N/A'))
        st.metric("Best F1-Score", f"{config.get('best_f1_score',0):.4f}")

    if 'email_text' not in st.session_state:
        st.session_state.email_text = ""

    col1, col2 = st.columns([3,2])
    with col1:
        st.subheader("📝 Enter Email Content")
        
        st.markdown("**Quick Examples:**")
        ex1, ex2, ex3 = st.columns(3)
        with ex1:
            if st.button("🚨 Spam Example", use_container_width=True):
                st.session_state.email_text = "Congratulations! You've won $1,000,000! Click here to claim your prize now! Act fast, this offer expires in 24 hours. Reply with your bank details to receive your winnings. Free money guaranteed!"
        with ex2:
            if st.button("✅ Ham Example", use_container_width=True):
                st.session_state.email_text = "Hi team, just confirming our meeting tomorrow at 3pm in the conference room. Please review the attached agenda beforehand. Let me know if you have any questions. Thanks!"
        with ex3:
            if st.button("⚠️ Phishing Example", use_container_width=True):
                st.session_state.email_text = "URGENT: Your account will be closed! Verify your information immediately by clicking this link. Failure to respond within 24 hours will result in permanent suspension. Call now!"
        
        st.markdown("---")
        
        email_text = st.text_area(
            "Email Message", 
            value=st.session_state.email_text,
            height=250, 
            placeholder="Paste email here or use examples above...",
            key="email_input"
        )
        
        st.session_state.email_text = email_text

        predict_button = st.button("🔍 Analyze Email", type="primary", use_container_width=True)

    with col2:
        st.subheader("📈 Analysis Results")
        result_container = st.container()

    if predict_button:
        if not email_text or not email_text.strip():
            st.warning("⚠️ Please enter an email message.")
        else:
            with st.spinner("🔄 Analyzing email..."):
                try:
                    result, confidence = predict_email(email_text, model_name, models, tfidf, scaler, tokenizer, config)
                    if result == "N/A":
                        st.error(f"❌ Model {model_name} failed to produce a result.")
                    else:
                        with result_container:
                            if result == 'Spam':
                                st.markdown(
                                    "<div style='background-color:#ffebee;padding:20px;border-left:5px solid #f44336;border-radius:10px'>"
                                    "<h2>🚨 SPAM DETECTED</h2></div>", unsafe_allow_html=True)
                            else:
                                st.markdown(
                                    "<div style='background-color:#e8f5e9;padding:20px;border-left:5px solid #4caf50;border-radius:10px'>"
                                    "<h2>✅ LEGITIMATE EMAIL</h2></div>", unsafe_allow_html=True)

                            try:
                                pct = float(np.clip(confidence, 0.0, 1.0))
                            except:
                                pct = 0.0
                            st.progress(pct)
                            st.metric("Confidence", f"{confidence:.2%}")
                            st.metric("Prediction", result)

                except Exception as e:
                    st.error(f"❌ Prediction error: {str(e)}")

if __name__ == "__main__":
    main()
