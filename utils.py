import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score, f1_score
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.tag import pos_tag
from nltk.stem import WordNetLemmatizer
import requests
from io import StringIO
import os

# Download datasets with error handling
def download_csv(url):
    """Downloads a CSV from a URL with timeout and error handling."""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return pd.read_csv(StringIO(response.text))
    except requests.RequestException as e:
        print(f"Error downloading {url}: {e}")
        return pd.DataFrame()  # Return empty DataFrame on failure

# Text preprocessing
def preprocess_text(text):
    """Tokenizes, removes stopwords, and lemmatizes text."""
    if not isinstance(text, str):
        return [], ''
    tokens = word_tokenize(text.lower())
    stop_words = set(stopwords.words('english'))
    tokens = [WordNetLemmatizer().lemmatize(token) for token in tokens if token.isalnum() and token not in stop_words]
    return tokens, ' '.join(tokens).strip()

# X-ANEW features
def get_xanew_features(tokens, xanew_df, is_lyric=False):
    """Calculates arousal and valence from X-ANEW lexicon with weighting for lyrics."""
    if xanew_df.empty:
        return 0.5, 0.5
    arousal_scores = []
    valence_scores = []
    weights = []
    for token in set(tokens):  # Use set to avoid redundancy
        if token in xanew_df['word'].values:
            row = xanew_df[xanew_df['word'] == token]
            count = tokens.count(token)
            arousal_scores.append(row['arousal'].values[0] * count)
            valence_scores.append(row['valence'].values[0] * count)
            weight = 2.0 if is_lyric and count > 1 else 1.0
            weights.append(weight * count)
    total_weight = sum(weights) if weights else 1.0
    arousal = sum(arousal_scores) / total_weight if arousal_scores else 0.5
    valence = sum(valence_scores) / total_weight if valence_scores else 0.5
    return arousal, valence

# POS tagging with single adjustment per sentence and clamping
def apply_pos_context(tokens, arousal, valence):
    """Adjusts scores based on POS tags and clamps to [0,1]."""
    tagged = pos_tag(tokens)
    adj_count = sum(1 for _, tag in tagged if tag.startswith('JJ'))
    adv_count = sum(1 for _, tag in tagged if tag in ['RB', 'RBR', 'RBS'])
    negation_words = {'not', 'never', 'dont', "don't"}
    negation_count = sum(1 for word, _ in tagged if word in negation_words)
    verb_neg_count = sum(1 for word, tag in tagged if tag.startswith('VB') and word in ['kill', 'destroy'])

    # Apply adjustments once
    arousal *= (1.2 ** (adj_count / max(len(tokens), 1))) * (1.1 ** (adv_count / max(len(tokens), 1)))
    valence *= (1.2 ** (adj_count / max(len(tokens), 1))) * (1.1 ** (adv_count / max(len(tokens), 1)))
    valence *= (0.8 ** verb_neg_count)

    # Flip valence if odd negations
    if negation_count % 2 == 1:
        valence = 1.0 - valence

    # Clamp to [0, 1]
    arousal = min(max(arousal, 0.0), 1.0)
    valence = min(max(valence, 0.0), 1.0)
    return arousal, valence

# Linear regression coefficients (Taylor & Francis 2020, fixed weights)
def calculate_linear_arousal(audio_df):
    weights = {
        'tempo': 0.4,
        'loudness': 0.3,
        'energy': 0.2,
        'speechiness': 0.05,
        'danceability': 0.05
    }
    arousal = sum(weights[f] * audio_df[f] for f in weights)
    arousal = (arousal - arousal.min()) / (arousal.max() - arousal.min() + 1e-10)
    return arousal

def calculate_linear_valence(audio_df):
    # Removed 'energy_extra'; redistributed weights
    weights = {
        'energy': 0.5,  # Increased to cover energy_extra
        'mode': 0.25,
        'tempo': 0.15,
        'danceability': 0.1
    }
    valence = sum(weights[f] * audio_df[f] for f in weights)
    valence = (valence - valence.min()) / (valence.max() - valence.min() + 1e-10)
    return valence

# Save metrics to CSV
def save_metrics(drive_folder, arousal_mse, arousal_r2, valence_mse, valence_r2, f1):
    metrics_df = pd.DataFrame({
        'Metric': ['Arousal MSE', 'Arousal R²', 'Valence MSE', 'Valence R²', 'F1 Score (Quadrant)'],
        'Value': [arousal_mse, arousal_r2, valence_mse, valence_r2, f1]
    })
    metrics_df.to_csv(drive_folder + 'training_metrics.csv', index=False)
    print("Training metrics saved to Google Drive.")
