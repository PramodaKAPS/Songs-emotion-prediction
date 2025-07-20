import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def process_audio(song_df):
    audio_features = ['tempo', 'loudness', 'energy', 'speechiness', 'danceability', 'mode']
    if not song_df.empty:
        # Handle missing values
        song_df[audio_features] = song_df[audio_features].fillna(song_df[audio_features].mean())

        # Normalize audio features
        scaler_audio = MinMaxScaler()
        audio_scaled = scaler_audio.fit_transform(song_df[audio_features])
        audio_scaled_df = pd.DataFrame(audio_scaled, columns=audio_features)
    else:
        audio_scaled_df = pd.DataFrame()

    return audio_scaled_df

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
    weights = {
        'energy': 0.5,
        'mode': 0.25,
        'tempo': 0.15,
        'danceability': 0.1
    }
    valence = sum(weights[f] * audio_df[f] for f in weights)
    valence = (valence - valence.min()) / (valence.max() - valence.min() + 1e-10)
    return valence
