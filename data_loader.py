import pandas as pd
import requests
from io import StringIO
from sklearn.preprocessing import MinMaxScaler
import os
from google.colab import drive
import sys

def mount_drive():
    if 'google.colab' in sys.modules:
        drive.mount('/content/drive')
    else:
        print("Not in Colab environment; skipping Drive mount.")

def create_drive_folder(folder_path):
    os.makedirs(folder_path, exist_ok=True)
    return folder_path

def download_csv(url):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return pd.read_csv(StringIO(response.text))
    except requests.RequestException as e:
        print(f"Error downloading {url}: {e}")
        return pd.DataFrame()

def load_datasets(xanew_url, emobank_url, spotify_url):
    xanew_df = download_csv(xanew_url)
    if not xanew_df.empty:
        xanew_df = xanew_df[['Word', 'V.Mean.Sum', 'A.Mean.Sum']].rename(columns={'Word': 'word', 'V.Mean.Sum': 'valence', 'A.Mean.Sum': 'arousal'})
        xanew_scaler = MinMaxScaler()
        xanew_df[['valence', 'arousal']] = xanew_scaler.fit_transform(xanew_df[['valence', 'arousal']])

    sentence_df = download_csv(emobank_url)
    song_df = download_csv(spotify_url)

    # Subset for testing (first 50 rows) - comment out for full datasets
    if not sentence_df.empty:
        sentence_df = sentence_df[sentence_df['split'] == 'train'].head(50)
    if not song_df.empty:
        song_df = song_df.head(50)

    # Normalize EmoBank arousal/valence
    if not sentence_df.empty:
        emo_scaler = MinMaxScaler()
        sentence_df[['V', 'A']] = emo_scaler.fit_transform(sentence_df[['V', 'A']])

    return xanew_df, sentence_df, song_df
