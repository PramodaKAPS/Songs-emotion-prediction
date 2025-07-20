import numpy as np
import pandas as pd
from data_loader import mount_drive, create_drive_folder, load_datasets
from audio_processor import process_audio, calculate_linear_arousal, calculate_linear_valence
from text_processor import load_bert, preprocess_text, get_xanew_features, apply_pos_context, get_bert_embeddings
from model_trainer import train_models
from predictor import predict_text, combine_predictions, save_predictions, validate_predictions, add_quadrant_and_save, add_spotify_columns_to_final_csv
from visualizer import create_thayer_plot
from utils import download_nltk_resources, assign_quadrant

# URLs
XANEW_URL = 'https://raw.githubusercontent.com/JULIELab/XANEW/master/Ratings_Warriner_et_al.csv'
EMOBANK_URL = 'https://raw.githubusercontent.com/PramodaKAPS/SongsEmotions/main/emobank.csv'
SPOTIFY_URL = 'https://raw.githubusercontent.com/PramodaKAPS/SongsEmotions/main/spotify_songs.csv'

# Setup
mount_drive()
drive_folder = create_drive_folder('/content/drive/MyDrive/SongEmotionPredictions/')
download_nltk_resources()

# Load data
xanew_df, sentence_df, song_df = load_datasets(XANEW_URL, EMOBANK_URL, SPOTIFY_URL)

# Process audio
audio_scaled_df = process_audio(song_df)
arousal_audio = calculate_linear_arousal(audio_scaled_df) if not audio_scaled_df.empty else pd.Series()
valence_audio = calculate_linear_valence(audio_scaled_df) if not audio_scaled_df.empty else pd.Series()

# Load BERT
tokenizer, model, device = load_bert()

# Preprocess texts
if not sentence_df.empty:
    sentence_df['tokens'], sentence_df['cleaned_text'] = zip(*sentence_df['text'].apply(preprocess_text))
if not song_df.empty:
    song_df['tokens'], song_df['cleaned_lyrics'] = zip(*song_df['lyrics'].apply(preprocess_text))

# Get X-ANEW features
if not sentence_df.empty:
    sentence_df['xanew_arousal'], sentence_df['xanew_valence'] = zip(*sentence_df['tokens'].apply(lambda x: get_xanew_features(x, xanew_df)))
if not song_df.empty:
    song_df['xanew_arousal'], song_df['xanew_valence'] = zip(*song_df['tokens'].apply(lambda x: get_xanew_features(x, xanew_df, is_lyric=True)))

# Apply POS context
if not sentence_df.empty:
    sentence_df['xanew_arousal'], sentence_df['xanew_valence'] = zip(*sentence_df.apply(
        lambda row: apply_pos_context(row['tokens'], row['xanew_arousal'], row['xanew_valence']), axis=1))
if not song_df.empty:
    song_df['xanew_arousal'], song_df['xanew_valence'] = zip(*song_df.apply(
        lambda row: apply_pos_context(row['tokens'], row['xanew_arousal'], row['xanew_valence']), axis=1))

# Extract embeddings
sentence_embeddings = get_bert_embeddings(sentence_df['cleaned_text'], tokenizer, model, device) if not sentence_df.empty else np.array([])
lyrics_embeddings = get_bert_embeddings(song_df['cleaned_lyrics'], tokenizer, model, device) if not song_df.empty else np.array([])

# Train models
if not sentence_df.empty and sentence_embeddings.size > 0:
    X_text_sentence = np.hstack([sentence_embeddings, sentence_df[['xanew_arousal', 'xanew_valence']].values])
    y_arousal = sentence_df['A'].values
    y_valence = sentence_df['V'].values
    arousal_model, valence_model = train_models(X_text_sentence, y_arousal, y_valence, drive_folder, assign_quadrant)

    # Predict on Spotify data
    if not song_df.empty and lyrics_embeddings.size > 0:
        X_text_lyrics = np.hstack([lyrics_embeddings, song_df[['xanew_arousal', 'xanew_valence']].values])
        arousal_text, valence_text = predict_text(arousal_model, valence_model, X_text_lyrics)
    else:
        arousal_text = pd.Series()
        valence_text = pd.Series()
else:
    arousal_text = pd.Series()
    valence_text = pd.Series()

# Combine and save predictions
arousal_final, valence_final = combine_predictions(arousal_text, valence_text, arousal_audio, valence_audio)
if not song_df.empty:
    predictions_df = save_predictions(song_df, arousal_text, valence_text, arousal_audio, valence_audio, arousal_final, valence_final, drive_folder)

    # Validate
    validate_predictions(song_df, valence_audio, arousal_audio, arousal_text)

    # Create plot
    create_thayer_plot(predictions_df, drive_folder + 'thayer_plot_taylor_francis.png')

    # Add quadrant and save
    add_quadrant_and_save(predictions_df, drive_folder, assign_quadrant)

    # Add Spotify columns
    add_spotify_columns_to_final_csv(song_df, predictions_df, drive_folder)
