import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
from google.colab import drive
import os
from utils import download_csv, preprocess_text, get_xanew_features, apply_pos_context, calculate_linear_arousal, calculate_linear_valence, save_metrics
from model import train_models, compute_training_metrics, predict_models

# Setup
drive_folder = '/content/drive/MyDrive/SongEmotionPredictions/'
os.makedirs(drive_folder, exist_ok=True)
drive.mount('/content/drive')

# Load DistilBERT
tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
model = AutoModel.from_pretrained('distilbert-base-uncased')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Load datasets
xanew_df = download_csv(XANEW_URL)
sentence_df = download_csv(EMOBANK_URL)
song_df = download_csv(SPOTIFY_URL)

# Use first 1000 rows for training and predictions (as requested)
if not sentence_df.empty:
    sentence_df = sentence_df[sentence_df['split'] == 'train'].head(1000)
if not song_df.empty:
    song_df = song_df.head(1000)

# Normalize EmoBank arousal/valence
if not sentence_df.empty:
    emo_scaler = MinMaxScaler()
    sentence_df[['V', 'A']] = emo_scaler.fit_transform(sentence_df[['V', 'A']])

# Audio features processing
audio_features = ['tempo', 'loudness', 'energy', 'speechiness', 'danceability', 'mode']
if not song_df.empty:
    song_df[audio_features] = song_df[audio_features].fillna(song_df[audio_features].mean())
    audio_scaled = MinMaxScaler().fit_transform(song_df[audio_features])
    audio_scaled_df = pd.DataFrame(audio_scaled, columns=audio_features)
arousal_audio = calculate_linear_arousal(audio_scaled_df) if not audio_scaled_df.empty else pd.Series()
valence_audio = calculate_linear_valence(audio_scaled_df) if not audio_scaled_df.empty else pd.Series()

# Preprocess texts
if not sentence_df.empty:
    sentence_df['tokens'], sentence_df['cleaned_text'] = zip(*sentence_df['text'].apply(preprocess_text))
if not song_df.empty:
    song_df['tokens'], song_df['cleaned_lyrics'] = zip(*song_df['lyrics'].apply(preprocess_text))

# Get X-ANEW features
if not sentence_df.empty:
    sentence_df['xanew_arousal'], sentence_df['xanew_valence'] = zip(*sentence_df['tokens'].apply(lambda x: get_xanew_features(x)))
if not song_df.empty:
    song_df['xanew_arousal'], song_df['xanew_valence'] = zip(*song_df['tokens'].apply(lambda x: get_xanew_features(x, is_lyric=True)))

# Apply POS context
if not sentence_df.empty:
    sentence_df['xanew_arousal'], sentence_df['xanew_valence'] = zip(*sentence_df.apply(
        lambda row: apply_pos_context(row['tokens'], row['xanew_arousal'], row['xanew_valence']), axis=1))
if not song_df.empty:
    song_df['xanew_arousal'], song_df['xanew_valence'] = zip(*song_df.apply(
        lambda row: apply_pos_context(row['tokens'], row['xanew_arousal'], row['xanew_valence']), axis=1))

# Extract embeddings
sentence_embeddings = get_bert_embeddings(sentence_df['cleaned_text']) if not sentence_df.empty else np.array([])
lyrics_embeddings = get_bert_embeddings(song_df['cleaned_lyrics']) if not song_df.empty else np.array([])

# Quadrant labels (adjusted for [-1,1] scale) - Moved earlier for F1 calculation
def assign_quadrant(arousal, valence):
    if arousal >= 0 and valence >= 0:
        return 'Happy/Excited'
    elif arousal >= 0 and valence < 0:
        return 'Angry/Stressed'
    elif arousal < 0 and valence >= 0:
        return 'Calm/Peaceful'
    else:
        return 'Sad/Depressed'

# Lyrics-based training with neural network for epochs
if not sentence_df.empty and sentence_embeddings.size > 0:
    X_text_sentence = np.hstack([sentence_embeddings, sentence_df[['xanew_arousal', 'xanew_valence']].values])
    y_arousal = sentence_df['A'].values
    y_valence = sentence_df['V'].values

    # Define simple MLP regressor model
    def create_mlp_regressor(input_dim):
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation='relu', input_dim=input_dim),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(1)  # Regression output
        ])
        model.compile(optimizer='adam', loss='mse')
        return model

    # Train arousal model with epochs
    input_dim = X_text_sentence.shape[1]
    arousal_model = create_mlp_regressor(input_dim)
    arousal_history = arousal_model.fit(X_text_sentence, y_arousal, epochs=10, verbose=1, validation_split=0.2)

    # Train valence model with epochs
    valence_model = create_mlp_regressor(input_dim)
    valence_history = valence_model.fit(X_text_sentence, y_valence, epochs=10, verbose=1, validation_split=0.2)

    # Predict on training data for metrics
    y_arousal_pred = arousal_model.predict(X_text_sentence).flatten()
    y_valence_pred = valence_model.predict(X_text_sentence).flatten()

    # Compute training "accuracy" (MSE and R² as proxies)
    arousal_mse = mean_squared_error(y_arousal, y_arousal_pred)
    arousal_r2 = r2_score(y_arousal, y_arousal_pred)
    valence_mse = mean_squared_error(y_valence, y_valence_pred)
    valence_r2 = r2_score(y_valence, y_valence_pred)

    # Discretize to quadrants for F1 score
    true_quadrants = [assign_quadrant(a, v) for a, v in zip(y_arousal, y_valence)]
    pred_quadrants = [assign_quadrant(a, v) for a, v in zip(y_arousal_pred, y_valence_pred)]
    f1 = f1_score(true_quadrants, pred_quadrants, average='macro')

    # Print metrics
    print(f"Training Metrics:\nArousal MSE: {arousal_mse:.4f}, R²: {arousal_r2:.4f}\nValence MSE: {valence_mse:.4f}, R²: {valence_r2:.4f}\nF1 Score (Quadrant): {f1:.4f}")

    # Save metrics to CSV
    metrics_df = pd.DataFrame({
        'Metric': ['Arousal MSE', 'Arousal R²', 'Valence MSE', 'Valence R²', 'F1 Score (Quadrant)'],
        'Value': [arousal_mse, arousal_r2, valence_mse, valence_r2, f1]
    })
    metrics_df.to_csv(drive_folder + 'training_metrics_first_1000.csv', index=False)
    print("Training metrics saved to Google Drive.")

    # Predict on Spotify data
    if not song_df.empty and lyrics_embeddings.size > 0:
        X_text_lyrics = np.hstack([lyrics_embeddings, song_df[['xanew_arousal', 'xanew_valence']].values])
        arousal_text = pd.Series(arousal_model.predict(X_text_lyrics).flatten())
        valence_text = pd.Series(valence_model.predict(X_text_lyrics).flatten())
    else:
        arousal_text = pd.Series()
        valence_text = pd.Series()
else:
    arousal_text = pd.Series()
    valence_text = pd.Series()

# Combine predictions
w_text = 0.6
w_audio = 0.4
arousal_final = w_text * arousal_text + w_audio * arousal_audio if not arousal_text.empty else arousal_audio
valence_final = w_text * valence_text + w_audio * valence_audio if not valence_text.empty else valence_audio

# Scale final arousal and valence to [-1, 1]
arousal_final = 2 * arousal_final - 1
valence_final = 2 * valence_final - 1

# Save predictions to Google Drive
if not song_df.empty:
    predictions_df = pd.DataFrame({
        'track_id': song_df['track_id'],
        'track_name': song_df['track_name'],
        'track_artist': song_df['track_artist'],
        'arousal_text': arousal_text,
        'valence_text': valence_text,
        'arousal_audio': arousal_audio,
        'valence_audio': valence_audio,
        'arousal_final': arousal_final,
        'valence_final': valence_final
    })
    predictions_df.to_csv(drive_folder + 'song_emotion_predictions_taylor_francis_first_1000.csv', index=False)
    print("Predictions saved to Google Drive: " + drive_folder + 'song_emotion_predictions_taylor_francis_first_1000.csv')

# Validation
if not song_df.empty and not valence_audio.empty:
    mse_valence = mean_squared_error(song_df['valence'], valence_audio)
    print(f"Valence MSE (linear audio vs. Spotify valence): {mse_valence:.4f}")

if not arousal_audio.empty and not arousal_text.empty:
    corr, _ = pearsonr(arousal_audio, arousal_text)
    print(f"Arousal correlation (audio vs. text): {corr:.4f}")

# Thayer's Plot (adjusted for [-1,1] scale)
def create_thayer_plot(predictions_df, output_file=drive_folder + 'thayer_plot_taylor_francis_first_1000.png'):
    if predictions_df.empty:
        print("No data for plot.")
        return
    sns.set_style("darkgrid")
    plt.figure(figsize=(10, 8))
    sns.scatterplot(data=predictions_df, x='valence_final', y='arousal_final',
                    hue='valence_final', size='arousal_final',
                    palette='viridis', alpha=0.6, sizes=(20, 200))
    plt.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    plt.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    plt.text(-0.5, 0.5, 'Angry/Stressed', fontsize=10, ha='center')
    plt.text(-0.5, -0.5, 'Sad/Depressed', fontsize=10, ha='center')
    plt.text(0.5, 0.5, 'Happy/Excited', fontsize=10, ha='center')
    plt.text(0.5, -0.5, 'Calm/Peaceful', fontsize=10, ha='center')
    top_songs = predictions_df.nlargest(5, 'arousal_final')
    for _, row in top_songs.iterrows():
        plt.text(row['valence_final'], row['arousal_final'], row['track_name'],
                 fontsize=8, ha='right', va='bottom')
    plt.xlabel('Valence (Negative to Positive)', fontsize=12)
    plt.ylabel('Arousal (Calm to Excited)', fontsize=12)
    plt.title('Thayer\'s Emotion Plane for Spotify Songs (Taylor & Francis 2020)', fontsize=14)
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    # plt.show()  # Comment out for headless environments
    print(f"Thayer's plot saved to Google Drive: '{output_file}'")

if 'predictions_df' in locals() and not predictions_df.empty:
    create_thayer_plot(predictions_df)

# Quadrant labels (adjusted for [-1,1] scale)
def assign_quadrant(arousal, valence):
    if arousal >= 0 and valence >= 0:
        return 'Happy/Excited'
    elif arousal >= 0 and valence < 0:
        return 'Angry/Stressed'
    elif arousal < 0 and valence >= 0:
        return 'Calm/Peaceful'
    else:
        return 'Sad/Depressed'

if 'predictions_df' in locals() and not predictions_df.empty:
    predictions_df['quadrant'] = predictions_df.apply(
        lambda row: assign_quadrant(row['arousal_final'], row['valence_final']), axis=1)
    predictions_df.to_csv(drive_folder + 'song_emotion_predictions_with_quadrant_taylor_francis_first_1000.csv', index=False)
    print("Predictions with quadrant labels saved to Google Drive: " + drive_folder + 'song_emotion_predictions_with_quadrant_taylor_francis_first_1000.csv')

# Function to add Spotify columns to final output CSV (as requested)
def add_spotify_columns_to_final_csv(song_df, predictions_df, drive_folder):
    if predictions_df.empty or song_df.empty:
        print("No data to merge.")
        return
    
    # Merge predictions into the original song_df (adds all Spotify columns + predictions)
    final_df = song_df.copy()  # Copy original Spotify DataFrame
    final_df = final_df.join(predictions_df.set_index(['track_id', 'track_name', 'track_artist']), on=['track_id', 'track_name', 'track_artist'])
    
    # Save the final enriched DataFrame
    final_output_file = drive_folder + 'spotify_full_with_predictions_first_1000.csv'
    final_df.to_csv(final_output_file, index=False)
    print(f"Final output CSV with all Spotify columns and predictions saved to: {final_output_file}")

# Call the function after predictions are ready
if 'predictions_df' in locals() and not predictions_df.empty and not song_df.empty:
    add_spotify_columns_to_final_csv(song_df, predictions_df, drive_folder)

# Output the 1000 predicted rows as a separate CSV (as requested)
if not predictions_df.empty:
    predictions_df.to_csv(drive_folder + 'spotify_first_1000_predictions.csv', index=False)
    print("First 1000 predicted rows saved to Google Drive: " + drive_folder + 'spotify_first_1000_predictions.csv')


