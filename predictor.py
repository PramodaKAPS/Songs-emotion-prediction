import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr

def predict_text(arousal_model, valence_model, X_text_lyrics):
    arousal_text = pd.Series(arousal_model.predict(X_text_lyrics).flatten())
    valence_text = pd.Series(valence_model.predict(X_text_lyrics).flatten())
    return arousal_text, valence_text

def combine_predictions(arousal_text, valence_text, arousal_audio, valence_audio):
    w_text = 0.6
    w_audio = 0.4
    arousal_final = w_text * arousal_text + w_audio * arousal_audio if not arousal_text.empty else arousal_audio
    valence_final = w_text * valence_text + w_audio * valence_audio if not valence_text.empty else valence_audio

    arousal_final = 2 * arousal_final - 1
    valence_final = 2 * valence_final - 1
    return arousal_final, valence_final

def save_predictions(song_df, arousal_text, valence_text, arousal_audio, valence_audio, arousal_final, valence_final, drive_folder):
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
    predictions_df.to_csv(drive_folder + 'song_emotion_predictions_taylor_francis.csv', index=False)
    print("Predictions saved to Google Drive: " + drive_folder + 'song_emotion_predictions_taylor_francis.csv')
    return predictions_df

def validate_predictions(song_df, valence_audio, arousal_audio, arousal_text):
    if not song_df.empty and not valence_audio.empty:
        mse_valence = mean_squared_error(song_df['valence'], valence_audio)
        print(f"Valence MSE (linear audio vs. Spotify valence): {mse_valence:.4f}")

    if not arousal_audio.empty and not arousal_text.empty:
        corr, _ = pearsonr(arousal_audio, arousal_text)
        print(f"Arousal correlation (audio vs. text): {corr:.4f}")

def add_quadrant_and_save(predictions_df, drive_folder, assign_quadrant):
    predictions_df['quadrant'] = predictions_df.apply(
        lambda row: assign_quadrant(row['arousal_final'], row['valence_final']), axis=1)
    predictions_df.to_csv(drive_folder + 'song_emotion_predictions_with_quadrant_taylor_francis.csv', index=False)
    print("Predictions with quadrant labels saved to Google Drive: " + drive_folder + 'song_emotion_predictions_with_quadrant_taylor_francis.csv')

def add_spotify_columns_to_final_csv(song_df, predictions_df, drive_folder):
    if predictions_df.empty or song_df.empty:
        print("No data to merge.")
        return

    final_df = song_df.copy()
    final_df = final_df.join(predictions_df.set_index(['track_id', 'track_name', 'track_artist']), on=['track_id', 'track_name', 'track_artist'])

    final_output_file = drive_folder + 'spotify_full_with_predictions.csv'
    final_df.to_csv(final_output_file, index=False)
    print(f"Final output CSV with all Spotify columns and predictions saved to: {final_output_file}")
