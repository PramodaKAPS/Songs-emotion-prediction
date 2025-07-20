import numpy as np
import tensorflow as tf
from sklearn.metrics import mean_squared_error, r2_score, f1_score
import pandas as pd

def create_mlp_regressor(input_dim):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu', input_dim=input_dim),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

def train_models(X_text_sentence, y_arousal, y_valence, drive_folder, assign_quadrant):
    input_dim = X_text_sentence.shape[1]
    arousal_model = create_mlp_regressor(input_dim)
    arousal_history = arousal_model.fit(X_text_sentence, y_arousal, epochs=10, verbose=1, validation_split=0.2)

    valence_model = create_mlp_regressor(input_dim)
    valence_history = valence_model.fit(X_text_sentence, y_valence, epochs=10, verbose=1, validation_split=0.2)

    y_arousal_pred = arousal_model.predict(X_text_sentence).flatten()
    y_valence_pred = valence_model.predict(X_text_sentence).flatten()

    arousal_mse = mean_squared_error(y_arousal, y_arousal_pred)
    arousal_r2 = r2_score(y_arousal, y_arousal_pred)
    valence_mse = mean_squared_error(y_valence, y_valence_pred)
    valence_r2 = r2_score(y_valence, y_valence_pred)

    true_quadrants = [assign_quadrant(a, v) for a, v in zip(y_arousal, y_valence)]
    pred_quadrants = [assign_quadrant(a, v) for a, v in zip(y_arousal_pred, y_valence_pred)]
    f1 = f1_score(true_quadrants, pred_quadrants, average='macro')

    print(f"Training Metrics:\nArousal MSE: {arousal_mse:.4f}, R²: {arousal_r2:.4f}\nValence MSE: {valence_mse:.4f}, R²: {valence_r2:.4f}\nF1 Score (Quadrant): {f1:.4f}")

    metrics_df = pd.DataFrame({
        'Metric': ['Arousal MSE', 'Arousal R²', 'Valence MSE', 'Valence R²', 'F1 Score (Quadrant)'],
        'Value': [arousal_mse, arousal_r2, valence_mse, valence_r2, f1]
    })
    metrics_df.to_csv(drive_folder + 'training_metrics.csv', index=False)
    print("Training metrics saved to Google Drive.")

    return arousal_model, valence_model
