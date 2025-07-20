import numpy as np
import tensorflow as tf
from sklearn.metrics import mean_squared_error, r2_score, f1_score

# Define simple MLP regressor model
def create_mlp_regressor(input_dim):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu', input_dim=input_dim),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1)  # Regression output
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

# Train models with epochs
def train_models(X, y_arousal, y_valence, epochs=10):
    input_dim = X.shape[1]
    arousal_model = create_mlp_regressor(input_dim)
    arousal_history = arousal_model.fit(X, y_arousal, epochs=epochs, verbose=1, validation_split=0.2)

    valence_model = create_mlp_regressor(input_dim)
    valence_history = valence_model.fit(X, y_valence, epochs=epochs, verbose=1, validation_split=0.2)
    return arousal_model, valence_model

# Compute training metrics
def compute_training_metrics(arousal_model, valence_model, X, y_arousal, y_valence, assign_quadrant):
    y_arousal_pred = arousal_model.predict(X).flatten()
    y_valence_pred = valence_model.predict(X).flatten()
    arousal_mse = mean_squared_error(y_arousal, y_arousal_pred)
    arousal_r2 = r2_score(y_arousal, y_arousal_pred)
    valence_mse = mean_squared_error(y_valence, y_valence_pred)
    valence_r2 = r2_score(y_valence, y_valence_pred)
    true_quadrants = [assign_quadrant(a, v) for a, v in zip(y_arousal, y_valence)]
    pred_quadrants = [assign_quadrant(a, v) for a, v in zip(y_arousal_pred, y_valence_pred)]
    f1 = f1_score(true_quadrants, pred_quadrants, average='macro')
    print(f"Training Metrics:\nArousal MSE: {arousal_mse:.4f}, R²: {arousal_r2:.4f}\nValence MSE: {valence_mse:.4f}, R²: {valence_r2:.4f}\nF1 Score (Quadrant): {f1:.4f}")
    return arousal_mse, arousal_r2, valence_mse, valence_r2, f1

# Predict on new data
def predict_models(arousal_model, valence_model, X):
    arousal_text = pd.Series(arousal_model.predict(X).flatten())
    valence_text = pd.Series(valence_model.predict(X).flatten())
    return arousal_text, valence_text

