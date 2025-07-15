import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping

# Define input features and target
features = [
    'lag_1', 'lag_7', 'rolling_mean_7', 'rolling_std_7',
    'dayofweek', 'month',
    'airTemperature', 'dewTemperature', 'precipDepth1HR',
    'windSpeed', 'seaLvlPressure'
]
target = 'value'

# Function to prepare LSTM inputs
def prepare_lstm_data(df, features, target, window=14):
    df = df.dropna(subset=features + [target])
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df[features + [target]])
    X, y = [], []
    for i in range(window, len(scaled)):
        X.append(scaled[i-window:i, :-1])
        y.append(scaled[i, -1])
    return np.array(X), np.array(y), scaler, df.iloc[window:].reset_index(drop=True)

# Track performance
lstm_results = []

# Get top 5 buildings by data size
top5 = df_long['building_id'].value_counts().index[:5]

for bld in top5:
    df_b = df_long[df_long['building_id'] == bld].copy()
    df_b = df_b.sort_values('timestamp')

    # Feature engineering
    df_b['dayofweek'] = df_b['timestamp'].dt.dayofweek
    df_b['month'] = df_b['timestamp'].dt.month
    df_b['lag_1'] = df_b['value'].shift(1)
    df_b['lag_7'] = df_b['value'].shift(7)
    df_b['rolling_mean_7'] = df_b['value'].rolling(7).mean()
    df_b['rolling_std_7'] = df_b['value'].rolling(7).std()

    # Prepare data for LSTM
    df_b = df_b.dropna()
    X, y, scaler, df_features = prepare_lstm_data(df_b, features, target)

    # Split train/test
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # Build LSTM model
    model = Sequential()
    model.add(LSTM(64, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')
    early_stop = EarlyStopping(monitor='val_loss', patience=5)

    model.fit(X_train, y_train, validation_split=0.1, epochs=20, batch_size=32,
              verbose=0, callbacks=[early_stop])

    # Make predictions
    y_pred = model.predict(X_test)

    # Inverse transform
    dummy_input = np.zeros((len(y_pred), len(features)+1))
    dummy_input[:, -1] = y_pred[:, 0]
    y_pred_inv = scaler.inverse_transform(dummy_input)[:, -1]

    dummy_input[:, -1] = y_test
    y_test_inv = scaler.inverse_transform(dummy_input)[:, -1]

    # Timestamps for test set
    test_timestamps = df_features['timestamp'].iloc[split:].reset_index(drop=True)

    # Save predictions for this building
    pred_df = pd.DataFrame({
        'timestamp': test_timestamps,
        'building_id': bld,
        'y_true': y_test_inv,
        'y_pred': y_pred_inv
    })

    if 'all_preds' not in locals():
        all_preds = pred_df
    else:
        all_preds = pd.concat([all_preds, pred_df], ignore_index=True)

    # Store evaluation metrics
    rmse = np.sqrt(mean_squared_error(y_test_inv, y_pred_inv))
    mae = mean_absolute_error(y_test_inv, y_pred_inv)

    lstm_results.append({
        'building': bld,
        'rmse_lstm': rmse,
        'mae_lstm': mae
    })

# Save full predictions to CSV
all_preds.to_csv("/content/drive/MyDrive/bdg2_energy_project/energy_dashboard/data/predictions.csv", index=False)
print("Saved predictions.csv with shape:", all_preds.shape)

# Print RMSE/MAE results
lstm_df = pd.DataFrame(lstm_results)
print("\n Evaluation Summary:")
print(lstm_df)
