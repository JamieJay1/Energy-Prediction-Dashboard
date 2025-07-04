import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# Top 5 buildings
top_5_buildings = [
    'Hog_office_Lizzie',
    'Hog_education_Jewel',
    'Hog_public_Octavia',
    'Hog_lodging_Francisco',
    'Hog_assembly_Dona'
]

# Collect all results here
all_anomalies = []

# Loop through each building
for building in top_5_buildings:
    df_building = df_long[df_long['building_id'] == building].copy()
    df_building = df_building.sort_values('timestamp')

    # Feature engineering
    df_building['lag_1'] = df_building['value'].shift(1)
    df_building['lag_7'] = df_building['value'].shift(7)
    df_building['rolling_mean_7'] = df_building['value'].rolling(7).mean()
    df_building['rolling_std_7'] = df_building['value'].rolling(7).std()
    df_building['dayofweek'] = df_building['timestamp'].dt.dayofweek
    df_building['month'] = df_building['timestamp'].dt.month

    # Drop rows with missing values
    df_building.dropna(inplace=True)

    # Feature selection
    features = [
        'value', 'lag_1', 'lag_7', 'rolling_mean_7', 'rolling_std_7',
        'dayofweek', 'month',
        'airTemperature', 'dewTemperature', 'windSpeed', 'seaLvlPressure'
    ]

    # Normalize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_building[features])

    # Apply Isolation Forest
    iso = IsolationForest(n_estimators=100, contamination="auto", random_state=42)
    df_building['anomaly'] = iso.fit_predict(X_scaled)

    # Add to result
    all_anomalies.append(df_building[['timestamp', 'building_id', 'value', 'anomaly']])

# Combine and save to CSV
anomaly_df = pd.concat(all_anomalies, ignore_index=True)
anomaly_df.to_csv("/content/drive/MyDrive/bdg2_energy_project/energy_dashboard/data/anomalies.csv", index=False)
print("anomalies.csv saved with shape:", anomaly_df.shape)
