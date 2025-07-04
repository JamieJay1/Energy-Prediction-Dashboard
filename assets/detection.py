import pandas as pd
from sklearn.ensemble import IsolationForest

# Create summary list
summary = []

# Create full anomaly records list (for anomalies.csv)
anomaly_records = []

# Top 5 buildings
top_5_buildings = ['Hog_office_Lizzie', 'Hog_education_Jewel', 'Hog_public_Octavia',
                   'Hog_lodging_Francisco', 'Hog_assembly_Dona']

for building_id in top_5_buildings:
    building_df = daily_df[daily_df['building_id'] == building_id].copy()
    building_df = building_df.dropna(subset=['value', 'airTemperature', 'dewTemperature', 'windSpeed', 'seaLvlPressure'])

    # Select features
    feature_cols = ['value', 'airTemperature', 'dewTemperature', 'windSpeed', 'seaLvlPressure']
    X = building_df[feature_cols]

    # Apply Isolation Forest
    iso_forest = IsolationForest(n_estimators=100, contamination='auto', random_state=42)
    preds = iso_forest.fit_predict(X)
    building_df['is_anomaly'] = (preds == -1).astype(int)


    # Convert to daily level: mark a day as anomalous if any anomaly occurred that day
    daily_anomaly = building_df.groupby('date')['is_anomaly'].max().reset_index()
    daily_anomaly['building_id'] = building_id
    daily_anomaly['model'] = 'isolation_forest'

    # Store for full export
    anomaly_records.append(daily_anomaly)

    # Summary
    total_days = len(daily_anomaly)
    anomaly_days = daily_anomaly['is_anomaly'].sum()
    anomaly_rate = round((anomaly_days / total_days) * 100, 2)

    summary.append({
        'Building': building_id,
        'Total Days': total_days,
        'Anomalies Detected': anomaly_days,
        'Anomaly Rate (%)': anomaly_rate
    })

# Convert and print summary
summary_df = pd.DataFrame(summary)
print(summary_df)

# Concatenate full anomalies and export
anomalies_df = pd.concat(anomaly_records, ignore_index=True)
anomalies_df.to_csv('/content/drive/MyDrive/bdg2_energy_project/energy_dashboard/data/anomalies.csv', index=False)
print("✅ anomalies.csv saved!")
