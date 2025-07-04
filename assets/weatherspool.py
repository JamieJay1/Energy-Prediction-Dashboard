import pandas as pd

# Load weather and metadata (if not already loaded)
weather = pd.read_csv('/content/drive/MyDrive/bdg2_energy_project/bdg2_data/weather.csv', parse_dates=['timestamp'])
metadata = pd.read_csv('/content/drive/MyDrive/bdg2_energy_project/bdg2_data/bdg2_data/metadata.csv')

# Top 5 buildings and their site_ids
top_5_buildings = ['Hog_office_Lizzie', 'Hog_education_Jewel', 'Hog_public_Octavia',
                   'Hog_lodging_Francisco', 'Hog_assembly_Dona']

top_5_site_ids = metadata[metadata['building_id'].isin(top_5_buildings)]['site_id'].unique()

# Filter weather data to relevant site_ids
weather_top_sites = weather[weather['site_id'].isin(top_5_site_ids)].copy()

# Create date column
weather_top_sites['date'] = weather_top_sites['timestamp'].dt.date

# Daily average weather by site
daily_weather = weather_top_sites.groupby(['site_id', 'date']).agg({
    'airTemperature': 'mean',
    'dewTemperature': 'mean',
    'windSpeed': 'mean',
    'seaLvlPressure': 'mean',
    'precipDepth1HR': 'mean'
}).reset_index()

# Save to CSV
daily_weather.to_csv('/content/drive/MyDrive/bdg2_energy_project/energy_dashboard/assets/weather.csv', index=False)
print("weather.csv saved successfully.")
