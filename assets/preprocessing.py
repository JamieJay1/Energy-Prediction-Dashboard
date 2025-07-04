
import pandas as pd

# Load data
electricity = pd.read_csv('/content/drive/MyDrive/bdg2_energy_project/bdg2_data/electricity_cleaned.csv', parse_dates=['timestamp'])
metadata = pd.read_csv('/content/drive/MyDrive/bdg2_energy_project/bdg2_data/metadata.csv')
weather = pd.read_csv('/content/drive/MyDrive/bdg2_energy_project/bdg2_data/weather.csv', parse_dates=['timestamp'])

# Step 1: Select top 5 buildings (you can replace these with your actual top performers)
top_5_buildings = ['Hog_office_Lizzie', 'Hog_education_Jewel', 'Hog_public_Octavia',
                   'Hog_lodging_Francisco', 'Hog_assembly_Dona']

# Step 2: Filter electricity data
electricity_subset = electricity[['timestamp'] + top_5_buildings]

# Step 3: Melt to long format
df_long = electricity_subset.melt(id_vars='timestamp', var_name='building_id', value_name='value')
print("After melting:", df_long.shape)

# Step 4: Merge with metadata to get site_id
df_long = df_long.merge(metadata[['building_id', 'site_id']], on='building_id', how='left')
print("After merging metadata:", df_long['site_id'].isna().sum(), "rows with missing site_id")

# Step 5: Merge with weather data on timestamp and site_id
df_long = df_long.merge(weather, on=['timestamp', 'site_id'], how='left')
print("After merging weather:", df_long.shape)
print("Missing airTemperature:", df_long['airTemperature'].isna().sum())

# Final check
print(df_long.head())
df_long = df_long.dropna(subset=['value'])
df_long = df_long.fillna(method='ffill')
df_long['date'] = pd.to_datetime(df_long['timestamp']).dt.date
daily_df = df_long.groupby(['building_id', 'date']).agg({
    'value': 'sum',
    'airTemperature': 'mean',
    'dewTemperature': 'mean',
    'precipDepth1HR': 'sum',
    'windSpeed': 'mean',
    'seaLvlPressure': 'mean'
}).reset_index()

