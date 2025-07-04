# Convert date back to datetime if needed
daily_df['date'] = pd.to_datetime(daily_df['date'])

# Sort before creating lag/rolling features
daily_df = daily_df.sort_values(['building_id', 'date'])

# Create time and lag features
def create_features(df):
    df['dayofweek'] = df['date'].dt.dayofweek
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day

    df['lag_1'] = df.groupby('building_id')['value'].shift(1)
    df['lag_7'] = df.groupby('building_id')['value'].shift(7)
    df['rolling_mean_7'] = df.groupby('building_id')['value'].shift(1).rolling(7).mean().reset_index(0, drop=True)
    df['rolling_std_7'] = df.groupby('building_id')['value'].shift(1).rolling(7).std().reset_index(0, drop=True)

    return df

daily_df = create_features(daily_df)
daily_df = daily_df.dropna()
