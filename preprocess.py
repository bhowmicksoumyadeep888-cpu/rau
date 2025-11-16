import pandas as pd

def load_and_preprocess(path):
    df = pd.read_csv(path)

    # Convert to datetime
    df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'])

    # Extract features
    df["hour"] = df["pickup_datetime"].dt.hour
    df["day"] = df["pickup_datetime"].dt.dayofweek
    df["month"] = df["pickup_datetime"].dt.month

    # Aggregate demand per zone per hour
    agg = df.groupby(["zone_id", "hour", "day", "month"]).size().reset_index(name="demand")

    return agg
