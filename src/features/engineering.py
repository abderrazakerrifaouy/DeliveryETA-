import numpy as np
import pandas as pd


def haversine_km(lat1, lon1, lat2, lon2) -> np.ndarray:
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def add_distance_km(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Distance_km"] = haversine_km( df["Restaurant_latitude"], df["Restaurant_longitude"], df["Delivery_location_latitude"], df["Delivery_location_longitude"] )
    return df


def add_preparation_time(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    t_order = pd.to_datetime(df["Time_Orderd"], format="%H:%M:%S")
    t_pick = pd.to_datetime(df["Time_Order_picked"], format="%H:%M:%S")
    delta_min = (t_pick - t_order).dt.total_seconds() / 60
    delta_min = np.where(delta_min < 0, delta_min + 24 * 60, delta_min)
    df["Preparation_time_min"] = np.clip(delta_min, 0, 60)
    return df


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    t_order = pd.to_datetime(df["Time_Orderd"], format="%H:%M:%S")
    df["Order_hour"] = t_order.dt.hour

    order_date = pd.to_datetime(df["Order_Date"])
    df["Day_of_week"] = order_date.dt.dayofweek
    df["Is_weekend"] = df["Day_of_week"].isin([5, 6]).astype(int)
    return df


def select_model_features(df: pd.DataFrame) -> pd.DataFrame:
    drop_cols = [
        "ID", "Delivery_person_ID", "Restaurant_latitude", "Restaurant_longitude", "Delivery_location_latitude", "Delivery_location_longitude", "Order_Date", "Time_Orderd", "Time_Order_picked", "Delivery_person_Ratings"]
    return df.drop(columns=drop_cols)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = add_distance_km(df)
    df = add_preparation_time(df)
    df = add_time_features(df)
    return df


def run_engineering_pipeline(input_path: str, output_path: str) -> None:
    df = pd.read_csv(input_path)
    df_engineered = engineer_features(df)
    df_model = select_model_features(df_engineered)
    df_model.to_csv(output_path, index=False)
    print(f"Colonnes retenues pour la modélisation : {df_model.shape}")
    print(df_model.columns.tolist())

if __name__ == "__main__":
    input_path = "data/processed/dataset_delivery_time_clean.csv"
    output_path = "data/processed/dataset_delivery_time_model_ready.csv"
    run_engineering_pipeline(input_path, output_path)