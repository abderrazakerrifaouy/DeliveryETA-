import numpy as np
import pandas as pd


class DataCleaner:

    NUMERIC_COLS = ["Delivery_person_Age", "Delivery_person_Ratings", "multiple_deliveries"]
    STRIP_COLS = [ "ID", "Delivery_person_ID", "Road_traffic_density", "Type_of_order","Type_of_vehicle", "Festival", "City"]
    COORD_COLS = ["Restaurant_latitude", "Restaurant_longitude","Delivery_location_latitude", "Delivery_location_longitude"]

    def __init__(self, path: str):
        self.path = path
        self.report: list[str] = []  

    def load(self) -> pd.DataFrame:
        df = pd.read_csv(self.path)
        self._log(f"Chargement : {df.shape[0]} lignes, {df.shape[1]} colonnes.")
        return df


    def clean_text_noise(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        for col in self.STRIP_COLS:
            df[col] = df[col].astype(str).str.strip()

        df["Weatherconditions"] = df["Weatherconditions"].astype(str).str.replace("conditions", "", regex=False).str.strip() 

        df["Time_taken(min)"] =  df["Time_taken(min)"].astype(str).str.extract(r"(\d+)").astype(float)

        self._log("Texte parasite retiré (espaces, préfixes 'conditions', '(min)').")
        return df


    def convert_types(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        for col in self.NUMERIC_COLS:
            df[col] = df[col].astype(str).str.strip().replace({"NaN": np.nan})
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df["Order_Date"] = pd.to_datetime(df["Order_Date"], format="%d-%m-%Y", errors="coerce")
        df["Time_Orderd"] = pd.to_datetime(df["Time_Orderd"], format="%H:%M:%S", errors="coerce").dt.time
        df["Time_Order_picked"] = pd.to_datetime(
            df["Time_Order_picked"], format="%H:%M:%S", errors="coerce"
        ).dt.time

        for col in ["Road_traffic_density", "multiple_deliveries", "Festival", "City", "Weatherconditions"]:
            df[col] = df[col].replace({"NaN": np.nan, "nan": np.nan})

        self._log("Types convertis : numériques (Age, Ratings, multiple_deliveries), dates et heures.")
        return df


    def handle_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        for col in ["Delivery_person_Age", "Delivery_person_Ratings"]:
            median = df[col].median()
            n_missing = df[col].isna().sum()
            df[col] = df[col].fillna(median)
            self._log(f"{col} : {n_missing} valeurs manquantes imputées par la médiane ({median}).")

        categorical_cols = [
            "Road_traffic_density", "City", "Festival",
            "multiple_deliveries", "Weatherconditions",
        ]
        for col in categorical_cols:
            mode = df[col].mode(dropna=True).iloc[0]
            n_missing = df[col].isna().sum()
            df[col] = df[col].fillna(mode)
            self._log(f"{col} : {n_missing} valeurs manquantes imputées par le mode ('{mode}').")

        for col in ["Time_Orderd", "Time_Order_picked"]:
            mode = df[col].mode(dropna=True).iloc[0]
            n_missing = df[col].isna().sum()
            df[col] = df[col].fillna(mode)
            self._log(f"{col} : {n_missing} valeurs manquantes imputées par le mode ({mode}).")

        return df


    def handle_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        for col in self.COORD_COLS:
            n_neg = (df[col] < 0).sum()
            df[col] = df[col].abs()
            self._log(f"{col} : {n_neg} valeurs négatives corrigées (valeur absolue).")

        zero_mask = (
            (df["Restaurant_latitude"] == 0) | (df["Restaurant_longitude"] == 0) |
            (df["Delivery_location_latitude"] == 0) | (df["Delivery_location_longitude"] == 0)
        )
        n_zero = zero_mask.sum()
        df = df[~zero_mask]
        self._log(f"Coordonnées (0, 0) : {n_zero} lignes supprimées (point invalide, hors Inde).")

        q1, q3 = df["Time_taken(min)"].quantile([0.25, 0.75])
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        n_out = ((df["Time_taken(min)"] < lower) | (df["Time_taken(min)"] > upper)).sum()
        df = df[(df["Time_taken(min)"] >= lower) & (df["Time_taken(min)"] <= upper)]
        self._log(f"Time_taken(min) : {n_out} valeurs aberrantes retirées (bornes IQR [{lower:.1f}, {upper:.1f}]).")

        return df


    def drop_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        n_before = len(df)
        df = df.drop_duplicates(subset=[c for c in df.columns if c != "ID"])
        n_removed = n_before - len(df)
        self._log(f"Doublons (hors ID) : {n_removed} lignes supprimées.")
        return df


    def run(self) -> pd.DataFrame:
        df = self.load()
        df = self.clean_text_noise(df)
        df = self.convert_types(df)
        df = self.handle_missing(df)
        df = self.handle_outliers(df)
        df = self.drop_duplicates(df)
        self._log(f"Dataset nettoyé final : {df.shape[0]} lignes, {df.shape[1]} colonnes.")
        return df.reset_index(drop=True)

    def save(self, df: pd.DataFrame, out_path: str):
        df.to_csv(out_path, index=False)
        self._log(f"Sauvegardé : {out_path}")

    def _log(self, msg: str):
        self.report.append(msg)
        print(msg)


def run_cleaning_pipeline(input_path: str, output_path: str) -> None:
    cleaner = DataCleaner(input_path)
    df_clean = cleaner.run()
    cleaner.save(df_clean, output_path)

if __name__ == "__main__":
    input_path = "data/raw/dataset_delivery_time.csv"
    output_path = "data/processed/dataset_delivery_time_clean.csv"
    run_cleaning_pipeline(input_path, output_path)