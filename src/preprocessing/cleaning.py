import pandas as pd


class DataCleaner:

    def __init__(self, filepath):
        self.filepath = filepath
        self.df = None

    def load_data(self):
        self.df = pd.read_csv(self.filepath)
        return self.df
  



    def clean_text(self):

        categorical_cols = [
            "Weather",
            "Traffic_Level",
            "Time_of_Day",
            "Vehicle_Type"
        ]

        for col in categorical_cols:
            self.df[col] = self.df[col].str.strip()



    def convert_types(self):

        numeric_cols = [
            "Order_ID",
            "Distance_km",
            "Preparation_Time_min",
            "Courier_Experience_yrs",
            "Delivery_Time_min"
        ]

        for col in numeric_cols:
            self.df[col] = pd.to_numeric(
                self.df[col],
                errors="coerce"
            )



    



    def handle_missing_values(self):

        self.df["Courier_Experience_yrs"] = (
            self.df["Courier_Experience_yrs"]
            .fillna(
                self.df["Courier_Experience_yrs"].median()
            )
        )

        categorical_cols = [
            "Weather",
            "Traffic_Level",
            "Time_of_Day"
        ]

        for col in categorical_cols:
            self.df[col] = self.df[col].fillna(
                self.df[col].mode()[0]
            )



    def remove_duplicates(self):


        before = len(self.df)

        self.df = self.df.drop_duplicates()

        after = len(self.df)

        removed = before - after

        print("Doublons supprimés :", removed)



    def validate_data(self):

        print("\n VALIDATION FINALE ")

        print("Dimensions :", self.df.shape)

        print("\nValeurs manquantes :")
        print(self.df.isnull().sum())

        print("\nDoublons :", self.df.duplicated().sum())

        print("\nTypes :")
        print(self.df.dtypes)


    def clean(self):

        self.load_data()


        self.clean_text()

        self.convert_types()

        self.handle_missing_values()

        self.remove_duplicates()

        self.validate_data()

        return self.df



if __name__ == "__main__":

    cleaner = DataCleaner(
        "data/dataset_delivery_time.csv"
    )

    df_clean = cleaner.clean()

    df_clean.to_csv(
        "data/dataset_delivery_time_clean.csv",
        index=False
    )

