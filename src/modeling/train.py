import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import  ExtraTreesRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Lasso , LinearRegression , Ridge
from sklearn.metrics import mean_absolute_error , mean_squared_error , r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import  OneHotEncoder , OrdinalEncoder , StandardScaler
from sklearn.tree import DecisionTreeRegressor




TARGET = "Time_taken(min)"

NUMERIC_COLS = ["Distance_km","Preparation_time_min","Order_hour","Day_of_week","Is_weekend","multiple_deliveries","Vehicle_condition","Delivery_person_Age" ]

ORDINAL_COLS = [ "Road_traffic_density" ]

NOMINAL_COLS = ["Weatherconditions","Type_of_vehicle","Festival","City" ,"Type_of_order" ]

TRAFFIC_ORDER = [ ["Low", "Medium", "High", "Jam"] ]



def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            (
                "num",StandardScaler(),NUMERIC_COLS,
            ),
            ( 
                "ord", OrdinalEncoder( categories=TRAFFIC_ORDER, handle_unknown="use_encoded_value", unknown_value=-1, ), ORDINAL_COLS,
            ),
            (
                "nom", OneHotEncoder( handle_unknown="ignore", drop="if_binary", sparse_output=False, ), NOMINAL_COLS,
            ),
        ]
    )




def get_models() -> dict:

    return {
        "LinearRegression": LinearRegression(),
        "Ridge": Ridge( alpha=1.0),
        "Lasso": Lasso( alpha=0.01, max_iter=10000, random_state=42),
        "DecisionTree": DecisionTreeRegressor( random_state=42 ),
        "RandomForest": RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1 ),
        "ExtraTrees": ExtraTreesRegressor(n_estimators=200, random_state=42, n_jobs=-1  ),
        "GradientBoosting": GradientBoostingRegressor(random_state=42 ),
        "HistGradientBoosting": HistGradientBoostingRegressor( random_state=42  ),
    }



def split_data(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42 ):

    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    return train_test_split( X, y, test_size=test_size, random_state=random_state, )




def evaluate( y_true, y_pred ) -> dict:
    
    mse = mean_squared_error(  y_true , y_pred )

    return {
        "MAE": mean_absolute_error(y_true, y_pred ),
        "MSE": mse,
        "RMSE": np.sqrt(mse),
        "R2": r2_score(y_true, y_pred ),
    }


def train_and_compare(df: pd.DataFrame ) -> tuple[pd.DataFrame, dict]:

    X_train, X_test, y_train, y_test = split_data(df)

    models = get_models()

    rows = []

    fitted_pipelines = {}

    for name, model in models.items():

        print(f"\nEntraînement : {name}...")

        preprocessor = build_preprocessor()

        pipe = Pipeline( steps=[ ("prep", preprocessor), ("model", model) ] )

        pipe.fit( X_train, y_train )

        fitted_pipelines[name] = pipe

        y_pred_test = pipe.predict( X_test)

        test_metrics = evaluate( y_test, y_pred_test )

        y_pred_train = pipe.predict( X_train  )

        train_metrics = evaluate(  y_train, y_pred_train )

        rows.append(
            {
                "model": name,

                "MAE": test_metrics["MAE"],

                "MSE": test_metrics["MSE"],

                "RMSE": test_metrics["RMSE"],

                "R2_test": test_metrics["R2"],

                "R2_train": train_metrics["R2"],
            }
        )

    results = pd.DataFrame( rows )

    results = results.sort_values( by="MAE", ascending=True  ).reset_index(drop=True)

    return results, fitted_pipelines


def run_training_pipeline(input_path: str, output_path: str) -> None:

    df = pd.read_csv(input_path)

    results, fitted_pipelines = train_and_compare(df)

    best_model_name = results.loc[0, "model"]

    best_pipeline = fitted_pipelines[best_model_name]

    best_pipeline.fit(df.drop(columns=[TARGET]), df[TARGET])

    pd.DataFrame(results).to_csv(output_path, index=False)

    print(f"\nMeilleur modèle : {best_model_name}")

if __name__ == "__main__":
    input_path = "data/processed/dataset_delivery_time_model_ready.csv"
    output_path = "data/processed/model_comparison_results.csv"
    run_training_pipeline(input_path, output_path)
