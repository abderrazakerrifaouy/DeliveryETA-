import os
import json

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline

from src.modeling.train import build_preprocessor, split_data


PARAM_DIST = {
    "model__learning_rate": [0.03, 0.05, 0.08, 0.1, 0.15],
    "model__max_iter": [100, 150, 200, 300, 400],
    "model__max_leaf_nodes": [15, 20, 31, 40, 50],
    "model__max_depth": [None, 5, 7, 10, 15],
    "model__min_samples_leaf": [10, 15, 20, 30, 40],
    "model__l2_regularization": [0.0, 0.01, 0.1, 1.0, 5.0],
}


def adjusted_r2(r2: float, n: int, p: int) -> float:
    return 1 - (1 - r2) * (n - 1) / (n - p - 1)


def search_hyperparameters(
    X_train,
    y_train,
    search_sample_size: int = 8000,
    n_iter: int = 20,
    cv: int = 3,
    random_state: int = 42,
) -> dict:
    if len(X_train) > search_sample_size:
        X_search = X_train.sample(search_sample_size, random_state=random_state)
        y_search = y_train.loc[X_search.index]
    else:
        X_search = X_train
        y_search = y_train

    pipe = Pipeline([
        ("prep", build_preprocessor()),
        ("model", HistGradientBoostingRegressor(random_state=random_state)),
    ])

    search = RandomizedSearchCV(
        estimator=pipe,
        param_distributions=PARAM_DIST,
        n_iter=n_iter,
        cv=cv,
        scoring="neg_root_mean_squared_error",
        random_state=random_state,
        n_jobs=-1,
        verbose=1,
    )

    search.fit(X_search, y_search)

    return {
        key.replace("model__", ""): value
        for key, value in search.best_params_.items()
    }


def train_final_model(
    df: pd.DataFrame,
    best_params: dict | None = None,
) -> dict:
    X_train, X_test, y_train, y_test = split_data(df)

    if best_params is None:
        best_params = search_hyperparameters(X_train, y_train)

    pipe = Pipeline([
        ("prep", build_preprocessor()),
        ("model", HistGradientBoostingRegressor(
            random_state=42,
            **best_params
        )),
    ])

    pipe.fit(X_train, y_train)

    pred_test = pipe.predict(X_test)
    pred_train = pipe.predict(X_train)

    n_features = pipe.named_steps["prep"].transform(
        X_test.iloc[:5]
    ).shape[1]

    r2_test = r2_score(y_test, pred_test)

    metrics = {
        "best_params": best_params,
        "MAE": float(mean_absolute_error(y_test, pred_test)),
        "RMSE": float(np.sqrt(mean_squared_error(y_test, pred_test))),
        "R2_test": float(r2_test),
        "R2_ajuste_test": float(
            adjusted_r2(r2_test, len(y_test), n_features)
        ),
        "R2_train": float(r2_score(y_train, pred_train)),
        "n_features": int(n_features),
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
    }

    return {
        "pipeline": pipe,
        "metrics": metrics,
        "y_test": y_test,
        "pred_test": pred_test,
    }


def save_artifacts(
    pipe: Pipeline,
    metrics: dict,
    y_test,
    pred_test,
    out_dir: str = "models",
) -> None:
    os.makedirs(out_dir, exist_ok=True)

    joblib.dump(
        pipe,
        os.path.join(out_dir, "model_final.joblib")
    )

    try:
        scaler = pipe.named_steps["prep"].named_transformers_["num"]
        joblib.dump(
            scaler,
            os.path.join(out_dir, "scaler.joblib")
        )
    except KeyError:
        print("Attention : aucun transformateur 'num'.")

    with open(
        os.path.join(out_dir, "metrics.json"),
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            metrics,
            f,
            indent=2,
            ensure_ascii=False
        )

    predictions_df = pd.DataFrame({
        "y_test": y_test.reset_index(drop=True),
        "y_pred": pred_test,
    })

    predictions_df.to_csv(
        os.path.join(out_dir, "test_predictions.csv"),
        index=False
    )


def run_training_pipeline(
    input_path: str,
    output_dir: str = "models",
) -> dict:
    df = pd.read_csv(input_path)

    X_train, X_test, y_train, y_test = split_data(df)

    print("\n" + "=" * 60)
    print("OPTIMISATION HISTGRADIENTBOOSTING")
    print("=" * 60)

    best_params = search_hyperparameters(
        X_train,
        y_train,
        search_sample_size=8000,
        n_iter=20,
        cv=3,
        random_state=42,
    )

    print("\nMeilleurs hyperparamètres :")

    for key, value in best_params.items():
        print(f"  {key}: {value}")

    result = train_final_model(
        df,
        best_params=best_params
    )

    metrics = result["metrics"]

    print("\n" + "=" * 60)
    print("RÉSULTATS FINAUX")
    print("=" * 60)
    print(f"MAE            : {metrics['MAE']:.3f} min")
    print(f"RMSE           : {metrics['RMSE']:.3f} min")
    print(f"R2 test        : {metrics['R2_test']:.4f}")
    print(f"R2 ajusté test : {metrics['R2_ajuste_test']:.4f}")
    print(f"R2 train       : {metrics['R2_train']:.4f}")
    print(f"Features       : {metrics['n_features']}")
    print(f"Train          : {metrics['n_train']}")
    print(f"Test           : {metrics['n_test']}")

    save_artifacts(
        pipe=result["pipeline"],
        metrics=result["metrics"],
        y_test=result["y_test"],
        pred_test=result["pred_test"],
        out_dir=output_dir,
    )

    print("\n" + "=" * 60)
    print("SAUVEGARDE TERMINÉE")
    print("=" * 60)
    print(f"Dossier : {output_dir}/")
    print("- model_final.joblib")
    print("- scaler.joblib")
    print("- metrics.json")
    print("- test_predictions.csv")

    return result


if __name__ == "__main__":
    input_path = "data/processed/dataset_delivery_time_model_ready.csv"
    output_dir = "models"

    run_training_pipeline(
        input_path=input_path,
        output_dir=output_dir
    )