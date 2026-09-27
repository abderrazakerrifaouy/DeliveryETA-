from pathlib import Path

import joblib
import pandas as pd
import streamlit as st



ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = ROOT / "models" / "model_final.joblib"
METRICS_PATH = ROOT / "models" / "metrics.json"
PREDICTIONS_PATH = ROOT / "models" / "test_predictions.csv"


st.set_page_config(
    page_title="DeliveryETA",
    layout="wide",
)



@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_metrics():
    import json

    with open(METRICS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def load_predictions():
    return pd.read_csv(PREDICTIONS_PATH)



missing_files = []

if not MODEL_PATH.exists():
    missing_files.append(str(MODEL_PATH))

if not METRICS_PATH.exists():
    missing_files.append(str(METRICS_PATH))

if not PREDICTIONS_PATH.exists():
    missing_files.append(str(PREDICTIONS_PATH))


if missing_files:
    st.error("Certains fichiers nécessaires sont introuvables :")

    for file in missing_files:
        st.code(file)

    st.info(
        "Vérifie que les fichiers sont bien présents dans le dossier models/."
    )

    st.stop()



model = load_model()
metrics = load_metrics()
test_predictions = load_predictions()



st.title("DeliveryETA")





st.sidebar.header("Informations de livraison")


distance_km = st.sidebar.number_input(
    "Distance (km)",
    min_value=0.0,
    max_value=100.0,
    value=5.0,
    step=0.5,
)



weather = st.sidebar.selectbox(
    "Météo",
    [
        "Sunny",
        "Stormy",
        "Sandstorms",
        "Windy",
        "Cloudy",
        "Fog",
    ],
)



traffic = st.sidebar.selectbox(
    "Trafic",
    [
        "Low",
        "Medium",
        "High",
        "Jam",
    ],
)




festival = st.sidebar.selectbox(
    "Festival",
    [
        "No",
        "Yes",
    ],
)



age = st.sidebar.number_input(
    "Âge du livreur",
    min_value=18.0,
    max_value=70.0,
    value=30.0,
    step=1.0,
)



vehicle_condition = st.sidebar.slider(
    "État du véhicule",
    min_value=0,
    max_value=3,
    value=2,
)




vehicle_type = st.sidebar.selectbox(
    "Type de véhicule",
    [
        "motorcycle",
        "scooter",
        "electric_scooter",
        "bicycle",
    ],
)



multiple_deliveries = st.sidebar.number_input(
    "Livraisons multiples",
    min_value=0.0,
    max_value=3.0,
    value=0.0,
    step=1.0,
)




city = st.sidebar.selectbox(
    "Ville",
    [
        "Urban",
        "Metropolitian",
        "Semi-Urban",
    ],
)



prep_time = st.sidebar.number_input(
    "Temps de préparation (min)",
    min_value=0.0,
    max_value=120.0,
    value=15.0,
    step=1.0,
)




order_hour = st.sidebar.slider(
    "Heure de commande",
    min_value=0,
    max_value=23,
    value=13,
)




order_date = st.sidebar.date_input(
    "Date de commande"
)



type_of_order = st.sidebar.selectbox(
    "Type de commande",
    [
        "Drinks",
        "Buffet",
        "Meal",
        "Snack",
    ],
)




day_of_week = order_date.weekday()

is_weekend = 1 if day_of_week >= 5 else 0



st.subheader("Données de la livraison")

input_data = {
    "Distance_km": distance_km,
    "Weatherconditions": weather,
    "Road_traffic_density": traffic,
    "Festival": festival,
    "Delivery_person_Age": age,
    "Vehicle_condition": vehicle_condition,
    "Type_of_vehicle": vehicle_type,
    "multiple_deliveries": multiple_deliveries,
    "City": city,
    "Preparation_time_min": prep_time,
    "Order_hour": order_hour,
    "Day_of_week": day_of_week,
    "Is_weekend": is_weekend,
    "Type_of_order": type_of_order,
}


input_df = pd.DataFrame([input_data])


st.dataframe(
    input_df,
    use_container_width=True,
    hide_index=True,
)



st.subheader("Prédiction")


if st.button(
    "Prédire le temps de livraison",
    use_container_width=True,
):

    try:

        prediction = model.predict(input_df)

        predicted_time = float(prediction[0])

        predicted_time = max(0.0, predicted_time)



        st.success(
            f"Temps de livraison estimé : "
            f"**{predicted_time:.1f} minutes**"
        )


        hours = int(predicted_time // 60)

        minutes = int(round(predicted_time % 60))


        if hours > 0:

            st.info(
                f"Environ **{hours} h {minutes} min**"
            )

        else:

            st.info(
                f"Environ **{minutes} minutes**"
            )


    except Exception as e:

        st.error(
            "Une erreur est survenue pendant la prédiction."
        )

        st.exception(e)

st.markdown("---")


tab_data, tab_metrics, tab_accuracy = st.tabs(
    [
        "Données",
        "Métriques",
        "Précision",
    ]
)




with tab_data:

    st.subheader("Variables utilisées")


    features = [
        "Distance_km",
        "Preparation_time_min",
        "Order_hour",
        "Day_of_week",
        "Is_weekend",
        "multiple_deliveries",
        "Vehicle_condition",
        "Delivery_person_Age",
        "Road_traffic_density",
        "Weatherconditions",
        "Type_of_vehicle",
        "Festival",
        "City",
        "Type_of_order",
    ]

    feature_df = pd.DataFrame(
        {
            "Feature": features
        }
    )

    st.dataframe(
        feature_df,
        use_container_width=True,
        hide_index=True,
    )



with tab_metrics:

    st.subheader(
        "Performance du modèle HistGradientBoosting"
    )


    col1, col2, col3, col4 = st.columns(4)


    with col1:

        st.metric(
            "MAE",
            f"{metrics['MAE']:.2f} min"
        )


    with col2:

        st.metric(
            "RMSE",
            f"{metrics['RMSE']:.2f} min"
        )


    with col3:

        st.metric(
            "R² test",
            f"{metrics['R2_test']:.3f}"
        )


    with col4:

        st.metric(
            "R² ajusté",
            f"{metrics['R2_ajuste_test']:.3f}"
        )


    r2_train = metrics["R2_train"]

    r2_test = metrics["R2_test"]

    gap = r2_train - r2_test


    st.write(
        f"**R² train :** {r2_train:.3f}"
    )

    st.write(
        f"**R² test :** {r2_test:.3f}"
    )

    st.write(
        f"**Écart train/test :** {gap:.3f}"
    )


    if gap < 0.05:

        st.success(
            "Le modèle présente un faible écart entre "
            "les performances train et test."
        )

    elif gap < 0.10:

        st.warning(
            "Il existe un léger écart entre train et test."
        )

    else:

        st.warning(
            "L'écart train/test est relativement important."
        )


    if "best_params" in metrics:

        st.markdown("---")

        with st.expander(
            "Voir les meilleurs hyperparamètres"
        ):

            st.json(
                metrics["best_params"]
            )



with tab_accuracy:

    st.subheader(
        "Analyse des prédictions"
    )




    actual_col = None
    predicted_col = None


    possible_actual = [
        "actual",
        "Actual",
        "y_test",
        "Time_taken(min)",
        "target",
        "true",
    ]


    possible_predicted = [
        "prediction",
        "Prediction",
        "y_pred",
        "predicted",
        "Predicted",
    ]


    for col in possible_actual:

        if col in test_predictions.columns:

            actual_col = col
            break


    for col in possible_predicted:

        if col in test_predictions.columns:

            predicted_col = col
            break



    if actual_col and predicted_col:

        result_df = test_predictions[
            [actual_col, predicted_col]
        ].copy()


        result_df["Erreur"] = (
            result_df[predicted_col]
            - result_df[actual_col]
        )


        result_df["Erreur_absolue"] = (
            result_df["Erreur"]
            .abs()
        )


        st.dataframe(
            result_df.head(20),
            use_container_width=True,
            hide_index=True,
        )


        st.markdown("---")


        average_error = (
            result_df["Erreur_absolue"]
            .mean()
        )


        st.metric(
            "Erreur absolue moyenne",
            f"{average_error:.2f} min"
        )


    else:

        st.warning(
            "Les colonnes de vraie valeur et de prédiction "
            "n'ont pas été détectées automatiquement."
        )


        st.write(
            "Colonnes disponibles dans test_predictions.csv :"
        )


        st.write(
            list(test_predictions.columns)
        )




st.caption(
    "DeliveryETA — Projet Machine Learning | "
    "HistGradientBoostingRegressor"
)