# DeliveryETA — Modélisation Prédictive des Temps de Livraison

## 1. Présentation du projet

**DeliveryETA** est un projet de Machine Learning visant à prédire le temps total nécessaire à la livraison d'une commande, depuis sa préparation jusqu'à sa réception par le client.

L'objectif est de développer un modèle de régression capable d'estimer la durée d'une livraison en minutes à partir des caractéristiques de la commande, du livreur, des conditions météorologiques et du trafic.

Cette solution permet d'aider les entreprises de livraison à améliorer leurs estimations de temps d'arrivée (*Estimated Time of Arrival — ETA*), à optimiser la planification des livraisons et à mieux comprendre les facteurs qui influencent les délais.

### Objectifs métier

- Prédire le temps de livraison en minutes.
- Fournir une estimation du délai au client.
- Identifier les facteurs qui influencent la durée des livraisons.
- Faciliter la planification des tournées et l'affectation des livreurs.
- Comparer et optimiser plusieurs modèles de régression.
- Mettre à disposition une application web interactive pour exploiter les prédictions.

---

## 2. Technologies utilisées

| Technologie | Utilisation |
|---|---|
| Python | Développement du projet |
| Pandas & NumPy | Nettoyage et manipulation des données |
| Matplotlib & Seaborn | Visualisation et analyse exploratoire |
| Scikit-learn | Prétraitement et modélisation |
| HistGradientBoostingRegressor | Modèle de régression retenu |
| RandomForestRegressor | Modèle de comparaison et d'optimisation |
| Pipeline | Regroupement du prétraitement et du modèle |
| Joblib | Sauvegarde et chargement du modèle |
| Streamlit | Développement de l'application web |
| MLflow | Suivi et comparaison des expériences |

Le projet peut également intégrer FastAPI et Apache Airflow pour l'exposition du modèle sous forme d'API et l'automatisation des entraînements.

---

## 3. Dataset

Le projet utilise le fichier `Food delivery.csv`, qui contient un historique de livraisons avec des informations relatives aux commandes, aux livreurs, aux conditions de livraison et aux temps réellement observés.

### Principales catégories de variables

- **Informations sur le livreur :** âge, expérience et note.
- **Informations sur la commande :** type de commande et caractéristiques associées.
- **Conditions de livraison :** météo, trafic et type de véhicule.
- **Localisation :** coordonnées GPS du restaurant et du client.
- **Variable cible :** `Time_taken(min)`.

La variable `Time_taken(min)` représente le temps réel de livraison en minutes. Elle constitue la variable cible que le modèle cherche à prédire.

### Source des données

Fichier utilisé : `data/raw/Food delivery.csv`

---

## 4. Architecture du projet

```text
DeliveryETA/
│
├── data/
│   ├── raw/
│   │   └── Food delivery.csv
│   └── processed/
│
├── notebooks/
│   └── delivery_eta_analysis.ipynb
│
├── src/
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── train.py
│   └── evaluation.py
│
├── models/
│   └── model_final.joblib
│
├── app/
│   └── streamlit_app.py
│
├── artifacts/
│   ├── metrics.json
│   ├── model_comparison_step3.csv
│   └── test_predictions.csv
│
├── requirements.txt
└── README.md
```

*Cette structure est indicative et doit être adaptée à l'organisation réelle du dépôt Git.*

---

## 5. Démarche de Machine Learning

### Étape 1 — Chargement et nettoyage des données

L'objectif est de transformer les données brutes en un jeu de données exploitable.

Les opérations comprennent :

- Chargement du fichier CSV avec Pandas.
- Analyse des dimensions, des types de données et des premières observations.
- Nettoyage des valeurs textuelles parasites.
- Conversion des colonnes vers les types appropriés.
- Détection et traitement des valeurs manquantes.
- Vérification des doublons.
- Détection des valeurs aberrantes, notamment les coordonnées GPS invalides et les durées irréalistes.

Chaque décision de nettoyage doit être justifiée afin de préserver la qualité des données et d'éviter la suppression injustifiée d'informations utiles.

### Étape 2 — Analyse exploratoire des données (EDA)

L'analyse exploratoire permet de comprendre la distribution des variables et leur relation avec le temps de livraison.

Les principales opérations sont :

- Calcul des statistiques descriptives.
- Analyse de la distribution de `Time_taken(min)`.
- Étude des relations entre les variables explicatives et la variable cible.
- Analyse des corrélations entre les variables numériques.
- Comparaison des temps de livraison selon la météo, le trafic et le type de véhicule.
- Identification des facteurs susceptibles d'influencer la durée de livraison.

Les résultats de cette analyse servent à orienter les choix de prétraitement et de modélisation.

### Étape 3 — Feature Engineering

Le *feature engineering* consiste à créer de nouvelles variables pertinentes à partir des données existantes.

Les variables dérivées comprennent notamment :

- `Distance_km` : distance estimée entre le restaurant et le client à partir des coordonnées géographiques.
- `Preparation_time_min` : durée estimée de préparation de la commande.
- `Order_hour` : heure de la commande.
- `Day_of_week` : jour de la semaine.
- `Is_weekend` : indicateur permettant d'identifier les commandes passées le week-end.

Ces variables permettent de représenter des facteurs opérationnels susceptibles d'influencer le temps de livraison.

La pertinence des nouvelles variables doit être évaluée à partir des performances du modèle et de leur disponibilité au moment de la prédiction.

### Étape 4 — Prétraitement et entraînement des modèles

Le jeu de données est divisé en deux ensembles :

- **80 %** pour l'entraînement.
- **20 %** pour le test.

Le prétraitement comprend l'encodage des variables catégorielles et la mise à l'échelle des variables numériques lorsque cela est pertinent pour le modèle utilisé.

Plusieurs modèles de régression sont comparés sur les mêmes ensembles d'entraînement et de test, notamment :

- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor
- HistGradientBoostingRegressor

Les performances sont évaluées à l'aide de métriques adaptées à un problème de régression.

L'utilisation d'un `Pipeline` scikit-learn permet de regrouper les transformations et le modèle afin de garantir la cohérence du traitement entre l'entraînement et l'inférence.

### Étape 5 — Optimisation et sélection du modèle final

Après la comparaison initiale, les hyperparamètres du modèle le plus prometteur sont optimisés avec une méthode telle que `RandomizedSearchCV` ou `GridSearchCV`.

Les critères d'évaluation comprennent :

- **MAE (Mean Absolute Error)** : erreur absolue moyenne, exprimée en minutes.
- **MSE (Mean Squared Error)** : moyenne des erreurs quadratiques.
- **RMSE (Root Mean Squared Error)** : racine de l'erreur quadratique moyenne, exprimée en minutes.
- **R²** : proportion de la variabilité de la cible expliquée par le modèle.
- **R² ajusté** : indicateur complémentaire tenant compte du nombre de variables explicatives.

Les résultats obtenus sur les ensembles d'entraînement et de test sont comparés afin d'évaluer la capacité de généralisation et de détecter un éventuel surapprentissage.

Le modèle final et les éléments nécessaires à l'inférence sont sauvegardés avec Joblib.

### Étape 6 — Application web avec Streamlit

L'application Streamlit rend le modèle accessible aux utilisateurs non techniques.

Elle permet notamment de :

- Saisir les caractéristiques d'une livraison.
- Renseigner les informations disponibles sur le livreur, le trafic et la météo.
- Fournir les informations nécessaires au calcul de la distance et des autres variables dérivées.
- Obtenir une estimation du temps de livraison en minutes.
- Consulter les visualisations issues de l'analyse exploratoire.
- Afficher les métriques et les performances du modèle.

L'interface doit appliquer les mêmes transformations que celles utilisées pendant l'entraînement, afin de garantir la cohérence des prédictions.

### Étape 7 — Suivi des expériences avec MLflow

MLflow permet de suivre et de comparer les expériences de Machine Learning.

Les informations enregistrées peuvent comprendre :

- Les paramètres des modèles.
- Les hyperparamètres testés.
- Les métriques de performance.
- Les résultats de comparaison des modèles.
- Les artefacts produits pendant l'entraînement.
- Les fichiers du modèle final et les résultats de prédiction.

Cette traçabilité facilite la comparaison des expériences et la reproduction des résultats.

---

## 6. Résultats et performances

Les résultats ci-dessous correspondent aux performances obtenues lors des expérimentations du projet. Les valeurs finales doivent être vérifiées à partir des artefacts générés par la dernière exécution.

### Modèles comparés

Les modèles de régression sont évalués sur le même ensemble de test afin de comparer leurs capacités de prédiction.

### Performances des modèles retenus

| Modèle | MAE (min) | RMSE (min) | R² |
|---|---:|---:|---:|
| HistGradientBoostingRegressor | 3,339 | 4,144 | 0,797 |
| Random Forest optimisé | 3,107 | À renseigner | 0,8181 |

Le modèle Random Forest optimisé présente une MAE plus faible et un meilleur coefficient R² dans les résultats enregistrés de cette expérimentation. La RMSE finale doit être récupérée dans les résultats correspondants avant de compléter la comparaison.

### Interprétation métier

Pour le modèle HistGradientBoostingRegressor, la MAE de **3,339 minutes** signifie que la prédiction s'écarte en moyenne d'environ 3,34 minutes du temps réel de livraison sur l'ensemble de test.

Pour le Random Forest optimisé, la MAE de **3,107 minutes** correspond à une erreur absolue moyenne d'environ 3,11 minutes.

Ces erreurs moyennes constituent des indicateurs utiles pour évaluer la précision des estimations de livraison. Elles ne garantissent toutefois pas que chaque prédiction se situera dans cet intervalle.

Le coefficient R² du Random Forest optimisé est d'environ **0,8181**, ce qui indique que le modèle explique environ 81,81 % de la variabilité du temps de livraison sur l'ensemble de test, dans les conditions de cette évaluation.

### Artefacts générés

Les principaux fichiers de résultats comprennent :

- `model_final.joblib` : modèle final sauvegardé.
- `metrics.json` : métriques de performance.
- `model_comparison_step3.csv` : comparaison des modèles.
- `test_predictions.csv` : prédictions réalisées sur l'ensemble de test.

---

## 7. Installation et exécution

### Prérequis

- Python et pip.
- Git.
- Un environnement compatible avec les dépendances du fichier `requirements.txt`.

### Étape 1 — Cloner le dépôt

```bash
git clone <URL_DU_DEPOT_GITHUB>
cd DeliveryETA
```

Remplacez `<URL_DU_DEPOT_GITHUB>` par l'adresse réelle de votre dépôt.

### Étape 2 — Créer un environnement virtuel

Sous Windows :

```bash
python -m venv .venv
.venv\Scripts\activate
```

Sous Linux :

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Étape 3 — Installer les dépendances

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Étape 4 — Lancer l'application Streamlit

Si le fichier d'entrée est `app/streamlit_app.py` :

```bash
python -m streamlit run app/streamlit_app.py
```

Si le fichier est situé à un autre emplacement, adaptez la commande à la structure réelle du projet.

L'application est généralement accessible à l'adresse :

http://localhost:8501

### Étape 5 — Suivre les expériences avec MLflow

Si MLflow est configuré avec un serveur local sur le port 5000 :

```bash
python -m mlflow server --host 127.0.0.1 --port 5000 --workers 1
```

Ouvrez ensuite :

http://127.0.0.1:5000

L'URI de tracking configurée dans le code d'entraînement doit correspondre à l'adresse du serveur utilisé.

---

## 8. Captures d'écran

Ajoutez des captures d'écran réelles de l'application Streamlit et des résultats du projet.

### Interface de prédiction

![Interface de prédiction](docs/screenshots/streamlit-prediction.png)

### Analyse exploratoire

![Analyse exploratoire](docs/screenshots/eda.png)

### Comparaison des modèles

![Comparaison des modèles](docs/screenshots/model-comparison.png)

### Suivi des expériences MLflow

![Suivi MLflow](docs/screenshots/mlflow-tracking.png)

Les chemins des images sont des exemples. Créez le dossier `docs/screenshots/` et placez-y les captures correspondantes pour qu'elles s'affichent sur GitHub.

---

## 9. Limites et perspectives d'amélioration

### Limites

- La qualité des prédictions dépend de la qualité et de la représentativité des données historiques.
- Les conditions réelles de circulation et les imprévus peuvent influencer les temps de livraison.
- Les performances observées sur l'ensemble de test ne garantissent pas les mêmes résultats en production.
- Certaines variables calculées doivent être disponibles au moment de la prédiction.
- Les résultats doivent être réévalués régulièrement pour détecter une éventuelle dérive des données.

### Perspectives d'amélioration

- Enrichir le dataset avec des données de circulation plus précises.
- Améliorer les variables géographiques et temporelles.
- Ajouter une API FastAPI pour exposer le modèle.
- Automatiser le réentraînement avec Apache Airflow.
- Renforcer les tests unitaires et les tests d'intégration.
- Mettre en place une pipeline CI/CD.
- Surveiller les performances du modèle après son déploiement.

---

## 10. Auteur et contexte pédagogique

**Projet :** DeliveryETA — Modélisation Prédictive des Temps de Livraison  
**Formateur / référent :** Hamid OUFAKIR  
**Date de création du brief :** 17/09/2026  
**Période de réalisation prévue :** du 21/09/2026 au 25/09/2026  
**Formation / référentiel :** Certification RNCP Développeur·se en Intelligence Artificielle  
**Type de travail :** Individuel

**Dépôt GitHub :** À compléter  
**Lien Jira :** À compléter

---

## Conclusion

DeliveryETA met en œuvre un processus complet de Machine Learning pour estimer les temps de livraison à partir de données historiques. De la préparation des données au feature engineering, en passant par la comparaison et l'optimisation des modèles, le projet aboutit à une application Streamlit permettant de consulter les prédictions et les performances du modèle.

L'intégration de MLflow apporte une meilleure traçabilité des expériences et facilite l'évolution du modèle. L'ensemble constitue une base pour une solution d'estimation des délais de livraison destinée à accompagner la prise de décision opérationnelle.
