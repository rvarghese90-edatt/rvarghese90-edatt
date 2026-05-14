import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from config.settings import OUTPUT_DIR
from src.utils import get_logger

logger = get_logger(__name__)

def train_models(df, categorical, numeric):

    y = df["target"]
    X = df.drop(columns=["target"])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=42
    )

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric),
            ("cat", categorical_transformer, categorical)
        ]
    )

    models = {
        "logistic_regression": LogisticRegression(max_iter=2000),
        "xgboost": XGBClassifier(eval_metric="logloss"),
        "random_forest": RandomForestClassifier(
            n_estimators=50, criterion="entropy", random_state=7
        )
    }

    results = {}

    for name, model in models.items():
        logger.info(f"Training model: {name}")

        pipeline = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("model", model)
        ])

        pipeline.fit(X_train, y_train)

        train_preds = pipeline.predict_proba(X_train)[:, 1]
        test_preds = pipeline.predict_proba(X_test)[:, 1]

        results[name] = {
            "train_auc": roc_auc_score(y_train, train_preds),
            "test_auc": roc_auc_score(y_test, test_preds)
        }

        model_path = os.path.join(OUTPUT_DIR, f"{name}.pkl")
        joblib.dump(pipeline, model_path)
        logger.info(f"Saved model: {model_path}")

    return results, X_train, y_train