import pandas as pd
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from src.utils import get_logger

logger = get_logger(__name__)

def run_rfe(X_train, y_train, categorical, numeric):

    logger.info("Running RFE with One-Hot Encoding + Imputation")

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

    estimator = LogisticRegression(max_iter=2000)

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("rfe", RFE(estimator, n_features_to_select=10))
    ])

    pipeline.fit(X_train, y_train)

    # Extract feature names after encoding
    encoded_features = pipeline.named_steps["preprocessor"].get_feature_names_out()
    rankings = pipeline.named_steps["rfe"].ranking_

    df_rankings = pd.DataFrame({
        "Feature": encoded_features,
        "Ranking": rankings
    }).sort_values("Ranking")

    logger.info("RFE complete")
    return df_rankings