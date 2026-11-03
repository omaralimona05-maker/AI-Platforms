"""Train the engineered Titanic pipeline and save it for the API.

Run:  python train.py
"""
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_openml
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from features import engineer_features

SEED = 42
MODEL_PATH = "titanic_engineered_pipeline.joblib"

RAW_COLS = ["age", "fare", "sibsp", "parch", "pclass", "sex", "embarked", "name"]
NUMERIC = ["age", "fare", "sibsp", "parch", "pclass", "family_size", "fare_per_person"]
CATEGORICAL = ["sex", "embarked", "is_alone", "is_child", "title"]


def build_pipeline() -> Pipeline:
    preprocess = ColumnTransformer([
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]), NUMERIC),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), CATEGORICAL),
    ])
    return Pipeline([
        ("features", FunctionTransformer(engineer_features, validate=False)),
        ("preprocess", preprocess),
        ("model", LogisticRegression(max_iter=3000)),
    ])


def main():
    df = fetch_openml("titanic", version=1, as_frame=True).frame.copy()
    df["survived"] = df["survived"].astype(int)
    X, y = df[RAW_COLS], df["survived"]

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=SEED
    )
    model = build_pipeline().fit(X_tr, y_tr)
    auc = roc_auc_score(y_te, model.predict_proba(X_te)[:, 1])
    print(f"Held-out AUC: {auc:.4f}")

    joblib.dump(model, MODEL_PATH)
    print(f"Saved {MODEL_PATH}")


if __name__ == "__main__":
    main()
