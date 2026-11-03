"""Feature engineering shared by training and serving.

Kept in its own module so the pickled pipeline can always find
`features.engineer_features` when it is loaded by the API.
"""
import pandas as pd


def engineer_features(data: pd.DataFrame) -> pd.DataFrame:
    d = data.copy()

    d["family_size"] = d["sibsp"] + d["parch"] + 1
    d["is_alone"] = (d["family_size"] == 1).astype(str)
    d["fare_per_person"] = d["fare"] / d["family_size"].clip(lower=1)
    d["is_child"] = (d["age"] < 16).astype(str)

    d["title"] = (
        d["name"].str.extract(r",\s*([^.]*)\.", expand=False)
        .str.strip()
        .replace({"Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs"})
    )
    common = ["Mr", "Miss", "Mrs", "Master"]
    d.loc[~d["title"].isin(common), "title"] = "Rare"

    return d.drop(columns="name")
