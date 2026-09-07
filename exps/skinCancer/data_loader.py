import pandas as pd
from pandas.api.types import CategoricalDtype

def fuse_hmnist_HAMmeta():
    hmnist_data = pd.read_csv("data/hmnist_28_28_RGB.csv")
    metadata = pd.read_csv("data/HAM10000_metadata.csv")

    if len(hmnist_data) != len(metadata):
        raise ValueError(f"hmnist_data and metadata must be the same length ({len(hmnist_data)} and {len(metadata)})")

    labels_type = CategoricalDtype(categories=["akiec", "bcc", "bkl", "df", "nv", "vasc", "mel"], ordered=False)
    hmnist_metadata = hmnist_data.copy()
    hmnist_metadata["lesion_id"] = metadata["lesion_id"]
    hmnist_metadata["image_id"] = metadata["image_id"]
    hmnist_metadata["age"] = metadata["age"]
    hmnist_metadata["sex"] = metadata["sex"]
    hmnist_metadata["localization"] = metadata["localization"]
    hmnist_metadata.drop(columns=["label"]) # replaced by diagnosis
    hmnist_metadata["diagnosis"] = metadata["dx"].astype(labels_type)

    hmnist_metadata.to_csv("hmnist_metadata.csv", index=False)

