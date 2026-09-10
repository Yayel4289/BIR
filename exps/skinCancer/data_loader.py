from typing import Tuple
import pandas as pd
from pandas.api.types import CategoricalDtype
import torch
from torch.utils.data import DataLoader
import numpy as np

from src.data.data_loader_set import DataLoaderSet
from src.data.data_loader import get_source_target_train_val_test_loaders_from_tensor, tensor_train_val_test_split

LABELS_TYPE = CategoricalDtype(categories=["akiec", "bcc", "bkl", "df", "nv", "vasc", "mel"], ordered=False)
SOURCE_TARGET_AGE_SPLIT = 30
DATA_PATH = "exps/skinCancer/data/"

MAX_COLOR = 255
MAX_AGE = 85

def fuse_hmnist_HAMmeta() -> None:
    hmnist_data = pd.read_csv(DATA_PATH + "hmnist_28_28_RGB.csv")
    metadata = pd.read_csv(DATA_PATH + "HAM10000_metadata.csv")

    if len(hmnist_data) != len(metadata):
        raise ValueError(f"hmnist_data and metadata must be the same length ({len(hmnist_data)} and {len(metadata)})")

    hmnist_metadata = hmnist_data.copy()
    #hmnist_metadata["lesion_id"] = metadata["lesion_id"]
    #hmnist_metadata["image_id"] = metadata["image_id"]
    hmnist_metadata["age"] = metadata["age"]
    #hmnist_metadata["sex"] = metadata["sex"]
    #hmnist_metadata["localization"] = metadata["localization"]
    #hmnist_metadata.drop(columns=["label"]) # replaced by diagnosis
    #hmnist_metadata["diagnosis"] = metadata["dx"].astype(LABELS_TYPE)

    hmnist_metadata.to_csv(DATA_PATH + "hmnist_age_label.csv", index=False)


def normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop(columns=["age", "label"]).div(MAX_COLOR).join(df[["age", "label"]])
    df["age"] /= MAX_AGE
    return df

def split_source_target(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df_source = df[df["age"] >  SOURCE_TARGET_AGE_SPLIT / MAX_AGE]
    df_target = df[df["age"] <= SOURCE_TARGET_AGE_SPLIT / MAX_AGE]
    df_source["age"] = 1 - df_source["age"] # convention : close to 0
    df_target["age"] = 1 - df_target["age"] # convention : close to 1
    return df_source, df_target

def make_domain_discrete(df_source, df_target) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df_source["age"] = 0
    df_target["age"] = 1
    return df_source, df_target

def get_X_y_tensors_from_df(df: pd.DataFrame) -> Tuple[torch.Tensor, torch.Tensor]:
    X_data = df.drop(columns=["label", "age"]).values.astype("float32")
    n = X_data.shape[0]
    X_images = X_data.reshape(n, 28, 28, 3) # imgs rgb 28x28
    X = torch.from_numpy(X_images)
    X = X.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)
    y = torch.from_numpy(df[["label", "age"]].values.astype("float32")) # label, domain
    return X, y

def get_skin_cancer_data_loader_set(continuous: bool = True) -> DataLoaderSet:
    df = pd.read_csv(DATA_PATH + "hmnist_age_label.csv")
    df = normalize_df(df)
    df_source, df_target = split_source_target(df)
    if not continuous:
        df_source, df_target = make_domain_discrete(df_source, df_target)
    X_source_tensor, y_source_tensor = get_X_y_tensors_from_df(df_source)
    X_target_tensor, y_target_tensor = get_X_y_tensors_from_df(df_target)
    source_train_loader, target_train_loader, source_val_loader, target_val_loader, source_test_loader, target_test_loader = get_source_target_train_val_test_loaders_from_tensor(
        X_source_tensor, y_source_tensor,
        X_target_tensor, y_target_tensor
    )
    return DataLoaderSet(source_train_loader, target_train_loader, source_val_loader, target_val_loader, source_test_loader, target_test_loader)

