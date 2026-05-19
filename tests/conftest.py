import os.path
import shutil
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import fetch_california_housing, make_classification

_PATH_TEST = os.path.dirname(__file__)
PATH_DATASETS = os.path.join(_PATH_TEST, ".datasets")
os.makedirs(PATH_DATASETS, exist_ok=True)
PATH_TMP = Path(_PATH_TEST) / ".tmp"
PATH_TMP.mkdir(exist_ok=True)

DATASET_ZIP_OCCUPANCY = os.path.join(PATH_DATASETS, "occupancy_data.zip")
if not os.path.isfile(DATASET_ZIP_OCCUPANCY):
    import urllib.request

    urllib.request.urlretrieve(
        "http://archive.ics.uci.edu/ml/machine-learning-databases/00357/occupancy_data.zip", DATASET_ZIP_OCCUPANCY
    )


def load_regression_data():
    dataset = fetch_california_housing(data_home=PATH_DATASETS, as_frame=True)
    df = dataset.frame.sample(5000)
    df["HouseAgeBin"] = pd.qcut(df["HouseAge"], q=4)
    df["HouseAgeBin"] = "age_" + df.HouseAgeBin.cat.codes.astype(str)
    test_idx = df.sample(int(0.2 * len(df)), random_state=42).index
    test = df[df.index.isin(test_idx)]
    train = df[~df.index.isin(test_idx)]
    return (train, test, dataset.target_names)


def load_classification_data():
    features, target = make_classification(
        n_samples=5000,
        n_features=54,
        n_informative=24,
        n_redundant=10,
        n_repeated=0,
        n_classes=7,
        n_clusters_per_class=1,
        class_sep=1.25,
        random_state=42,
    )
    data = pd.DataFrame(features, columns=[f"feature_{i}" for i in range(features.shape[-1])])
    data["feature_53"] = (data["feature_53"] > data["feature_53"].median()).astype(np.int64)
    data["target"] = target.astype(np.int64)
    data["feature_0_cat"] = pd.qcut(data["feature_0"], q=4)
    data["feature_0_cat"] = "feature_0_" + data.feature_0_cat.cat.codes.astype(str)
    test_idx = data.sample(int(0.2 * len(data)), random_state=42).index
    test = data[data.index.isin(test_idx)]
    train = data[~data.index.isin(test_idx)]
    return (train, test, ["target"])


def load_timeseries_data():
    zipfile = ZipFile(DATASET_ZIP_OCCUPANCY)
    train = pd.read_csv(zipfile.open("datatraining.txt"), sep=",")
    val = pd.read_csv(zipfile.open("datatest.txt"), sep=",")
    test = pd.read_csv(zipfile.open("datatest2.txt"), sep=",")
    return (pd.concat([train, val], sort=False), test, ["Occupancy"])


@pytest.fixture(scope="session")
def regression_data():
    return load_regression_data()


@pytest.fixture(scope="session")
def classification_data():
    return load_classification_data()


@pytest.fixture(scope="session")
def timeseries_data():
    return load_timeseries_data()


class _WorkspaceTmpPathFactory:
    def __init__(self, base: Path):
        self.base = base
        self.base.mkdir(parents=True, exist_ok=True)
        self._counter = 0
        self._retention_policy = "all"
        self._retention_count = 0

    def mktemp(self, basename: str, numbered: bool = True) -> Path:
        if numbered:
            path = self.base / f"{basename}_{self._counter:03d}"
            self._counter += 1
        else:
            path = self.base / basename
        if path.exists():
            shutil.rmtree(path, ignore_errors=True)
        path.mkdir(parents=True, exist_ok=True)
        return path


@pytest.fixture(scope="session")
def tmp_path_factory():
    session_tmp = PATH_TMP / "pytest_session"
    session_tmp.mkdir(parents=True, exist_ok=True)
    return _WorkspaceTmpPathFactory(session_tmp)
