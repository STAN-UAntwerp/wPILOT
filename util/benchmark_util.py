from __future__ import annotations
import time
import os
import json
from dataclasses import dataclass
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import ParameterGrid
from sklearn.tree import DecisionTreeRegressor
import numpy as np

from pilot.pilot import PILOT
from pilot.copilot import coPILOT
from util.benchmark_info import PILOT_DATASETS_CAT_IDS, PMLB_DATASETS_CAT_IDS, DENSE_DATASETS_CAT_IDS, CS_DATASETS_CAT_IDS

@dataclass
class Data:
    name: str
    X: np.ndarray
    y: np.ndarray
    cat_ids: np.ndarray

    def subset(self, idx: list[int]) -> Data:
        return Data(
            self.name,
            self.X[idx, :].copy(),
            self.y[idx].copy(),
            self.cat_ids
        )

    def n_samples(self) -> int:
        return self.X.shape[0]

    def n_features(self) -> int:
        return self.X.shape[1]

def load_cs_data(dataset_name: str, directory: str = None) -> Data:
    if directory is None:
        base_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        directory = os.path.join(base_directory, "datasets_CS", dataset_name)

    try:
        X_and_Y = np.loadtxt(os.path.join(directory, dataset_name + ".data"), delimiter=",")
    except ValueError:
        X_and_Y = np.loadtxt(os.path.join(directory, dataset_name + ".data"), delimiter=None)
    X = X_and_Y[:,:-1]
    y = X_and_Y[:,-1]
    try:
        categorical_ids = CS_DATASETS_CAT_IDS[dataset_name]
    except:
        raise ValueError(f"Dataset name {dataset_name} not found in CS_DATASETS_CAT_IDS")

    return Data(name=dataset_name, X=X, y=y, cat_ids=categorical_ids)

def cs_split(w, X, random_state=None):
    rng = np.random.default_rng(random_state)

    X = np.asarray(X)
    w = np.asarray(w)
    n = X.shape[0]

    all_idx = np.arange(n)
    rng.shuffle(all_idx)
    pretrain_cutoff = int(0.9 * n)
    pretrain_index = all_idx[:pretrain_cutoff]
    test_index = all_idx[pretrain_cutoff:]
    X_pre = X[pretrain_index]

    x_bar = X_pre.mean(axis=0)
    z = (X_pre - x_bar) @ w

    sigma = np.std(z)
    if sigma == 0:
        raise ValueError("Projection has zero variance; choose a different w.")

    v = 4 * z / sigma - 3
    p = 1 / (1 + np.exp(-v))  # logistic

    s = rng.binomial(1, p)

    # map back to global indices
    train_index = pretrain_index[s == 1]
    w_ideal = (1 / p)[s == 1]

    return train_index, test_index, w_ideal

def load_dense_weight_data(dataset_name: str, directory: str = None) -> Data:
    if directory is None:
        base_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        directory = os.path.join(base_directory, "datasets_dense_weight_2", dataset_name)

    X = np.loadtxt(os.path.join(directory, dataset_name + "_X.csv"), delimiter=",", skiprows=1)
    y = np.loadtxt(os.path.join(directory, dataset_name + "_y.csv"), delimiter=",", skiprows=1)
    try:
        categorical_ids = DENSE_DATASETS_CAT_IDS[dataset_name]
    except:
        raise ValueError(f"Dataset name {dataset_name} not found in DENSE_DATASETS_CAT_IDS2")

    return Data(name=dataset_name, X=X, y=y, cat_ids=categorical_ids)

def load_pmlb_data(dataset_name: str, directory: str = None) -> Data:
    if directory is None:
        base_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        directory = os.path.join(base_directory, "datasets_PMLB", dataset_name)

    X = np.loadtxt(os.path.join(directory, dataset_name + "_X.csv"), delimiter=",")
    y = np.loadtxt(os.path.join(directory, dataset_name + "_y.csv"), delimiter=",")
    try:
        categorical_ids = PMLB_DATASETS_CAT_IDS[dataset_name]
    except:
        raise ValueError(f"Dataset name {dataset_name} not found in PMLB_DATASETS_CAT_IDS")

    return Data(name=dataset_name, X=X, y=y, cat_ids=categorical_ids)

def load_pilot_paper_data(dataset_name: str, directory: str = None) -> Data:
    if directory is None:
        base_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        directory = os.path.join(base_directory, "datasets_pilot", dataset_name)

    X = np.loadtxt(os.path.join(directory, dataset_name + "_X.csv"), delimiter=",")
    y = np.loadtxt(os.path.join(directory, dataset_name + "_y.csv"), delimiter=",")
    try:
        categorical_ids = PILOT_DATASETS_CAT_IDS[dataset_name]
    except:
        raise ValueError(f"Dataset name {dataset_name} not found in PILOT_DATASETS_CAT_IDS")

    return Data(name=dataset_name, X=X, y=y, cat_ids=categorical_ids)

@dataclass
class FitResult:
    r2: float
    mse: float
    fit_duration: float
    predict_duration: float

def fit_cart(train_dataset: Data, test_dataset: Data, max_depth=12, **parameters) -> FitResult:
    t1 = time.time()
    model = DecisionTreeRegressor(random_state=123, max_depth=max_depth, **parameters)
    model.fit(train_dataset.X, train_dataset.y)
    t2 = time.time()
    y_pred = model.predict(test_dataset.X)
    t3 = time.time()

    r2 = float(r2_score(test_dataset.y, y_pred))
    mse = float(mean_squared_error(test_dataset.y, y_pred))
    return FitResult(
        r2=r2, mse=mse, fit_duration=t2 - t1, predict_duration=t3 - t2
    )

def fit_pilot(train_dataset: Data, test_dataset: Data, **parameters) -> FitResult:
    t1 = time.time()
    model = PILOT(**parameters)
    model.fit(
        train_dataset.X,
        train_dataset.y,
        categorical=train_dataset.cat_ids
    )
    t2 = time.time()
    y_pred = model.predict(test_dataset.X)
    t3 = time.time()

    r2 = float(r2_score(test_dataset.y, y_pred))
    mse = float(mean_squared_error(test_dataset.y, y_pred))
    return FitResult(
        r2=r2, mse=mse, fit_duration=t2 - t1, predict_duration=t3 - t2
    )

def fit_copilot_avg(train_dataset: Data, test_dataset: Data, alpha = 1, **parameters) -> FitResult:
    t1 = time.time()
    model = coPILOT(alpha = alpha, **parameters)
    model.fit(
        train_dataset.X,
        train_dataset.y,
        categorical=train_dataset.cat_ids,
        stop_early=False
    )
    t2 = time.time()
    y_pred = model.predict(test_dataset.X, method="Average")
    t3 = time.time()

    r2 = float(r2_score(test_dataset.y, y_pred))
    mse = float(mean_squared_error(test_dataset.y, y_pred))
    return FitResult(
        r2=r2, mse=mse, fit_duration=t2 - t1, predict_duration=t3 - t2
    )

METHOD_FUNCTIONS = {
    "Cart": fit_cart,
    "Pilot": fit_pilot,
    "coPilot_avg": fit_copilot_avg,
    }

class BenchmarkSettings:
    methods_to_save: dict[tuple[str, str], set[str]]
    methods_to_run: dict[str, set[str]]
    # methods_to_run is a dictionary mapping each method to a set of parameters to run it with.

    def __init__(self):
        self.methods_to_save = {}
        self.methods_to_run = {}

    def get_methods(self):
        return self.methods_to_run.keys()

    def get_method_ids(self, method, grid_parameters):
        key = (method, grid_parameters)
        return self.methods_to_save[key]

    def get_parameters(self, method):
        if method not in self.get_methods():
            raise ValueError("Method not in methods to run.")
        return self.methods_to_run[method]

    def add_method(self, method, parameters=None):
        if parameters is None:
            parameters = {}

        method_id = method + str(parameters)

        if method not in self.methods_to_run:
            self.methods_to_run[method] = set()

        parameters = {
            key: value if isinstance(value, list) else [value]
            for key, value in parameters.items()
        }
        for grid_parameters in ParameterGrid(parameters):
            json_grid_parameters = json.dumps(grid_parameters)
            self.methods_to_run[method].add(json_grid_parameters)

            key = (method, json_grid_parameters)
            if key not in self.methods_to_save.keys():
                self.methods_to_save[key] = set()
            self.methods_to_save[key].add(method_id)
