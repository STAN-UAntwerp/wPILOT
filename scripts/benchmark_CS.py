print("importing PILOT...")
from pilot.pilot import PILOT

print("import done")

from util.benchmark_util import load_cs_data, load_pilot_paper_data, cs_split
import pandas as pd
import numpy as np
import os
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import r2_score, mean_squared_error
from datetime import datetime
from sklearn.svm import SVR
import glob
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

# %%
def run_benchmark(dataset_names: dict[str, str], methods: list[str], directory_result: str, verbose: bool = False, resume: bool = False):
    """
    Run benchmark on multiple datasets and save results incrementally.

    Parameters:
    -----------
    dataset_names : dict[str, str]
        Dict of dataset names to benchmark
    methods : list[str]
        List of methods to run
    directory_result : str, optional
        Directory to save results to
    verbose : bool, default=False
        Whether to print progress information
    resume : bool, default=False
        Whether to attempt to resume from existing results
    """
    if verbose: print(f"Started at time: {datetime.now().strftime('%H-%M-%S')}")

    # Create a unique ID for this benchmark run
    id_results = datetime.now().strftime('%d-%m-%y_%H-%M-%S')
    print(f"Id of the results: {id_results}")

    # Create the directory for results
    directory_folder = os.path.join(directory_result, f"Results_{id_results}")
    os.makedirs(directory_folder, exist_ok=True)

    # Path for raw results file that will be updated incrementally
    raw_results_path = os.path.join(directory_folder, "raw_results_" + id_results + ".csv")

    # Initialize results list
    results = []

    # Check if we should resume from existing results
    completed_datasets = set()
    if resume:
        resume_path = find_latest_results_file(directory_result)
        if resume_path:
            if verbose:
                print(f"Resuming from existing results: {resume_path}")
            existing_results = pd.read_csv(resume_path)
            results = existing_results.to_dict('records')
            completed_datasets = set(existing_results['Dataset'].unique())
            if verbose:
                print(f"Already completed datasets: {completed_datasets}")

    # Process each dataset
    for dataset_name in dataset_names:
        # Skip datasets that were already completed if resuming
        if resume and dataset_name in completed_datasets:
            if verbose:
                print(f"Skipping already completed dataset: {dataset_name}")
            continue

        if verbose:
            print(f"--- Working on dataset: {dataset_name} ---")

        dataset_results = []
        try:
            # Load the dataset
            if dataset_names[dataset_name] == "cs":
                dataset = load_cs_data(dataset_name)
            else:
                dataset = load_pilot_paper_data(dataset_name)
            (n, p) = (dataset.n_samples(), dataset.n_features())

            rng = np.random.default_rng(42)
            list_w = [rng.uniform(-1, 1, p) for _ in range(10)]
            for w_id, w in enumerate(list_w):
                if verbose:
                    print(f"-> W id = {w_id + 1}")
                for fold_number in range(10):
                    if verbose:
                        print(f"---> Fold number = {fold_number + 1}")

                    train_index, test_index, W_ideal = cs_split(w, dataset.X, random_state=fold_number)
                    train_data = dataset.subset(train_index)
                    test_data = dataset.subset(test_index)

                    # Compute density ratio weights with random forest classifier
                    X_class = np.vstack([train_data.X, test_data.X])
                    y_class = np.hstack([np.ones(len(train_data.X)), np.zeros(len(test_data.X))])
                    clf = LogisticRegression(random_state=42, max_iter=1000)
                    clf.fit(X_class, y_class)
                    prob = clf.predict_proba(train_data.X)[:, 1]
                    W = 1 - prob

                    try:
                        methods_predict = []

                        if "Pilot" in methods:
                            model_NW = PILOT()
                            model_NW.fit(train_data.X, train_data.y, categorical=train_data.cat_ids)
                            methods_predict.append(model_NW.predict(test_data.X))

                        if "PilotW" in methods:
                            model = PILOT()
                            model.fit(train_data.X, train_data.y, W, categorical=train_data.cat_ids)
                            methods_predict.append(model.predict(test_data.X))

                        if "PilotI" in methods:
                            model_ideal = PILOT()
                            model_ideal.fit(train_data.X, train_data.y, W_ideal, categorical=train_data.cat_ids)
                            methods_predict.append(model_ideal.predict(test_data.X))

                        if "Sklearn" in methods:
                            skl_model_no_weights = SVR(kernel='rbf', C=1.0, epsilon=0.1, gamma=2 / p)
                            skl_model_no_weights.fit(train_data.X, train_data.y)
                            methods_predict.append(skl_model_no_weights.predict(test_data.X))

                        if "SklearnW" in methods:
                            skl_model_weighted = SVR(kernel='rbf', C=1.0, epsilon=0.1, gamma=2 / p)
                            skl_model_weighted.fit(train_data.X, train_data.y, sample_weight=W)
                            methods_predict.append(skl_model_weighted.predict(test_data.X))

                        if "SklearnI" in methods:
                            skl_model_weighted = SVR(kernel='rbf', C=1.0, epsilon=0.1, gamma=2 / p)
                            skl_model_weighted.fit(train_data.X, train_data.y, sample_weight=W_ideal)
                            methods_predict.append(skl_model_weighted.predict(test_data.X))

                    except Exception as e:
                        # When a method fails, we stop processing the entire dataset
                        print(
                            f"Failed on dataset {dataset_name}, fold {fold_number + 1} with error: {e}")
                        print(f"Skipping the rest of dataset {dataset_name} and moving to next dataset")
                        raise  # Re-raise the exception to be caught by the outer try-except

                    # for y_pred, method in zip([y_pred_pilot_NW, y_pred_pilot, skl_pred_NW, skl_pred], ["Pilot", "PilotW", "Cart", "CartW"]):
                    for y_pred, method in zip(methods_predict, methods):
                        mse = float(mean_squared_error(test_data.y, y_pred))
                        nmse = float(mean_squared_error(test_data.y, y_pred)/np.var(test_data.y, ddof=1))
                        r2 = float(r2_score(test_data.y, y_pred))

                        results.append({
                            "Dataset": dataset_name,
                            "Number_of_samples": n,
                            "Number_of_features": p,
                            "Method": method,
                            "Fold": fold_number + 1,
                            "w_id": w_id + 1,
                            "mse": mse,
                            "nmse": nmse,
                            "r2": r2,
                        })

            # Save incremental progress after each dataset is complete
            df = pd.DataFrame(results)
            df.to_csv(raw_results_path, index=False)

            if verbose:
                print(f"Saved incremental results after completing dataset: {dataset_name}")

        except Exception as e:
            print(f"Error processing dataset {dataset_name}: {e}")

    # All datasets have been processed, create final results
    if not results:
        print("No results were generated.")
        return

    df = pd.DataFrame(results)

    # Save the final raw results (should be identical to the last incremental save, but just to be sure)
    df.to_csv(raw_results_path, index=False)

    # average values over folds
    df_fold_avg = (
        df.groupby(['Dataset', 'Method', "w_id"], as_index=False)
        .agg({
            'Number_of_samples': 'first',
            'Number_of_features': 'first',
            'Method': 'first',
            'mse': ['mean', 'std'],
            'nmse': ['mean', 'std'],
            'r2': ['mean', 'std']
        })
    )

    df_fold_avg.columns = ['_'.join(col).strip('_').replace('_first', '') for col in df_fold_avg.columns]
    df_fold_avg.to_csv(os.path.join(directory_folder, "fold_avg_results_" + id_results + ".csv"), index=False)

    # Take optimal best W for each dataset
    # pivot just to find the diff per dataset + W
    df_pivot = df_fold_avg.pivot_table(index=["Dataset", "w_id"],
                                       columns="Method",
                                       values="nmse_mean").reset_index()
    if "Pilot" in methods:
        df_pivot["diff"] = df_pivot["Pilot"] - df_pivot["PilotI"]
    elif "Sklearn" in methods:
        df_pivot["diff"] = df_pivot["Sklearn"] - df_pivot["SklearnI"]
    else:
        raise ValueError("Method 'Pilot' or 'Sklearn' not found.")

    # get best W indices per dataset
    idx = df_pivot.groupby("Dataset")["diff"].idxmax()

    # map those dataset+W back to the original df
    best = df_pivot.loc[idx, ["Dataset", "w_id"]]

    # keep only matching rows in original df
    df_final = df_fold_avg.merge(best, on=["Dataset", "w_id"])
    df_final.to_csv(os.path.join(directory_folder, "final_results_" + id_results + ".csv"), index=False)

    if verbose:
        print(f"Finished at time: {datetime.now().strftime('%H-%M-%S')}")
        print(f"Id of the results: {id_results}")
    return df


def find_latest_results_file(directory_result):
    """
    Find the most recent raw results file to resume from.

    Parameters:
    -----------
    directory_result : str
        Base directory to search for results files

    Returns:
    --------
    str or None
        Path to the most recent raw results file, or None if no files found
    """

    # Get all results directories
    result_dirs = glob.glob(os.path.join(directory_result, "Results_*"))

    if not result_dirs:
        return None

    # Sort by modification time (most recent first)
    result_dirs.sort(key=os.path.getmtime, reverse=True)

    # Look for raw results file in the most recent directory
    for dir_path in result_dirs:
        raw_files = glob.glob(os.path.join(dir_path, "raw_results_*.csv"))
        if raw_files:
            # Return the most recent raw file
            raw_files.sort(key=os.path.getmtime, reverse=True)
            return raw_files[0]

    return None

# %%
methods = ["Pilot", "PilotW", "PilotI", "Sklearn", "SklearnW", "SklearnI"]
dataset_names = ['abalone', 'bank8FM', 'cpu_small', 'housing', 'kin8nm', 'puma8NH', 'bank32nh', 'cpu_act']

all_datasets_names2 = ["Abalone", "Airfoil", "Auto mpg", "Bike", "Bodyfat", "Boston Housing", "California Housing",
                       "Communities", "Concrete", "Diabetes", "Electricity", "Energy", "Graduate Admission", "Ozone",
                       "Power plant", "Real estate", "Residential", "Riboflavin", "Skills", "Slump test",
                       "Superconductor", "Temperature", "Thermography", "Walmart", "Wine"]
remove_names1 = ["Bike", "California Housing", "Electricity", "Power plant", "Slump test", "Superconductor"]
remove_names2 = ["Abalone", "Communities", "Residential", "Riboflavin", "Skills", "Temperature", "Thermography", "Wine"]
datasets_names2 = [name for name in all_datasets_names2 if name not in remove_names1 and name not in remove_names2]

dict_datasets_names = {name: "cs" for name in dataset_names} | {name: "pilot" for name in datasets_names2}
dict_datasets_names = {"bank8FM": "cs"}

output_directory_folder = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")), 'output/covariate_shift_benchmark')
os.makedirs(output_directory_folder, exist_ok=True)
run_benchmark(dict_datasets_names, methods, directory_result = output_directory_folder, verbose=True, resume=False)
