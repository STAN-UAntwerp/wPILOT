print("importing PILOT...")
from pilot.pilot import PILOT

print("import done")

from util.f1_score import f1_score
from util.benchmark_util import load_dense_weight_data
import pandas as pd
import os
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from denseweight import DenseWeight
from datetime import datetime
import glob

# %%
def run_benchmark(dataset_names: list[str], directory_result: str, verbose: bool = False, resume: bool = False):
    """
    Run benchmark on multiple datasets and save results incrementally.

    Parameters:
    -----------
    dataset_names : list[str]
        List of dataset names to benchmark
    settings : benchmarkSettings
        Settings for the benchmark
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
            dataset = load_dense_weight_data(dataset_name)
            (n, p) = (dataset.n_samples(), dataset.n_features())
            cross_validation = KFold(n_splits=5, random_state=123, shuffle=True)

            for fold_number, (train_index, test_index) in enumerate(cross_validation.split(dataset.X, dataset.y)):
                if verbose:
                    print(f"-> Fold number = {fold_number + 1}")

                train_data = dataset.subset(train_index)
                test_data = dataset.subset(test_index)

                try:
                    model = PILOT()
                    model.fit(train_data.X, train_data.y, categorical=train_data.cat_ids)
                    y_pred_pilot = model.predict(test_data.X)

                    denseweight = DenseWeight()
                    w = denseweight.fit(train_data.y)

                    model2 = PILOT()
                    model2.fit(train_data.X, train_data.y, w=w, categorical=train_data.cat_ids)
                    y_pred_pilotW = model2.predict(test_data.X)

                    model3 = DecisionTreeRegressor(random_state=123, max_depth=12)
                    model3.fit(train_data.X, train_data.y, sample_weight=w)
                    y_pred_cartW = model3.predict(test_data.X)
                except Exception as e:
                    # When a method fails, we stop processing the entire dataset
                    print(
                        f"Failed on dataset {dataset_name}, fold {fold_number + 1} with error: {e}")
                    print(f"Skipping the rest of dataset {dataset_name} and moving to next dataset")
                    raise  # Re-raise the exception to be caught by the outer try-except

                for y_pred, method in zip([y_pred_pilot, y_pred_pilotW, y_pred_cartW], ["Pilot", "PilotW", "CartW"]):
                    mse = float(mean_squared_error(test_data.y, y_pred))
                    r2 = float(r2_score(test_data.y, y_pred))
                    f1 = float(f1_score(test_data.y, y_pred, y_train=train_data.y))

                    results.append({
                        "Dataset": dataset_name,
                        "Number_of_samples": n,
                        "Number_of_features": p,
                        "Method": method,
                        "Fold": fold_number + 1,
                        "mse": mse,
                        "r2": r2,
                        "f1": f1
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
        df.groupby(['Dataset', 'Method'], as_index=False)
        .agg({
            'Number_of_samples': 'first',
            'Number_of_features': 'first',
            'Method': 'first',
            'mse': ['mean', 'std'],
            'r2': ['mean', 'std'],
            'f1': ['mean', 'std']
        })
    )

    df_fold_avg.columns = ['_'.join(col).strip('_').replace('_first', '') for col in df_fold_avg.columns]
    df_fold_avg.to_csv(os.path.join(directory_folder, "fold_avg_results_" + id_results + ".csv"), index=False)
    df_fold_avg.to_csv(os.path.join(directory_folder, "final_results_" + id_results + ".csv"), index=False)

    if verbose: print(f"Finished at time: {datetime.now().strftime('%H-%M-%S')}")
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
dataset_names = ['a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'a7', 'abalone', 'acceleration', 'airfoild', 'available_power',
                 'bank8FM', 'boston', 'concreteStrength', 'cpu_small', 'delta_ailerons',
                 'fuel_consumption_country', 'machineCpu', 'maximal_torque', 'servo']

output_directory_folder = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")), 'output/imbalanced_regression_benchmark')
os.makedirs(output_directory_folder, exist_ok=True)
run_benchmark(dataset_names, directory_result=output_directory_folder, verbose=True, resume=False)