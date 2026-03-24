print("importing PILOT...")
from pilot.pilot import DEFAULT_DF_SETTINGS
from util.benchmark_util import BenchmarkSettings, load_pmlb_data, load_pilot_paper_data, METHOD_FUNCTIONS

print("import done")

import pandas as pd
import numpy as np
import json
import os
from sklearn.model_selection import KFold
from datetime import datetime
import glob

# %%
def run_benchmark(dataset_names: list[str], settings: BenchmarkSettings,
                  directory_result: str, verbose: bool = False, resume: bool = False):
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
    if verbose:
        print(f"Started at time: {datetime.now().strftime('%H-%M-%S')}")

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
            if dataset_name[0].isdigit():
                dataset = load_pmlb_data(dataset_name)
            else:
                dataset = load_pilot_paper_data(dataset_name)

            (n, p) = (dataset.n_samples(), dataset.n_features())
            cross_validation = KFold(n_splits=5, random_state=123, shuffle=True)

            for fold_number, (train_index, test_index) in enumerate(cross_validation.split(dataset.X, dataset.y)):
                if verbose:
                    print(f"-> Fold number = {fold_number + 1}")

                train_data = dataset.subset(train_index)
                test_data = dataset.subset(test_index)

                for method in settings.get_methods():
                    for parameters in settings.get_parameters(method):
                        try:
                            result = METHOD_FUNCTIONS[method](train_data, test_data, **json.loads(parameters))
                        except Exception as e:
                            # When a method fails, we stop processing the entire dataset
                            print(
                                f"Method {method} failed on dataset {dataset_name}, fold {fold_number + 1} with error: {e}")
                            print(f"Skipping the rest of dataset {dataset_name} and moving to next dataset")
                            raise  # Re-raise the exception to be caught by the outer try-except

                        for method_id in settings.get_method_ids(method, parameters):
                            dataset_results.append({
                                "Dataset": dataset_name,
                                "Number_of_samples": n,
                                "Number_of_features": p,
                                "Fold": fold_number + 1,
                                "Method_id": method_id,
                                "Method": method,
                                "Parameters": json.dumps(parameters),
                                "mse": result.mse,
                                "r2": result.r2,
                                "node_count": sum(result.node_count.values()),
                                "node_count_no_lin": sum(v for k, v in result.node_count.items() if k != "lin"),
                                "fit_duration": result.fit_duration,
                                "predict_duration": result.predict_duration,
                            })

            # Add the results from this dataset to the main results list
            results.extend(dataset_results)

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
        df.groupby(['Dataset', 'Method_id', 'Parameters'], as_index=False)
        .agg({
            'Number_of_samples': 'first',
            'Number_of_features': 'first',
            'Method': 'first',
            'mse': ['mean', 'std'],
            'r2': ['mean', 'std'],
            'node_count': ['mean', 'max'],
            'node_count_no_lin': ['mean', 'max'],
            'fit_duration': 'mean',
            'predict_duration': 'mean'
        })
    )

    df_fold_avg.columns = ['_'.join(col).strip('_').replace('_first', '') for col in df_fold_avg.columns]
    df_fold_avg.to_csv(os.path.join(directory_folder, "fold_avg_results_" + id_results + ".csv"), index=False)

    counts = df_fold_avg.groupby(['Dataset', 'Method_id']).size().rename("count")
    df_final = df_fold_avg.loc[df_fold_avg.groupby(['Dataset', 'Method_id'])['r2_mean'].idxmax()].copy()

    # Take optimal methods
    df_final = df_final.merge(counts, on=['Dataset', 'Method_id'])
    df_final['optimal'] = df_final['count'] > 1
    df_final.drop(columns='count', inplace=True)

    df_final = df_final.rename(columns={"Method_id": "Method_call"})

    df_final = df_final[[
        'Dataset', 'Number_of_samples', 'Number_of_features',
        'Method', 'Parameters', 'mse_mean', 'mse_std', 'r2_mean', 'r2_std',
        'node_count_mean','node_count_max', 'node_count_no_lin_mean', 'node_count_no_lin_max',
        'fit_duration_mean', 'predict_duration_mean',
        'Method_call', 'optimal'
    ]]
    df_final.to_csv(os.path.join(directory_folder, "final_results_" + id_results + ".csv"), index=False)

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
max_depth_list = [5, 12, 30]
alpha_list = [0.2, 0.5, 1]
df_settings_list = [(dict(zip(DEFAULT_DF_SETTINGS.keys(),(1 + alpha * (np.array(list(DEFAULT_DF_SETTINGS.values())) - 1)).tolist()))) for alpha in alpha_list]

benchmark_settings = BenchmarkSettings()
#benchmark_settings.add_method("Pilot",{"max_depth": 12, "df_settings": DEFAULT_DF_SETTINGS})
benchmark_settings.add_method("Pilot",{"max_depth": max_depth_list, "df_settings": df_settings_list})
benchmark_settings.add_method("coPilot_avg", {"max_n_estimators": 2, "max_depth": max_depth_list, "alpha": alpha_list})
benchmark_settings.add_method("XGBoost", {"max_n_estimators": 2, "max_depth": max_depth_list, "alpha": alpha_list})

datasets_names = ['556_analcatdata_apnea2', '557_analcatdata_apnea1', '522_pm10', '1028_SWD',
                  '485_analcatdata_vehicle', '547_no2', '665_sleuth_case2002', '210_cloud',
                  '229_pwLinear', '230_machine_cpu', '656_fri_c1_100_5', '192_vineyard', '653_fri_c0_250_25',
                  '687_sleuth_ex1605', '651_fri_c0_100_25',
                  '658_fri_c3_250_25']  # '574_house_16H' '1201_BNG_breastTumor'

all_datasets_names2 = ["Abalone", "Airfoil", "Auto mpg", "Bike", "Bodyfat", "Boston Housing", "California Housing",
                       "Communities", "Concrete", "Diabetes", "Electricity", "Energy", "Graduate Admission", "Ozone",
                       "Power plant", "Real estate", "Residential", "Riboflavin", "Skills", "Slump test",
                       "Superconductor", "Temperature", "Thermography", "Walmart", "Wine"]
remove_names2 = ["Bike", "California Housing", "Electricity", "Power plant", "Superconductor"]
datasets_names2 = [name for name in all_datasets_names2 if name not in remove_names2]

final_datasets_names = datasets_names + datasets_names2

# %%
output_directory_folder = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")), 'output/copilot_benchmark')
os.makedirs(output_directory_folder, exist_ok=True)
run_benchmark(final_datasets_names, benchmark_settings, directory_result=output_directory_folder, verbose=True, resume=False)
