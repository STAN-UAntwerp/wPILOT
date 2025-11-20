# Fast Weighted Linear Model Trees

This repository contains the implementation for the Weighted PIecewise Linear Organic Tree (PILOT), a linear model tree algorithm
first proposed in the paper Raymaekers, J., Rousseeuw, P. J., Verdonck, T., & Yao, R. (2024). Fast linear model trees by
PILOT. Machine Learning, 1-50. https://doi.org/10.1007/s10994-024-06590-3.

The code implements the weighted extension of the PILOT algorithm to support observation
weights while maintaining the same computational complexity, as proposed in the paper Debois, F., Raymaekers, 
J., Servotte, T. & Verdonck, T. (2025). Fast weighted linear model trees. _Under Review_

## Overview

PILOT is a linear model tree algorithm that combines the simplicity and speed of CART (Classification and Regression
Trees) with the flexibility of linear models. The weighted extension allows each observation to be assigned importance
weights during training, enabling applications in boosting, imbalanced regression, and covariate shift problems.

## Repository Structure

### Core Implementation (`pilot/`)

- **`pilot.py`**: Main implementation of the (weighted) PILOT algorithm
- **`copilot.py`**: One-step boosting extension that trains two complementary PILOT trees
- **`tree.py`**: Tree structure and node implementations

### Scripts (`scripts/`)

Contains scripts for generating figures and running benchmark for comparing PILOT variants.

- **`baseline_python_example.py`**: Simple example of (unweighted) PILOT
- **`benchmark_copilot.py`**: coPILOT benchmark
- **`Benchmark_CS.py`**: Covariate shift benchmark
- **`Benchmark_dense_weight.py`**: Imbalanced regression benchmark
- **`copilot_example_trees.py`**: Visualization of coPILOT tree example
- **`copilot_figures.py`**: Detail figures of coPILOT tree example
- **`dense_weight_figures.py`**: Figures for imbalanced regression results
- **`five_models.py`**: Visualization of the five PILOT model types
- **`simple_example.py`**: Simple demonstration of weighted PILOT

### Datasets

- **`datasets_pilot/`**: Datasets used in the original PILOT paper
- **`datasets_dense_weight_2/`**: Imbalanced regression datasets
- **`datasets_CS/`**: Covariate shift datasets
- **`datasets_PMLB/`**: Penn Machine Learning Benchmark datasets

### Utilities (`util/`)

- **`benchmark_info.py`**: Categorical indices for all datasets
- **`benchmark_util.py`**: Utility functions for benchmark data processing
- **`f1_Score.py`**: Implementation of weighted F1 score for imbalanced regression

### Output (`Output/`)

- Contains the benchmark results and generated figures form various files.

## Requirements

See `requirements.txt` for the complete list of dependencies.