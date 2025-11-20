import numpy as np

PILOT_DATASETS_CAT_IDS = {
    'Abalone': np.array([0]),
    'Airfoil': np.array([-1]),
    'Auto mpg': np.array([1, 6]),
    'Bodyfat': np.array([-1]),
    'Boston Housing': np.array([3,8]),
    'Communities': np.array([-1]),
    'Concrete': np.array([-1]),
    'Diabetes': np.array([1]),
    'Energy': np.array([5]),
    'Graduate Admission': np.array([6]),
    'Ozone': np.array([10,11]),
    'Real estate': np.array([3]),
    'Residential': np.array([1,3,4]),
    'Riboflavin': np.array([-1]),
    'Skills': np.array([-1]),
    'Slump test': np.array([-1]),
    'Temperature': np.array([0,22,23,24]),
    'Thermography': np.array([0,1,2]),
    'Walmart': np.array([0,1,6,7,8]),
    'Wine': np.array([-1])
}
PMLB_DATASETS_CAT_IDS = {
    '556_analcatdata_apnea2': np.array([0,1]),
    '557_analcatdata_apnea1': np.array([0,1]),
    '522_pm10': np.array([-1]),
    '1028_SWD': np.array([0,1,2,3,4,5,6,7,8,9]),
    '485_analcatdata_vehicle': np.array([0,1,2,3]),
    '547_no2': np.array([-1]),
    '665_sleuth_case2002': np.array([0,1,2,3]),
    '210_cloud': np.array([0,1]),
    '229_pwLinear': np.array([0,1,2,3,4,5,6,7,8,9]),
    '230_machine_cpu': np.array([-1]),
    '656_fri_c1_100_5': np.array([-1]),
    '192_vineyard': np.array([-1]),
    '653_fri_c0_250_25': np.array([-1]),
    '687_sleuth_ex1605': np.array([-1]),
    '651_fri_c0_100_25': np.array([-1]),
    '658_fri_c3_250_25': np.array([-1])
}

DENSE_DATASETS_CAT_IDS = {
    'a1': np.array([8,9,10]),
    'a2': np.array([8,9,10]),
    'a3': np.array([8,9,10]),
    'a4': np.array([8,9,10]),
    'a5': np.array([8,9,10]),
    'a6': np.array([8,9,10]),
    'a7': np.array([8,9,10]),
    'abalone': np.array([7]),
    'acceleration': np.array([11,12,13]),
    'airfoild': np.array([-1]),
    'available_power': np.array([8,9,10,11,12,13,14]),
    'bank8FM': np.array([-1]),
    'boston': np.array([3,8]), # Manually selected
    'concreteStrength': np.array([-1]),
    'cpu_small': np.array([-1]),
    'delta_ailerons': np.array([-1]),
    'fuel_consumption_country': np.array([25,26,27,28,29,30,31,32,33,34,35,36]),
    'machineCpu': np.array([-1]),
    'maximal_torque': np.array([19,20,21,22,23,24,25,26,27,28,29,30,31]),
    'servo': np.array([2,3])
}

CS_DATASETS_CAT_IDS = {
    'abalone': np.array([0]),
    'bank8FM': np.array([-1]),
    'bank32nh': np.array([-1]),
    'cal_housing': np.array([-1]),
    'cpu_act': np.array([-1]),
    'cpu_small': np.array([-1]),
    'housing': np.array([-1]),
    'kin8nm': np.array([-1]),
    'puma8NH': np.array([-1])
}
