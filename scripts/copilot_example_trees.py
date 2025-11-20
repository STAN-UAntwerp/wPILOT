print("importing PILOT...")
from pilot.copilot import coPILOT

print("import done")

from util.benchmark_util import load_pmlb_data
from pilot.tree import visualize_tree
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split



# %%
def copilot_improvement(alpha, dataset_name, directory_result:str = None, directory_data: str = None, verbose : bool = False, use_train=False):
    dataset = load_pmlb_data(dataset_name)

    X_train, X_test, y_train, y_test = train_test_split(dataset.X, dataset.y, test_size=0.2, random_state=123)
    idx_point = 28
    X_train = np.delete(X_train, idx_point, axis=0)
    X_test = np.delete(X_test, idx_point, axis=0)
    y_train = np.delete(y_train, idx_point, axis=0)
    y_test = np.delete(y_test, idx_point, axis=0)
    model = coPILOT(alpha=alpha, max_n_estimators=2, max_depth=3, max_model_depth=5)
    model.fit(X_train, y_train, categorical=dataset.cat_ids, stop_early=False)

    if use_train:
        y_pred1 = model.pilot_trees[0].predict(X_train)
        y_pred = model.predict(X_train, method="Average")
        res_pred1 = y_train - y_pred1
        res_pred = y_train - y_pred
        y_true = y_train
    else:
        y_pred1 = model.pilot_trees[0].predict(X_test)
        y_pred = model.predict(X_test, method="Average")
        res_pred1 = y_test - y_pred1
        res_pred = y_test - y_pred
        y_true = y_test

    under_pred1 = res_pred1 > 0
    under_pred = res_pred > 0
    same_direction = under_pred1 & under_pred | ~under_pred1 & ~under_pred

    res_diff = np.abs(res_pred1) - np.abs(res_pred)
    improvement = res_diff > 0

    types = np.empty(len(res_diff), dtype=object)
    types[same_direction & improvement] = 'type1'
    types[~same_direction & improvement] = 'type2'
    types[same_direction & ~improvement] = 'type3'
    types[~same_direction & ~improvement] = 'type4'

    res_percent_pred = res_pred/y_true
    res_percent_pred1 = res_pred1/y_true
    return model, res_diff, types, res_percent_pred, res_percent_pred1

def get_linear_model(tree, n_features, x_values):
    coef = np.zeros(n_features + 1)
    if tree.node == "END":
        return coef # stop recursion at leaf node

    feature_idx = tree.pivot[0]
    if tree.node == "lin":
        coef[-1] += tree.lm_l[1] # add constant value
        coef[feature_idx] += tree.lm_l[0] # add linear coef
        next_tree = tree.left

    elif tree.node == "pconc":
        if np.isin(x_values[feature_idx], tree.pivot_c): # Determine left or right subtree
            coef[-1] += tree.lm_l[1]
            next_tree = tree.left
        else:
            coef[-1] += tree.lm_r[1]
            next_tree = tree.right

    else: # tree.node == "pcon", "plin", "blin"
        if x_values[feature_idx] <= tree.pivot[1]: # Determine left or right subtree
            coef[-1] += tree.lm_l[1]
            coef[feature_idx] += tree.lm_l[0]
            next_tree = tree.left
        else:
            coef[-1] += tree.lm_r[1]
            coef[feature_idx] += tree.lm_r[0]
            next_tree = tree.right
    # Do recursion on tree depth
    return coef + get_linear_model(next_tree, n_features, x_values)

# %%
dataset_name = "547_no2"
alpha = 0.5
use_train = True

model, res_diff, types, res_percent_pred, res_percent_pred1 = copilot_improvement(alpha, dataset_name, use_train=use_train)
dataset = load_pmlb_data(dataset_name)
X_train, X_test, y_train, y_test = train_test_split(dataset.X, dataset.y, test_size=0.2, random_state=123)

# %%
idx = 28
print(f"Improvement = {res_diff[idx]}")
print(f"Weight = {model.pilot_trees[1].w.flatten()[idx]}")
n_features = model.pilot_trees[0].n_features
np.set_printoptions(suppress=True,precision=3, floatmode='fixed')
coef1 = get_linear_model(model.pilot_trees[0].model_tree, n_features, X_train[idx,])
coef2 = get_linear_model(model.pilot_trees[1].model_tree, n_features, X_train[idx,])
coef_avg = np.array((coef1 + coef2)/2)
print(coef1)
print(coef2)
print(coef_avg)

# %%
print(f"coPILOT 1 prediction: {coef1[-1] + coef1[:-1]@X_train[idx,]}")
print(f"coPILOT 2 prediction: {coef2[-1] + coef2[:-1]@X_train[idx,]}")
prediction = coef_avg[-1] + coef_avg[:-1]@X_train[idx,]
print(f"coPILOT   prediction: {prediction}")
print(f"True      prediction: {y_train[idx]}")

# %%
visualize_tree(model.pilot_trees[0].model_tree, X_train)
plt.show()
visualize_tree(model.pilot_trees[1].model_tree, X_train)
plt.show()
model.pilot_trees[0].print_tree()
print("-------")
model.pilot_trees[1].print_tree()


