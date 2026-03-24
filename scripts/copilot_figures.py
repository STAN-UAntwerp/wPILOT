print("importing PILOT...")
from pilot.copilot import coPILOT

print("import done")

from util.benchmark_util import load_pmlb_data
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

#%%
COLOR_NODES = {"lin": "C4", "blin":"C0","plin": "C1", "pcon":"C3"}

class pilot_model:
    type: str
    feature_idx: int

    def __init__(self, type, feature_idx, left_model, right_model):
        self.type = type
        self.feature_idx = feature_idx
        self.left_model = left_model
        self.right_model = right_model


def plot_models_along_path(tree, idx_point, X_train, y_res, directory_folder, w = None):
    if tree.node == "END":
        return []

    fig = plt.figure()
    if w is None:
        w = np.ones(len(y_res))
    feature_idx = tree.pivot[0]
    min_x = min(X_train[:, feature_idx])
    max_x = max(X_train[:, feature_idx])
    scaled_weights = (w.flatten() - np.mean(w))*100 + 10
    plt.scatter(X_train[:, feature_idx], y_res, s=scaled_weights, color='slategrey')
    plt.scatter(X_train[idx_point, feature_idx], y_res[idx_point], s=60+scaled_weights[idx_point],
                facecolors='r', marker='*')
    
    x_label_string = r"x^{(" + str(feature_idx+1) + r")}"
    if tree.node == "lin":
        x = [min_x, max_x]
        y = [tree.lm_l[1] + tree.lm_l[0]*x for x in x]
        plt.plot(x,y, color=COLOR_NODES[tree.node], linewidth=3)
        plt.title(f"Depth = {tree.model_depth} - Node: LIN - Feature: ${x_label_string}$")
        plt.xlabel(f"${x_label_string}$")
        y_label = "y" if tree.model_depth == 1 else "Residuals"
        plt.ylabel(y_label)
        plt.savefig(directory_folder + f"depth={tree.model_depth}_{tree.node}.pdf", bbox_inches='tight')
        plt.show()

        y_res = y_res - (tree.lm_l[1] + tree.lm_l[0]*X_train[:, feature_idx])
        next_tree = tree.left

    elif tree.node == "pconc":
        pass

    else: # tree.node == "pcon", "plin", "blin"
        pivot = tree.pivot[1]
        x1 = [min_x, pivot]
        x2 = [pivot, max_x]
        y1 = [tree.lm_l[1] + tree.lm_l[0] * x for x in x1]
        y2 = [tree.lm_r[1] + tree.lm_r[0] * x for x in x2]
        plt.plot(x1, y1, x2, y2, color=COLOR_NODES[tree.node], linewidth=3)
        node_name = str(tree.node).upper()
        plt.title(f"Depth = {tree.model_depth} - Node: {node_name} - Feature: ${x_label_string}$ - Pivot: {np.round(pivot,2)}")
        plt.xlabel(f"${x_label_string}$")
        y_label = "y" if tree.model_depth == 1 else "Residuals"
        plt.ylabel(y_label)
        plt.savefig(directory_folder + f"depth={tree.model_depth}_{tree.node}.pdf", bbox_inches='tight')
        plt.show()

        if X_train[idx_point,feature_idx] <= pivot: # Determine left or right subtree
            idx = X_train[:,feature_idx] <= pivot
            X_train = X_train[idx,:]
            y_res = y_res[idx]
            y_res = y_res - (tree.lm_l[1] + tree.lm_l[0] * X_train[:, feature_idx])
            next_tree = tree.left
        else:
            idx = X_train[:, feature_idx] > pivot
            X_train = X_train[idx, :]
            y_res = y_res[idx]
            y_res = y_res - (tree.lm_r[1] + tree.lm_r[0] * X_train[:, feature_idx])
            next_tree = tree.right

        w = w[idx]
        idx_point = np.sum(idx[:idx_point])

    return [fig] + plot_models_along_path(next_tree, idx_point, X_train, y_res, directory_folder, w)

def save_two_figs(figs, directory):
    fig_combined, axes = plt.subplots(1, 2, figsize=(10, 4))
    for i, fig in enumerate(figs):
        original_ax = fig.axes[0] # Assuming one axes per figure

        # Copy scatter points (PathCollection)
        for collection in original_ax.collections:
            # Get paths/offsets and colors for scatter points
            offsets = collection.get_offsets()
            sizes = collection.get_sizes()
            facecolors = collection.get_facecolors()
            edgecolors = collection.get_edgecolors()
            # Markers are handled by paths, but recreating is simpler
            # For simplicity, we'll try to get the original label for the legend
            label = collection.get_label() if collection.get_label() else "_nolegend_"

            axes[i].scatter(offsets[:, 0], offsets[:, 1],
                            s=sizes, facecolors=facecolors,
                            edgecolors=edgecolors, label=label,
                            marker=collection.get_paths()[0])  # Try to get original marker


        for line in original_ax.lines:
            axes[i].plot(line.get_xdata(), line.get_ydata(),
                         color=line.get_color(), linestyle=line.get_linestyle(),
                         marker=line.get_marker(), label=line.get_label(), linewidth=line.get_linewidth())

        # Copy titles, labels, limits, etc.
        # axes[i].set_title(original_ax.get_title())
        axes[i].set_xlabel(original_ax.get_xlabel())
        axes[i].set_ylabel(original_ax.get_ylabel())
        axes[i].set_xlim(original_ax.get_xlim())
        axes[i].set_ylim(original_ax.get_ylim())

        if original_ax.get_legend() is not None:
            axes[i].legend(loc=original_ax.get_legend().get_loc()) # Copy legend properties


    plt.tight_layout()
    plt.savefig(directory, bbox_inches='tight')
    plt.show()

# %%
dataset = load_pmlb_data("547_no2")

#idx_point = 361
idx_point = 28
X_train, _, y_train, _ = train_test_split(dataset.X, dataset.y, test_size=0.2, random_state=123)
X_train_fit = np.delete(X_train, idx_point, axis=0)
y_train_fit = np.delete(y_train, idx_point, axis=0)
model = coPILOT(alpha=0.5, max_n_estimators=2, max_depth=3, max_model_depth=5)
model.fit(X_train_fit, y_train_fit, categorical=dataset.cat_ids, stop_early=False)

tree1 = model.pilot_trees[0]
tree2 = model.pilot_trees[1]

#%%
output_directory_folder = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")), 'output/copilot_figures')
os.makedirs(output_directory_folder, exist_ok=True)
figs1 = plot_models_along_path(tree1.model_tree, idx_point, X_train, y_train, os.path.join(output_directory_folder, "547_no2_tree1_"))
save_two_figs([figs1[0], figs1[-1]], os.path.join(output_directory_folder, "547_no2_tree1_combined.pdf"))

# %%
error = (y_train - model.pilot_trees[0].predict(X_train))**2
loss = error/error.max()
weighted_loss = np.sum(loss)/len(loss)
beta = weighted_loss/(1-weighted_loss)
w = beta**(1-loss)

figs2 = plot_models_along_path(tree2.model_tree, idx_point, X_train, y_train, os.path.join(output_directory_folder, "547_no2_tree2_"), w)
save_two_figs([figs2[0], figs2[-1]], os.path.join(output_directory_folder, "547_no2_tree2_combined.pdf"))



