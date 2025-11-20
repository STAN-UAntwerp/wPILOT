import matplotlib.pyplot as plt
import numpy as np
from denseweight import DenseWeight
import matplotlib as mpl
import os
mpl.rcParams['mathtext.fontset'] = 'dejavuserif'

#%%
# Make plotting dat
# Generate sample data points
np.random.seed(42)
data_points = np.array([1, 1.7, 2, 2.1, 2.3, 2.35, 2.5, 2.5, 2.65, 2.7, 2.9, 2.9, 3.2, 3.2, 3.25, 3.5, 3.7, 4, 5.5, 6.2, 8]) + 1
x_range = np.linspace(0, 11, 1000)

denseweight = DenseWeight(alpha=1)
W_data_points = denseweight.fit(data_points)
density = denseweight.kernel.evaluate(x_range)
x_range_weights = denseweight(x_range)

#%%
# Set up the figure with 2x2 subplots
fig, ((ax1, ax2, ax3, ax4)) = plt.subplots(1, 4, figsize=(14, 2))
xmin = -1; xmax = 12
ymin = -0.05; ymax = 0.45

# Panel 1: Original data points
ax1.plot(x_range, np.zeros_like(x_range), color='k', linewidth=0.5, alpha=0.5)
ax1.scatter(data_points, np.zeros_like(data_points),
           marker='x', s=100, c='black', linewidth=1)
ax1.set_xlim(xmin, xmax)
ax1.set_ylim(ymin, ymax)
ax1.set_xticks([])
ax1.set_yticks([])

# Panel 2: Kernel Density Estimation result
ax2.plot(x_range, np.zeros_like(x_range), color='k', linewidth=0.5, alpha=0.5)
ax2.fill_between(x_range, density, alpha=0.7, color='steelblue', label='KDE')
ax2.scatter(data_points, np.zeros_like(data_points),
           marker='x', s=100, c='black', linewidth=1)
ax2.set_xlim(xmin, xmax)
ax2.set_ylim(ymin, ymax)
ax2.set_xticks([])
ax2.set_yticks([])

# Panel 3: Weighting function
ax3.plot(x_range, np.zeros_like(x_range), color='k', linewidth=0.5, alpha=0.5)
ax3.fill_between(x_range, density, alpha=0.7, color='steelblue')
ax3.plot(x_range, x_range_weights/12 + 0.06, color='black', linewidth=1)
ax3.scatter(data_points, np.zeros_like(data_points),
            marker='x', s=100, c='black', linewidth=1)
ax3.set_xlim(xmin, xmax)
ax3.set_ylim(ymin, ymax)
ax3.set_xticks([])
ax3.set_yticks([])

# Panel 4: Loss weighting visualization
ax4.plot(x_range, np.zeros_like(x_range), color='k', linewidth=0.5, alpha=0.5)
ax4.plot(x_range, x_range_weights/12 + 0.06, color='black', linewidth=1)
ax4.scatter(data_points, np.zeros_like(data_points),
            marker='x', s=50*(W_data_points + 0.4), c='black', linewidth=1)
ax4.set_xlim(xmin, xmax)
ax4.set_ylim(ymin, ymax)
ax4.set_xticks([])
ax4.set_yticks([])


arrow_start, arrow_stop = 1.11, 1.62
arrow_center = 1.36
arrow_height = 0.3
i_height: float = 0.12
text_height = 0.45

# Add arrow form 1 to 2
ax1.annotate('', xy=(arrow_stop, arrow_height), xytext=(arrow_start, arrow_height),
            xycoords='axes fraction', textcoords='axes fraction',
            arrowprops=dict(arrowstyle='->', lw=2, color='black'))
ax1.text(arrow_center, i_height, '(i)', transform=ax1.transAxes, ha='center', fontsize=16, fontfamily='serif')
ax1.text(arrow_center, text_height, 'Kernel\nDensity\nEstimation', transform=ax1.transAxes, ha='center', fontsize=16, fontfamily='serif')

# Add arrow form 2 to 3
ax2.annotate('', xy=(arrow_stop, arrow_height), xytext=(arrow_start, arrow_height),
            xycoords='axes fraction', textcoords='axes fraction',
            arrowprops=dict(arrowstyle='->', lw=2, color='black'))
ax2.text(arrow_center, i_height, '(ii)', transform=ax2.transAxes, ha='center', fontsize=16, fontfamily='serif')
ax2.text(arrow_center, text_height, 'Weighting\nFunction', transform=ax2.transAxes, ha='center', fontsize=16, fontfamily='serif')

# Add arrow form 3 to 4
ax3.annotate('', xy=(arrow_stop, arrow_height), xytext=(arrow_start, arrow_height),
            xycoords='axes fraction', textcoords='axes fraction',
            arrowprops=dict(arrowstyle='->', lw=2, color='black'))
ax3.text(arrow_center, i_height, '(iii)', transform=ax3.transAxes, ha='center', fontsize=16, fontfamily='serif')
ax3.text(arrow_center, text_height, 'Calculate\nWeights', transform=ax3.transAxes, ha='center', fontsize=16, fontfamily='serif')


plt.tight_layout()
output_directory = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")), 'output/dense_weight_idea.pdf')
plt.savefig(output_directory)
plt.show()

#%%
np.random.seed(42)
data_points = np.random.normal(loc=0, scale=1, size=500)
max_abs_value = np.max(np.abs(data_points))
x_range = np.linspace(-max_abs_value -0.01, max_abs_value+0.01, 1000)

dw0 = DenseWeight(alpha=0)
dw0.fit(data_points)
dw05 = DenseWeight(alpha=0.5)
dw05.fit(data_points)
dw075 = DenseWeight(alpha=0.75)
dw075.fit(data_points)
dw1 = DenseWeight(alpha=1)
dw1.fit(data_points)
dw11 = DenseWeight(alpha=1.1)
dw11.fit(data_points)

#%%
# Create figure and primary axis
fig, ax1 = plt.subplots(1, 1, figsize=(7.5, 3))

# Left y-axis for weights (kernel)
density_fill = ax1.fill_between(x_range, dw0.kernel(x_range), alpha=0.5, color='lightblue')
ax1.set_xlabel('$y$')
ax1.set_ylabel('Density')
ax1.set_ylim(0, 0.7)  # Set scale as mentioned

# Create secondary y-axis for density
ax2 = ax1.twinx()

colors = ["C0", "purple", "red", "C2", "brown"]
alpha_values = [0, 0.5, 0.75, 1.0, 1.1]
line_styles = ['-', '--', '-.', (0, (3, 1, 1, 1, 1, 1)), (0, (3, 1, 1, 1, 1, 1, 1, 1))]
dw_objects = [dw0, dw05, dw075, dw1, dw11]

# Plot density curves on right y-axis
for i, (dw, alpha, line_style, color) in enumerate(zip(dw_objects, alpha_values, line_styles, colors)):
    ax2.plot(x_range, dw(x_range), linestyle=line_style, color=color, linewidth=2,
             label=f'$f_w({alpha}, y)$')

ax2.set_ylabel('Weights')
ax2.set_ylim(0, 4.6)  # Set scale as mentioned

# Add legend - combine both line plots and fill
lines1, labels1 = ax2.get_legend_handles_labels()
lines2 = [density_fill]
labels2 = [r'$p^\prime(y)$']

# Combine and create legend
all_lines = lines1 + lines2
all_labels = labels1 + labels2
#ax2.legend(all_lines, all_labels, loc='upper left')
ax2.legend(all_lines, all_labels,bbox_to_anchor=(1.12, 0.5), loc="center left", borderaxespad=0)

plt.tight_layout()
output_directory = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")), 'output/dense_weight_function.pdf')
plt.savefig(output_directory)
plt.show()
