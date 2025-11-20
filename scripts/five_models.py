import matplotlib.pyplot as plt
import numpy as np
import os

#%%
col = '#2D6AA4'
size = 18
line_width = 3
# Create figure with 5 subplots
fig, axes = plt.subplots(1, 5, figsize=(15, 3))

# Function 2: Constant
x2 = np.linspace(0, 3, 100)
y2 = np.ones_like(x2) * 1.5
axes[0].plot(x2, y2, color=col, linewidth=line_width)
axes[0].set_title('Constant', fontsize=size)
axes[0].set_xlim(0, 3)
axes[0].set_ylim(0, 3)
axes[0].grid(True, alpha=0.3)
axes[0].set_xticks([])
axes[0].set_yticks([])

# Function 1: Piecewise constant
x1_1 = np.array([0, 1])
y1_1 = np.array([0.7, 0.7])
x1_2 = np.array([1, 2])
y1_2 = np.array([2.3, 2.3])
axes[1].plot(x1_1, y1_1, color=col, linewidth=line_width)
axes[1].plot(x1_2, y1_2, color=col, linewidth=line_width)
axes[1].set_title('Piecewise Constant', fontsize=size)
axes[1].set_xlim(0, 2)
axes[1].set_ylim(0, 3)
axes[1].grid(True, alpha=0.3)
axes[1].set_xticks([])
axes[1].set_yticks([])

# Function 3: Linear
x3 = np.linspace(0, 3, 100)
y3 = 1.5 + 0.6 * (x3 - 1.5)
axes[2].plot(x3, y3, color=col, linewidth=line_width)
axes[2].set_title('Linear', fontsize=size)
axes[2].set_xlim(0, 3)
axes[2].set_ylim(0, 3)
axes[2].grid(True, alpha=0.3)
axes[2].set_xticks([])
axes[2].set_yticks([])

# Function 4: Broken linear
x4_1 = np.linspace(0, 1.5, 50)
y4_1 = 2.3 - 0.8 * x4_1
x4_2 = np.linspace(1.5, 3, 50)
y4_2 = 1.1 + 0.5 * (x4_2 - 1.5)
axes[3].plot(x4_1, y4_1, color=col, linewidth=line_width)
axes[3].plot(x4_2, y4_2, color=col, linewidth=line_width)
axes[3].set_title('Broken Linear', fontsize=size)
axes[3].set_xlim(0, 3)
axes[3].set_ylim(0, 3)
axes[3].grid(True, alpha=0.3)
axes[3].set_xticks([])
axes[3].set_yticks([])

# Function 5: Piecewise linear
x5_1 = np.linspace(0, 1, 50)
y5_1 = 2.5 - 1 * x5_1
x5_2 = np.linspace(1, 2, 50)
y5_2 = 0.7 + 0.3 * (x5_2 - 1)
axes[4].plot(x5_1, y5_1, color=col, linewidth=line_width)
axes[4].plot(x5_2, y5_2, color=col, linewidth=line_width)
axes[4].set_title('Piecewise Linear', fontsize=size)
axes[4].set_xlim(0, 2)
axes[4].set_ylim(0, 3)
axes[4].grid(True, alpha=0.3)
axes[4].set_xticks([])
axes[4].set_yticks([])

# Add vertical dashed lines for piecewise functions
axes[1].axvline(x=1, ymin=0.7/3, ymax=2.3/3, color=col, linestyle='--', alpha=0.7)
axes[4].axvline(x=1, ymin=0.7/3, ymax=1.5/3, color=col, linestyle='--', alpha=0.7)

plt.tight_layout()
output_directory = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")), 'output/five_models.pdf')
plt.savefig(output_directory)
plt.show()