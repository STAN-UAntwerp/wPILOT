# %%
print("importing PILOT...")
from pilot.pilot import PILOT

print("import done")
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# %%
np.random.seed(0)
n = 1001
n_w = 1001
n4 = int(n / 4)
n_w4 = int(n_w / 4)
X1 = np.c_[np.linspace(0, 100, n)]
X_w = np.c_[np.linspace(0, 100, n_w)]

Y1 = np.c_[22 * np.ones(n4)]
Y2 = 10 + 0.3 * X1[n4:2 * n4]
Y3 = 60 - 0.7 * X1[2 * n4:3 * n4]
Y4 = 10 + 0.1 * X1[3 * n4:n]
Y_w1 = np.c_[26 * np.ones(n_w4)]
Y_w2 = 2 + 0.6 * X_w[n_w4:2 * n_w4]
Y_w3 = 81 - 0.98 * X_w[2 * n_w4:3 * n_w4]
Y_w4 = -1.5 + 0.25 * X_w[3 * n_w4:n_w]

f = np.r_[Y1, Y2, Y3, Y4]
f_w = np.r_[Y_w1, Y_w2, Y_w3, Y_w4]
Y1 = f + np.c_[np.random.randn(n)]
Y_w = f_w + 3*np.c_[np.random.randn(n_w)]
X = np.r_[X1, X_w]
Y = np.r_[Y1, Y_w][:, 0]
W1 = np.r_[np.ones(n), 0 * np.ones(n_w)]
W2 = np.r_[np.ones(n), 0.3 * np.ones(n_w)]
W3 = np.r_[np.ones(n), np.ones(n_w)]
W4 = np.r_[1*np.ones(n), 10 * np.ones(n_w)]
# W1 = W1/(W1.sum()/(n+n_w))
# W2 = W2/(W2.sum()/(n+n_w))
# W3 = W3/(W3.sum()/(n+n_w))
# W4 = W4/(W4.sum()/(n+n_w))

# plt.plot(X1, Y1, '.', markersize=2, color="orange")
# plt.plot(X_w, Y_w, '.', markersize=2, color="green")
# plt.plot(X1, f, color="blue")
# plt.plot(X_w, f_w, color="red")
# plt.show()
# %%
model1 = PILOT()
model1.fit(X, Y, W1)
model2 = PILOT()
model2.fit(X, Y, W2)
model3 = PILOT()
model3.fit(X, Y, W3)
model4 = PILOT()
model4.fit(X, Y, W4)

# %%
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# First plot (left)
ax1.scatter(X1, Y1, 4, marker='s', color='orange', edgecolors='none', label="Data f", alpha=.7)
ax1.scatter(X_w, Y_w, 4, marker='o', color='green', edgecolors='none', label="Data g", alpha=.7)
[true1] = ax1.plot(X1, f, color="orange", linewidth=2, label="True function f")
[true2] = ax1.plot(X_w, f_w, color="green", linewidth=2, linestyle="dashed", label="True function g")

# Create first legend for ax1 (Data)
data_legend_elements = [
    Line2D([0], [0], marker='s', color='w', markerfacecolor='orange', markersize=6, label='Data f'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=6, label='Data g'),
]

first_legend = ax1.legend(handles=data_legend_elements, loc='upper left')
ax1.add_artist(first_legend)

ax1.legend(handles=[true1, true2], loc='lower left')

# Second plot (right)
ax2.scatter(X1, Y1, 4, marker='s', color='orange', edgecolors='none', label="Data f", alpha=.7)
ax2.scatter(X_w, Y_w, 4, marker='o', color='green', edgecolors='none', label="Data g", alpha=.7)
[line1] = ax2.plot(X1, model1.predict(X1), label='$W_f = 1, W_g = 0$', color="red", linewidth=2)
[line2] = ax2.plot(X1, model2.predict(X1), label='$W_f = 1, W_g = 0.3$', color="purple", linestyle="--", linewidth=2)
[line3] = ax2.plot(X1, model3.predict(X1), label='$W_f = 1, W_g = 1$', color="blue", linestyle="-.", linewidth=2)
[line4] = ax2.plot(X1, model4.predict(X1), label='$W_f = 1, W_g = 10$', color="brown", linestyle=(0, (3, 1, 1, 1, 1, 1)), linewidth=2)

# Add the first legend to ax2
first_legend_ax2 = ax2.legend(handles=data_legend_elements, loc='upper left')
ax2.add_artist(first_legend_ax2)  # Keep the first legend when adding the second

# Add the second legend with title to ax2
ax2.legend(handles=[line1, line2, line3, line4], title='Weighted PILOT',
                          loc='lower left')

ax2.set_xlabel("Explanatory variable")
ax2.set_ylabel("Response variable")

# Optional: Add labels to the first plot as well
ax1.set_xlabel("Explanatory variable")
ax1.set_ylabel("Response variable")

output_directory = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")), 'output/simple_example.pdf')
plt.savefig(output_directory, bbox_inches='tight')
plt.show()
