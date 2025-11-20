import numpy as np
from scipy.interpolate import CubicHermiteSpline
from sklearn.isotonic import IsotonicRegression
import matplotlib.pyplot as plt

#%%
def next_change_in_value(i, y_rel):
    if i >= len(y_rel) - 1:
        return 0

    if not y_rel[i + 1] == y_rel[i]:
        return i + 1

    return next_change_in_value(i + 1, y_rel)

def find_bumps(y_not_unique, y_rel_not_unique):
    _, unique_indices = np.unique(y_not_unique, return_index=True)
    y = y_not_unique[unique_indices]
    y_rel = y_rel_not_unique[unique_indices]

    b_star = []
    b_min = []

    n = len(y_rel)

    # Begin point
    i_next = next_change_in_value(0, y_rel)
    if y_rel[0] > y_rel[i_next]:
        b_star.append(y[0])
        b_min.append(-np.inf)
    elif y_rel[0] < y_rel[i_next]:
        b_min.append(y[0])

    i = 1
    while i < n:
        i_next = next_change_in_value(i, y_rel)

        # End point
        if i_next == 0 and y_rel[i] > y_rel[i - 1]:
            b_star.append(y[-1])
            i = n
        elif i_next == 0 and y_rel[i] < y_rel[i - 1]:
            b_min.append(y[-1])
            b_star.append(np.inf)
            i = n

        # Interior point
        elif y_rel[i] > y_rel[i - 1] and y_rel[i] > y_rel[i_next]:
            b_star.append(np.mean(y[np.arange(i, i_next)]))
            i = i_next
        elif y_rel[i] < y_rel[i - 1] and y_rel[i] < y_rel[i_next]:
            b_min.append(np.mean(y[np.arange(i, i_next)]))
            i = i_next

        else:
            i += 1

    b_tri = []
    for i in range(len(b_min)):
        b_star_val = b_star[i]
        if b_star_val == np.inf:
            b_tri.append(None)
        else:
            if i == len(b_min) - 1:
                b_tri.append(2 * np.abs(b_min[i] - b_star_val))
            else:
                b_tri.append(2 * np.minimum(np.abs(b_min[i] - b_star_val), np.abs(b_star_val - b_min[i + 1])))

    return b_star, b_min, b_tri

def bounded_loss(y, y_hat, y_rel, y_train, y_train_rel):
    y_all = np.append(y, y_train)
    y_all_rel = np.append(y_rel, y_train_rel)
    sorted_idx = np.argsort(y_all)
    b_star_list, b_min_list, b_tri_list = find_bumps(y_all[sorted_idx], y_all_rel[sorted_idx])

    benefit_tres = []
    cost_tres = []
    for y_val, y_hat_val in zip(y, y_hat):
        bump_i = np.digitize(y_val, b_min_list) - 1
        if y_hat_val < y_val:
            if bump_i == 0:
                b_star_val = -np.inf
            else:
                b_star_val = b_star_list[bump_i - 1]
            benefit_tres2 = np.abs(y_val - b_min_list[bump_i])
            cost_tres2 = np.abs(y_val - b_star_val)
        else:
            if bump_i == len(b_min_list) - 1:
                b_min_val = np.inf
                b_star_val = np.inf
            else:
                b_min_val = b_min_list[bump_i + 1]
                b_star_val = b_star_list[bump_i + 1]

            benefit_tres2 = np.abs(y_val - b_min_val)
            cost_tres2 = np.abs(y_val - b_star_val)
        benefit_tres.append(np.minimum(benefit_tres2, b_tri_list[bump_i]))
        cost_tres.append(np.minimum(cost_tres2, b_tri_list[bump_i]))

    loss = np.abs(y - y_hat) # Use absolute loss
    benefit_tres = np.asarray(benefit_tres)
    cost_tres = np.asarray(cost_tres)

    gamma_b = np.ones_like(loss)
    mask_b = (benefit_tres != 0) & (loss < benefit_tres)
    gamma_b[mask_b] = loss[mask_b] / benefit_tres[mask_b]

    gamma_c = np.ones_like(loss)
    mask_c = (cost_tres != 0) & (loss < cost_tres)
    gamma_c[mask_c] = loss[mask_c] / cost_tres[mask_c]
    return gamma_b, gamma_c

def utility_score(y, y_hat, y_rel, y_train, y_train_rel, phi, p=0.5):
    # y are the true values - array
    # y_hat is the predicted y values (the target) array
    # phi is the relevance function (coming form dense weights) - function
    # p - probability used in calculations
    y_hat_rel = phi(y_hat)
    y_y_hat_rel = (1-p)*y_hat_rel + p*y_rel

    gamma_b, gamma_c = bounded_loss(y, y_hat, y_rel, y_train, y_train_rel)
    u = y_rel*(1 - gamma_b) - y_y_hat_rel*gamma_c
    return u

def f1_score(y_test, y_hat, y_train = np.array([]), beta=1, p=0.5, phi=None, t_e=1, verbose=False):
    # function is based on the work in chapter 3 and 4 of Rita P. Ribeiro. Utility-based Regression. PhD thesis, 2011
    # y are the true values - array
    # y_hat is the predicted y values (the target) - array
    # w are the weights - array
    # beta is the value used to determine the f1 score and how it combines recall and precision value between 0 and 1
    # phi is the relevance function (coming form dense weights) - function
    # t_e - the threshold value for determining a target event - value between 0 and 1

    y_all = np.append(y_train, y_test)
    if phi is None:
        Q1 = np.percentile(y_all, 25)
        Q3 = np.percentile(y_all, 75)
        IQR = Q3 - Q1
        median = np.median(y_all)
        left_border = Q1 - 1.5 * IQR
        right_border = Q3 + 1.5 * IQR
        phi_x = np.array([left_border, median, right_border])
        phi_y = np.array([1, 0, 1])
        phi_d = np.zeros_like(phi_x)
        spline = CubicHermiteSpline(phi_x, phi_y, phi_d, extrapolate=True)

        def phi(x_input):
            x_input = np.asarray(x_input)
            y_out = spline(x_input)
            y_out[x_input < left_border] = 1
            y_out[x_input > right_border] = 1
            return y_out

    if verbose:
        x_plot = np.linspace(np.min(y_all), np.max(y_all), 1000)
        y_plot = phi(x_plot)
        plt.plot(x_plot, y_plot)
        plt.xlabel('True y values range')
        plt.ylabel('Relevance')
        plt.title(f"Relevance function phi")
        plt.show()

    z = np.where(phi(y_test) >= t_e, 1, 0)
    y_rel = phi(y_test)
    y_train_rel = phi(y_train)

    u = utility_score(y_test, y_hat, y_rel, y_train, y_train_rel, phi)
    sorted_idx = np.argsort(u)
    u_sorted = u[sorted_idx]
    z_sorted = z[sorted_idx]

    pav = IsotonicRegression(y_min=0, y_max=1, increasing=True, out_of_bounds='clip')
    s_sorted = pav.fit_transform(u_sorted, z_sorted)
    s = np.empty_like(s_sorted)
    s[sorted_idx] = s_sorted

    z_hat = np.where(s > (1 - u) / 2, 1, 0)


    if sum((z_hat == 1) * (z == 1)) == 0:
        return 0
    recall = np.sum((z_hat == 1) * (z == 1) * (1 + u)) / np.sum((z == 1) * (1 + y_rel))
    precision = np.sum((z_hat == 1) * (z == 1) * (1 + u)) / (
            np.sum((z_hat == 1) * (z == 1) * (1 + y_rel)) + np.sum((z_hat == 1) * (z == 0) * (2 - p * (1 - y_rel))) )
    if verbose:
        print(f'Mean utility={np.mean(u)}')
        print(f'Recall={recall}')
        print(f'Precision:={precision}')

    f1 = ((beta ** 2 + 1) * precision * recall) / (beta ** 2 * precision + recall)
    return f1
