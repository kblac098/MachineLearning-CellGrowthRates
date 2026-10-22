# =====================================================================
# MAIN PROGRAM
# Your code starts here. This is the file that will be run.
# =====================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV, KFold, learning_curve
from sklearn.metrics import mean_squared_error, r2_score

import src.functions as sf  # This allows you to access functions from functions.py using sf.function_name()
from src.classes import Batch  # This allows you to import individual Classes from classes.py
from src.constants import *  # This imports all constants from constants.py

ph_optimal = sf.calculate_average(PH_LIMITS)
temperature_optimal = sf.calculate_average(TEMPERATURE_LIMITS)

batch = Batch(batch_id=1, ph_opt=ph_optimal, temperature_opt=temperature_optimal)
batch.display_info()

# ---------------------------------------------------------------------
# 1. Load the dataset and split it into 5 equal-sized arrays
# ---------------------------------------------------------------------
DATA_PATH = "datasets/dataset_cell_growth_4factor.csv"
NUM_SPLITS = 5

# Read the CSV into a pandas DataFrame
data = pd.read_csv(DATA_PATH)

# Count the number of datapoints (rows)
num_datapoints = len(data)
print(f"\nNumber of datapoints in file: {num_datapoints}")

# Work out how many datapoints each array gets; leftovers are ignored
split_size = num_datapoints // NUM_SPLITS
num_ignored = num_datapoints - split_size * NUM_SPLITS
print(f"Each array will hold {split_size} datapoints ({num_ignored} extra ignored)")

# Convert to a numpy array so each split is a plain 2D array
# (rows = datapoints, columns = the CSV columns)
data_array = data.to_numpy()

# Step through the data, filling each array one at a time:
# array 1 gets the first split_size rows, array 2 the next split_size, etc.
splits = []
for i in range(NUM_SPLITS):
    start = i * split_size
    end = start + split_size
    splits.append(data_array[start:end])

for i, split in enumerate(splits, start=1):
    print(f"Array {i}: shape {split.shape}")

# Column 0 is the target (specific growth rate); columns 1-4 are the features
TARGET_NAME = data.columns[0]
FEATURE_NAMES = list(data.columns[1:])


def split_xy(array):
    """Separate a data array into features (X) and target (y)."""
    return array[:, 1:], array[:, 0]


def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


# ---------------------------------------------------------------------
# 2. Define the two candidate models and their hyperparameter grids
#    (Pipelines keep the scaler fitted on training data only)
# ---------------------------------------------------------------------
INNER_CV = KFold(n_splits=5, shuffle=True, random_state=0)
SCORING = "neg_root_mean_squared_error"  # GridSearchCV maximizes, so RMSE is negated

models = {
    "Polynomial": {
        "pipeline": Pipeline([
            ("scaler", StandardScaler()),
            ("poly", PolynomialFeatures(include_bias=False)),
            ("regressor", LinearRegression()),
        ]),
        "param_grid": {"poly__degree": list(range(1, 7))},
    },
    "kNN": {
        "pipeline": Pipeline([
            ("scaler", StandardScaler()),
            ("regressor", KNeighborsRegressor()),
        ]),
        "param_grid": {
            "regressor__n_neighbors": list(range(1, 31)),
            "regressor__weights": ["uniform", "distance"],
        },
    },
}

# ---------------------------------------------------------------------
# 3. Outer cross-validation using the 5 arrays
#    Each array takes a turn as the test set; the other 4 are training data.
#    Inside the training data, GridSearchCV tunes the hyperparameters.
# ---------------------------------------------------------------------
results = {name: {"test_rmse": [], "test_r2": [], "train_rmse": [], "best_params": []}
           for name in models}

print("\n=== Outer cross-validation (5 folds) ===")
for fold in range(NUM_SPLITS):
    test_array = splits[fold]
    train_array = np.vstack([splits[j] for j in range(NUM_SPLITS) if j != fold])
    X_train, y_train = split_xy(train_array)
    X_test, y_test = split_xy(test_array)

    print(f"\nFold {fold + 1}: train on {len(y_train)} samples, test on array {fold + 1} ({len(y_test)} samples)")
    for name, spec in models.items():
        search = GridSearchCV(spec["pipeline"], spec["param_grid"], cv=INNER_CV, scoring=SCORING)
        search.fit(X_train, y_train)

        y_test_pred = search.predict(X_test)
        y_train_pred = search.predict(X_train)

        results[name]["test_rmse"].append(rmse(y_test, y_test_pred))
        results[name]["test_r2"].append(r2_score(y_test, y_test_pred))
        results[name]["train_rmse"].append(rmse(y_train, y_train_pred))
        results[name]["best_params"].append(search.best_params_)

        print(f"  {name:<10} best params: {search.best_params_}  "
              f"train RMSE = {results[name]['train_rmse'][-1]:.4f}  "
              f"test RMSE = {results[name]['test_rmse'][-1]:.4f}  "
              f"test R2 = {results[name]['test_r2'][-1]:.3f}")

print("\n=== Summary: test-set performance across the 5 folds (mean ± SD) ===")
print(f"Target: {TARGET_NAME} (mean = {data[TARGET_NAME].mean():.3f}, SD = {data[TARGET_NAME].std():.3f})")
for name in models:
    r = results[name]
    print(f"{name:<10} train RMSE = {np.mean(r['train_rmse']):.4f} ± {np.std(r['train_rmse']):.4f}   "
          f"test RMSE = {np.mean(r['test_rmse']):.4f} ± {np.std(r['test_rmse']):.4f} d^-1   "
          f"test R2 = {np.mean(r['test_r2']):.3f} ± {np.std(r['test_r2']):.3f}")

# Export the per-fold results and the summary as CSV tables
fold_rows = []
for name in models:
    r = results[name]
    for fold in range(NUM_SPLITS):
        fold_rows.append({
            "Model": name,
            "Test Fold": fold + 1,
            "Best Hyperparameters": str(r["best_params"][fold]),
            "Train RMSE [d^-1]": round(r["train_rmse"][fold], 4),
            "Test RMSE [d^-1]": round(r["test_rmse"][fold], 4),
            "Test R2": round(r["test_r2"][fold], 3),
        })
pd.DataFrame(fold_rows).to_csv("tables/cv_results_by_fold.csv", index=False)

summary_rows = []
for name in models:
    r = results[name]
    summary_rows.append({
        "Model": name,
        "Mean Train RMSE [d^-1]": round(np.mean(r["train_rmse"]), 4),
        "Mean Test RMSE [d^-1]": round(np.mean(r["test_rmse"]), 4),
        "SD Test RMSE [d^-1]": round(np.std(r["test_rmse"]), 4),
        "Mean Test R2": round(np.mean(r["test_r2"]), 3),
        "SD Test R2": round(np.std(r["test_r2"]), 3),
    })
pd.DataFrame(summary_rows).to_csv("tables/cv_summary.csv", index=False)

# Bar chart of test RMSE per fold
fig, ax = plt.subplots(figsize=(7, 4))
bar_width = 0.38
fold_ids = np.arange(1, NUM_SPLITS + 1)
for k, name in enumerate(models):
    ax.bar(fold_ids + (k - 0.5) * bar_width, results[name]["test_rmse"], bar_width,
           label=f"{name} (mean {np.mean(results[name]['test_rmse']):.3f})")
ax.set_xlabel("Test fold (array number)")
ax.set_ylabel(f"Test RMSE [d$^{{-1}}$]")
ax.set_title("Test RMSE by fold")
ax.set_xticks(fold_ids)
ax.legend()
fig.tight_layout()
fig.savefig("figures/fig_test_rmse_by_fold.png", dpi=150)
plt.close(fig)

# ---------------------------------------------------------------------
# 4. Validation curves: training vs validation RMSE across hyperparameters
#    (uses the grid search on all 5 arrays combined)
# ---------------------------------------------------------------------
X_all, y_all = split_xy(np.vstack(splits))

final_models = {}
for name, spec in models.items():
    search = GridSearchCV(spec["pipeline"], spec["param_grid"], cv=INNER_CV,
                          scoring=SCORING, return_train_score=True)
    search.fit(X_all, y_all)
    final_models[name] = search

fig, axes = plt.subplots(1, 2, figsize=(11, 4))

cv_poly = pd.DataFrame(final_models["Polynomial"].cv_results_)
axes[0].plot(cv_poly["param_poly__degree"], -cv_poly["mean_train_score"], "o-", label="Training")
axes[0].plot(cv_poly["param_poly__degree"], -cv_poly["mean_test_score"], "o-", label="Validation")
axes[0].set_xlabel("Polynomial degree")
axes[0].set_ylabel(r"RMSE [d$^{-1}$]")
axes[0].set_title("Polynomial regression")
axes[0].legend()

# Export the hyperparameter search results as CSV tables
pd.DataFrame({
    "Polynomial Degree": cv_poly["param_poly__degree"],
    "Mean Train RMSE [d^-1]": (-cv_poly["mean_train_score"]).round(4),
    "Mean Validation RMSE [d^-1]": (-cv_poly["mean_test_score"]).round(4),
}).to_csv("tables/validation_polynomial.csv", index=False)

cv_knn = pd.DataFrame(final_models["kNN"].cv_results_)
for weights, style in [("uniform", "-"), ("distance", "--")]:
    subset = cv_knn[cv_knn["param_regressor__weights"] == weights]
    axes[1].plot(subset["param_regressor__n_neighbors"], -subset["mean_train_score"], style,
                 label=f"Training ({weights})")
    axes[1].plot(subset["param_regressor__n_neighbors"], -subset["mean_test_score"], style,
                 label=f"Validation ({weights})")
axes[1].set_xlabel("k (n_neighbors)")
axes[1].set_ylabel(r"RMSE [d$^{-1}$]")
axes[1].set_title("k-nearest neighbors")
axes[1].legend(fontsize=8)

pd.DataFrame({
    "k (n_neighbors)": cv_knn["param_regressor__n_neighbors"],
    "Weights": cv_knn["param_regressor__weights"],
    "Mean Train RMSE [d^-1]": (-cv_knn["mean_train_score"]).round(4) + 0.0,
    "Mean Validation RMSE [d^-1]": (-cv_knn["mean_test_score"]).round(4),
}).to_csv("tables/validation_knn.csv", index=False)

fig.tight_layout()
fig.savefig("figures/fig_validation_curves.png", dpi=150)
plt.close(fig)

# ---------------------------------------------------------------------
# 5. Learning curves: does adding more training data help?
# ---------------------------------------------------------------------
train_sizes = np.linspace(0.1, 1.0, 8)
fig, ax = plt.subplots(figsize=(7, 4))
for name, search in final_models.items():
    sizes, train_scores, test_scores = learning_curve(
        search.best_estimator_, X_all, y_all, train_sizes=train_sizes,
        cv=KFold(n_splits=5, shuffle=True, random_state=0), scoring=SCORING)
    ax.plot(sizes, -test_scores.mean(axis=1), "o-", label=f"{name} (validation)")
    ax.plot(sizes, -train_scores.mean(axis=1), "o--", alpha=0.6, label=f"{name} (training)")
ax.set_xlabel("Number of training samples")
ax.set_ylabel(r"RMSE [d$^{-1}$]")
ax.set_title("Learning curves (best hyperparameters)")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig("figures/fig_learning_curves.png", dpi=150)
plt.close(fig)

# ---------------------------------------------------------------------
# 6. Interpretation
# ---------------------------------------------------------------------
print(f"\n=== Final models (tuned on all {len(y_all)} samples) ===")
for name, search in final_models.items():
    print(f"{name:<10} best params: {search.best_params_}  validation RMSE = {-search.best_score_:.4f}")

# 6a. Polynomial weights (on standardized features, so magnitudes are comparable)
poly_model = final_models["Polynomial"].best_estimator_
term_names = poly_model.named_steps["poly"].get_feature_names_out(
    ["Temperature", "pH", "Glucose", "Lactate"])
regressor = poly_model.named_steps["regressor"]
print(f"\nPolynomial model (standardized features):")
print(f"Growth Rate = {regressor.intercept_:.4f}")
order = np.argsort(-np.abs(regressor.coef_))
for idx in order:
    print(f"   {regressor.coef_[idx]:+.4f} * [{term_names[idx].replace(' ', '] * [')}]")

pd.DataFrame({
    "Term": ["Intercept"] + [term_names[idx].replace(" ", " x ") for idx in order],
    "Coefficient (standardized features)": [round(regressor.intercept_, 4)]
                                           + [round(regressor.coef_[idx], 4) for idx in order],
}).to_csv("tables/polynomial_coefficients.csv", index=False)

# Predict growth across a grid of conditions spanning the observed data range,
# then report the conditions with the highest and lowest predicted growth rates
grid_axes = [np.linspace(X_all[:, j].min(), X_all[:, j].max(), 25) for j in range(X_all.shape[1])]
X_cond = np.array(np.meshgrid(*grid_axes)).reshape(X_all.shape[1], -1).T
y_cond = poly_model.predict(X_cond)
extremes = pd.DataFrame(X_cond, columns=FEATURE_NAMES)
extremes[f"Predicted {TARGET_NAME}"] = y_cond
extremes = extremes.sort_values(f"Predicted {TARGET_NAME}", ascending=False)
pd.concat([extremes.head(5).assign(Condition="Highest"),
           extremes.tail(5).assign(Condition="Lowest")]) \
    .round(3).to_csv("tables/predicted_extreme_conditions.csv", index=False)

# 6b. Partial dependence plots: vary one feature, hold the others at their means
feature_means = X_all.mean(axis=0)
fig, axes = plt.subplots(1, 4, figsize=(15, 3.8), sharey=True)
for j, ax in enumerate(axes):
    grid = np.linspace(X_all[:, j].min(), X_all[:, j].max(), 200)
    X_grid = np.tile(feature_means, (len(grid), 1))
    X_grid[:, j] = grid
    ax.scatter(X_all[:, j], y_all, s=8, color="lightgray", label="Data")
    for name, search in final_models.items():
        ax.plot(grid, search.predict(X_grid), label=name)
    ax.set_xlabel(FEATURE_NAMES[j])
    if j == 0:
        ax.set_ylabel(r"Specific growth rate [d$^{-1}$]")
axes[0].legend(fontsize=8)
fig.suptitle("Predicted growth rate vs each factor (other factors held at their means)")
fig.tight_layout()
fig.savefig("figures/fig_partial_dependence.png", dpi=150)
plt.close(fig)

# 6c. Parity plot for the outer-CV test predictions of each model
fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharey=True)
for ax, name in zip(axes, models):
    y_true_all, y_pred_all = [], []
    for fold in range(NUM_SPLITS):
        train_array = np.vstack([splits[j] for j in range(NUM_SPLITS) if j != fold])
        X_train, y_train = split_xy(train_array)
        X_test, y_test = split_xy(splits[fold])
        model = models[name]["pipeline"].set_params(**results[name]["best_params"][fold])
        model.fit(X_train, y_train)
        y_true_all.append(y_test)
        y_pred_all.append(model.predict(X_test))
    y_true_all = np.concatenate(y_true_all)
    y_pred_all = np.concatenate(y_pred_all)
    ax.scatter(y_true_all, y_pred_all, s=10)
    lims = [min(y_true_all.min(), y_pred_all.min()), max(y_true_all.max(), y_pred_all.max())]
    ax.plot(lims, lims, "k--", linewidth=1)
    ax.set_xlabel(r"Actual growth rate [d$^{-1}$]")
    ax.set_title(f"{name}: test RMSE = {rmse(y_true_all, y_pred_all):.3f}")
axes[0].set_ylabel(r"Predicted growth rate [d$^{-1}$]")
fig.tight_layout()
fig.savefig("figures/fig_parity.png", dpi=150)
plt.close(fig)

print("\nFigures saved to figures/: fig_test_rmse_by_fold.png, fig_validation_curves.png, "
      "fig_learning_curves.png, fig_partial_dependence.png, fig_parity.png")
print("Tables saved to tables/: cv_results_by_fold.csv, cv_summary.csv, validation_polynomial.csv, "
      "validation_knn.csv, polynomial_coefficients.csv, predicted_extreme_conditions.csv")