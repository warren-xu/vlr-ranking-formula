# Reverse engineering VLR player rating
# Purpose: To reverse engineer the VLR.gg player rating calculation using their stat data.
# File: analyze_data.py - trains and tests constrained and unconstrained models.
# Mark Zhdan | 01-05-2023

import numpy as np
import pandas as pd
from scipy.optimize import lsq_linear
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error


def print_metrics(model_name, y_true, y_pred):
    print(f"{model_name} R2 score: {r2_score(y_true, y_pred)}")
    print(f"{model_name} RMSE: {mean_squared_error(y_true, y_pred, squared=False)}")
    print(f"{model_name} MAE: {mean_absolute_error(y_true, y_pred)}")


def fit_constrained_linear_regression(X_train, y_train):
    # Encode known trend constraints from the rating update:
    # 1) SR included in the formula and constrained to a positive effect
    # 2) APR effect decreased from original 0.228
    # 3) KAST effect decreased from original 0.313 but must remain positive
    # Additional sign priors keep coefficients directionally stable.
    # Feature order: KPR, APR, DPR, ADRa, SR, KAST, intercept
    X_aug = np.column_stack([X_train.to_numpy(), np.ones(len(X_train))])

    lower_bounds = np.array([0.0, 0.0, -np.inf, 0.0, 0.0, 0.0001, -np.inf])
    upper_bounds = np.array([np.inf, 0.228, 0.0, np.inf, np.inf, 0.313, np.inf])

    result = lsq_linear(X_aug, y_train.to_numpy(), bounds=(lower_bounds, upper_bounds))
    if not result.success:
        raise RuntimeError(f"Constrained optimization failed: {result.message}")

    coefficients = result.x[:-1]
    intercept = result.x[-1]
    return coefficients, intercept


# stats_alltime gives more stable coefficient estimates than small event-only samples.
df = pd.read_csv("data/clean/stats_recent.csv")

# SR added back.
feature_keys = ["KPR", "APR", "DPR", "ADRa", "SR", "KAST"]
features = df[feature_keys]
target = df["R2.0"]

X_train, X_test, y_train, y_test = train_test_split(
    features, target, random_state=3, test_size=0.1
)

# Baseline unconstrained model.
unconstrained_model = LinearRegression()
unconstrained_model.fit(X_train, y_train)
unconstrained_predictions = unconstrained_model.predict(X_test)

# Constrained model with trend priors.
constrained_coefs, constrained_intercept = fit_constrained_linear_regression(X_train, y_train)
constrained_predictions = X_test.to_numpy() @ constrained_coefs + constrained_intercept

print("Unconstrained coefficients:")
formatted_unconstrained = ", ".join(f"{coef:.12f}" for coef in unconstrained_model.coef_)
print(f"[{formatted_unconstrained}]")
print(f"Unconstrained intercept: {unconstrained_model.intercept_}\n")

print("Constrained coefficients:")
formatted_constrained = ", ".join(f"{coef:.12f}" for coef in constrained_coefs)
print(f"[{formatted_constrained}]")
print(f"Constrained intercept: {constrained_intercept}\n")

print_metrics("Unconstrained", y_test, unconstrained_predictions)
print("")
print_metrics("Constrained", y_test, constrained_predictions)
