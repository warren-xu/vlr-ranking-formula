# Reverse engineering VLR player rating
# Purpose: To reverse engineer the VLR.gg player rating calculation using their stat data.
# File: analyze_data.py - trains and tests constrained and unconstrained models.
# Warren Xu

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import lsq_linear
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold

# SR is left out on purpose: SR = 1 - DPR exactly, so the two are perfectly collinear
# and a regression can only split one effect arbitrarily between them.
feature_keys = ["KPR", "APR", "DPR", "ADRa", "KAST"]
extended_keys = feature_keys + ["FKPR", "FDPR"]
target_key = "R2.0"

# Encode known trend constraints from the rating update:
# 1) APR effect decreased from original 0.228
# 2) KAST effect decreased from original 0.313 but must remain positive
# Additional sign priors keep coefficients directionally stable.
bounds = {
    "KPR": (0.0, np.inf),
    "APR": (0.0, 0.228),
    "DPR": (-np.inf, 0.0),
    "ADRa": (0.0, np.inf),
    "KAST": (0.0001, 0.313),
    "FKPR": (-np.inf, np.inf),
    "FDPR": (-np.inf, np.inf),
}


class ConstrainedLinearRegression:
    """Least squares with per-coefficient bounds (sklearn-style fit/predict)."""

    def __init__(self, keys):
        self.keys = keys

    def fit(self, X, y):
        X_aug = np.column_stack([np.asarray(X), np.ones(len(X))])
        lower = [bounds[k][0] for k in self.keys] + [-np.inf]
        upper = [bounds[k][1] for k in self.keys] + [np.inf]
        result = lsq_linear(X_aug, np.asarray(y), bounds=(lower, upper))
        if not result.success:
            raise RuntimeError(f"Constrained optimization failed: {result.message}")
        self.coef_, self.intercept_ = result.x[:-1], result.x[-1]
        return self

    def predict(self, X):
        return np.asarray(X) @ self.coef_ + self.intercept_


def metrics(y_true, y_pred):
    return {
        "R2": r2_score(y_true, y_pred),
        "RMSE": mean_squared_error(y_true, y_pred) ** 0.5,
        "MAE": mean_absolute_error(y_true, y_pred),
        "within_0.05": np.mean(np.abs(y_true - y_pred) <= 0.05),
    }


def cross_validate(make_model, X, y, folds=10):
    # Rows are sorted by rating, so folds must be shuffled or each one covers a narrow band.
    scores = []
    for train_idx, test_idx in KFold(folds, shuffle=True, random_state=0).split(X):
        model = make_model().fit(X.iloc[train_idx], y.iloc[train_idx])
        scores.append(r2_score(y.iloc[test_idx], model.predict(X.iloc[test_idx])))
    return np.mean(scores), np.std(scores)


def print_row(name, cv, test):
    print(
        f"{name:<26} CV R2 {cv[0]:.3f} ± {cv[1]:.3f} | "
        f"holdout R2 {test['R2']:.3f}  RMSE {test['RMSE']:.4f}  MAE {test['MAE']:.4f}  "
        f"within ±0.05 {test['within_0.05']:.0%}"
    )


# Main training set: all-time VCT stats (largest sample, most stable coefficients).
train = pd.read_csv("data/clean/stats_alltime.csv")

# Out-of-time check: fit on the last 90 days and predict Champions 2023. The two share
# no matches, so this tests whether the formula holds across years and patches.
recent = pd.read_csv("data/clean/stats_recent.csv")
holdout = pd.read_csv("data/clean/champions2023.csv")

models = {
    "Unconstrained": lambda keys: LinearRegression(),
    "Constrained": lambda keys: ConstrainedLinearRegression(keys),
}

print(f"Train: stats_alltime ({len(train)} players) | holdout: recent ({len(recent)}) -> champions2023 ({len(holdout)})\n")
results = {}
for keys, label in [(feature_keys, "5 features"), (extended_keys, "+ FKPR, FDPR")]:
    for model_name, make_model in models.items():
        name = f"{model_name} ({label})"
        cv = cross_validate(lambda: make_model(keys), train[keys], train[target_key])
        holdout_model = make_model(keys).fit(recent[keys], recent[target_key])
        test = metrics(holdout[target_key], holdout_model.predict(holdout[keys]))
        print_row(name, cv, test)

        final = make_model(keys).fit(train[keys], train[target_key])
        results[name] = (keys, final, holdout_model)

baseline = np.full(len(holdout), recent[target_key].mean())
print(f"{'Baseline (predict mean)':<26} holdout MAE {mean_absolute_error(holdout[target_key], baseline):.4f}\n")

for name, (keys, model, _) in results.items():
    coefs = ", ".join(f"{k}={c:.4f}" for k, c in zip(keys, model.coef_))
    print(f"{name}: {coefs}, intercept={model.intercept_:.4f}")

# Save the headline formula so check_formula.py always uses the latest fit.
keys, model, holdout_model = results["Unconstrained (5 features)"]
with open("data/formula.json", "w") as f:
    json.dump(
        {
            "trained_on": "data/clean/stats_alltime.csv",
            "features": keys,
            "coefficients": dict(zip(keys, model.coef_.round(6).tolist())),
            "intercept": round(float(model.intercept_), 6),
        },
        f,
        indent=2,
    )
print("\nSaved formula to data/formula.json")

# Predicted vs. actual on the out-of-time holdout.
predicted = holdout_model.predict(holdout[keys])
actual = holdout[target_key]
test = metrics(actual, predicted)

os.makedirs("docs", exist_ok=True)
fig, ax = plt.subplots(figsize=(6, 6), dpi=150)
lims = [min(actual.min(), predicted.min()) - 0.05, max(actual.max(), predicted.max()) + 0.05]
ax.fill_between(lims, [l - 0.05 for l in lims], [l + 0.05 for l in lims], color="#2a78d6", alpha=0.08, lw=0)
ax.plot(lims, lims, color="#8c8b87", lw=1, ls="--", zorder=1)
ax.scatter(actual, predicted, s=36, color="#2a78d6", edgecolor="white", linewidth=1, zorder=2)
ax.text(lims[1] - 0.02, lims[1] - 0.075, "±0.05", color="#52514e", fontsize=9, ha="right")
ax.set(xlim=lims, ylim=lims, aspect="equal")
ax.set_xlabel("Actual VLR rating", color="#52514e")
ax.set_ylabel("Predicted rating", color="#52514e")
ax.set_title(
    "Champions 2023, predicted from a formula fit on 2026 data\n"
    f"R² {test['R2']:.3f} · MAE {test['MAE']:.3f} · {test['within_0.05']:.0%} within ±0.05",
    fontsize=10,
    color="#0b0b0b",
    loc="left",
)
ax.grid(color="#e6e5e0", lw=0.6)
ax.set_axisbelow(True)
for side in ["top", "right"]:
    ax.spines[side].set_visible(False)
for side in ["left", "bottom"]:
    ax.spines[side].set_color("#c3c2b7")
ax.tick_params(colors="#52514e", labelsize=8)
fig.tight_layout()
fig.savefig("docs/predicted_vs_actual.png", facecolor="white")
print("Saved plot to docs/predicted_vs_actual.png")
