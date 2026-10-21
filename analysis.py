"""STL labelling experiment: class-balanced LightGBM on sliding-window features
vs. an sklearn MLP on raw signals. Seeds 42/43/44; each seed simulates its own
signals, splits once (stratified), tunes by CV on the training part only, and
scores the held-out test part once per model."""
import json, sys, warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score
from sklearn.exceptions import ConvergenceWarning
from lightgbm import LGBMClassifier

warnings.filterwarnings("ignore", category=ConvergenceWarning)
SEEDS = [42, 43, 44]
N_SIGNALS = 3000      # signals per seed (small enough for a fast CI run)
T = 100               # trace length
TEST_SIZE = 0.25
CV_FOLDS = 3


def simulate(n, seed):
    """Random-walk traces squashed into (0,1); exact labels from the fixed STL formula
    'eventually in [0,20] x>0.6 and always in [0,50] x<0.95'."""
    rng = np.random.default_rng(seed)
    start = rng.normal(0, 1, size=(n, 1))
    walk = start + np.cumsum(rng.normal(0, 0.15, size=(n, T)), axis=1)
    x = 1 / (1 + np.exp(-walk))
    ev = (x[:, 0:21] > 0.6).any(axis=1)     # eventually in [0,20]: x > 0.6
    alw = (x[:, 0:51] < 0.95).all(axis=1)   # always in [0,50]: x < 0.95
    return x, (ev & alw).astype(int)


def raw_features(x):
    """Baseline input: the raw signal itself."""
    return x


def window_features(x, W=20, stride=10):
    """STL-style features: per sliding window max/min/mean and the normalised time of
    the first crossing of 0.6 and 0.95 (1.0 if never crossed), plus prefix extrema
    matching the formula horizons."""
    feats = []
    for s in range(0, T - W + 1, stride):
        w = x[:, s:s + W]
        feats += [w.max(1), w.min(1), w.mean(1)]
        for thr in (0.6, 0.95):
            above = w > thr
            feats.append(np.where(above.any(1), above.argmax(1), W) / W)
    feats += [x[:, :21].max(1), x[:, :51].max(1), x[:, :51].min(1)]
    return np.column_stack(feats)


def fit_eval(model, grid, X, y, tr, te, cv):
    """Shared evaluation: CV grid search on train only, then one test-set scoring."""
    gs = GridSearchCV(model, grid, cv=cv, scoring="accuracy")
    gs.fit(X[tr], y[tr])
    pred = gs.predict(X[te])
    return accuracy_score(y[te], pred), f1_score(y[te], pred, average="macro"), gs.best_params_


def run_seed(seed):
    x, y = simulate(N_SIGNALS, seed)
    tr, te = train_test_split(np.arange(N_SIGNALS), test_size=TEST_SIZE,
                              stratify=y, random_state=seed)
    cv = StratifiedKFold(CV_FOLDS, shuffle=True, random_state=seed)  # same folds for both
    print(f"seed {seed}: positive rate={y.mean():.3f}, train={len(tr)}, test={len(te)}")

    # Baseline: standard MLP on raw signal, small grid chosen by CV
    base = make_pipeline(StandardScaler(), MLPClassifier(max_iter=300, random_state=seed))
    base_grid = {"mlpclassifier__learning_rate_init": [1e-3, 1e-2],
                 "mlpclassifier__hidden_layer_sizes": [(32,), (64,)]}
    b_acc, b_f1, b_p = fit_eval(base, base_grid, raw_features(x), y, tr, te, cv)
    print(f"  baseline MLP  acc={b_acc:.4f} f1={b_f1:.4f} {b_p}")

    # Improved: class-balanced LightGBM on window features, CV-tuned
    imp = LGBMClassifier(class_weight="balanced", n_estimators=200,
                         random_state=seed, verbose=-1)
    imp_grid = {"num_leaves": [15, 31], "learning_rate": [0.05, 0.1]}
    i_acc, i_f1, i_p = fit_eval(imp, imp_grid, window_features(x), y, tr, te, cv)
    print(f"  improved LGBM acc={i_acc:.4f} f1={i_f1:.4f} {i_p}")
    return dict(b_acc=b_acc, b_f1=b_f1, i_acc=i_acc, i_f1=i_f1,
                n_train=len(tr), n_test=len(te))


def main():
    res = [run_seed(s) for s in SEEDS]
    col = lambda k: np.array([r[k] for r in res])
    b, i = col("b_acc"), col("i_acc")
    metrics = {
        "baseline_score": float(b.mean()),
        "improved_score": float(i.mean()),
        "improvement": float(i.mean() - b.mean()),
        "baseline_score_std": float(b.std()),
        "improved_score_std": float(i.std()),
        "baseline_f1": float(col("b_f1").mean()),
        "improved_f1": float(col("i_f1").mean()),
    }
    if not all(np.isfinite(v) for v in metrics.values()):
        print("Non-finite metrics; exiting.")
        sys.exit(1)

    # Optional chart: mean test accuracy with std across seeds
    import os
    os.makedirs("figures", exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(["MLP (raw)", "LGBM (window)"], [b.mean(), i.mean()],
           yerr=[b.std(), i.std()], capsize=6, color=["gray", "tab:blue"])
    ax.set_ylabel("Test accuracy (mean ± std, seeds 42-44)")
    ax.set_title("STL labelling: baseline vs improved")
    fig.tight_layout()
    fig.savefig("figures/accuracy_bar.png", dpi=120)

    results = {
        "question": ("Can a class-balanced, CV-tuned gradient-boosting model that uses "
                     "STL-style window features match or beat a neural sequence baseline at "
                     "labelling signals with a known temporal logic formula, and does it do so "
                     "more stably across seeds?"),
        "method": ("Seeded random-walk signals (length 100) labelled by the fixed STL formula "
                   "'eventually in [0,20] x>0.6 and always in [0,50] x<0.95'. Per seed: one "
                   "stratified train/test split; CV grid search on train only; test scored once."),
        "dataset": {
            "name": "Simulated random-walk signals with exact STL labels",
            "source_url": "numpy seeded simulation (no download)",
            "n_rows": N_SIGNALS,
            "n_train": res[0]["n_train"],
            "n_test": res[0]["n_test"],
            "same_as_paper": False,
            "baseline_is_papers_method": False,
        },
        "metric_name": "test accuracy",
        "higher_is_better": True,
        "baseline": ("sklearn MLPClassifier on the raw 100-point signal (StandardScaler), "
                     "grid over learning rate and hidden size chosen by 3-fold CV; stands in "
                     "for the paper's recurrent network."),
        "improved": ("Class-balanced LightGBM on sliding-window max/min/mean and threshold-"
                     "crossing-time features plus formula-horizon prefix extrema; grid over "
                     "num_leaves and learning rate chosen by 3-fold CV."),
        "metrics": metrics,
        "notes": ("Simulated data only, so results do not transfer to the paper's own "
                  "benchmark. Only 3 seeds and small grids; std is across seeds. The formula is "
                  "fixed and known, which favours hand-crafted window features. Macro-F1 is a "
                  "secondary metric. Per-seed positive rate is printed in the log."),
    }
    with open("results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print("Wrote results.json")


if __name__ == "__main__":
    main()
