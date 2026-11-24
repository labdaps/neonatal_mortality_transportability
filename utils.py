# Functions shared by the notebooks: reading the raw national file, models and hyperparameter
# search, calibration, performance metrics, model comparison tests, the per-municipality external
# evaluation, SHAP importance, subgroups and death-timing windows, and saving tables and figures.
# The cohort construction is written out step by step in 01_data.ipynb. SEED fixes every stochastic step.

import os
import numpy as np
import pandas as pd

SEED = 42

# ----------------------------------------------------------------------------------
# Paths (relative to this file)
# ----------------------------------------------------------------------------------

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "raw")                                  # raw SINASC-SIM linkage files and the IBGE population file
DATA = os.path.join(HERE, "data")                                # built cohorts (joblib)
MODELS = os.path.join(HERE, "models")                            # hyperparameter searches, fitted models, calibrators, thresholds
CACHE = os.path.join(HERE, "cache")                              # heavy intermediate results
FIGURES = os.path.join(HERE, "results", "figures")               # main-text figures
TABLES = os.path.join(HERE, "results", "tables")                 # main-text tables
SUPPLEMENT_FIGURES = os.path.join(HERE, "results", "supplement_figures")
SUPPLEMENT_TABLES = os.path.join(HERE, "results", "supplement_tables")

# ----------------------------------------------------------------------------------
# Study constants (predictors, cohorts, coding)
# ----------------------------------------------------------------------------------

SP_CODE = 355030                  # Sao Paulo city municipality code (CODMUNNASC)

# The eight predictors (WHO minimum perinatal indicator set plus extras)
CONTINUOUS = ["IDADEMAE", "PESO", "SEMAGESTAC", "APGAR1", "APGAR5"]
CATEGORICAL = ["LOCNASC", "IDANOMAL", "PARTO"]
FEATURES = CONTINUOUS + CATEGORICAL

# Kept for subgroup/fairness analysis only (never model inputs)
SUBGROUP_VARS = ["RACACORMAE", "ESCMAE2010"]

# The four models (short key -> label used in tables and figures)
MODELS_ORDER = [("catboost", "CatBoost"), ("xgboost", "XGBoost"),
                ("lightgbm", "LightGBM"), ("logistic_regression", "Logistic regression")]
REFERENCE_MODEL = "logistic_regression"     # baseline for the model-comparison tests

# model colours used in every figure (colourblind-safe palette)
MODEL_COLORS = {"CatBoost": "#0072B2", "XGBoost": "#D55E00",
                "LightGBM": "#009E73", "Logistic regression": "#CC79A7"}

# predictor labels for the SHAP figures
SHAP_LABELS = {"IDADEMAE": "Age of the mother", "PESO": "Birth weight", "SEMAGESTAC": "Gestational age",
               "LOCNASC": "Place of delivery", "APGAR1": "Apgar 1st minute", "APGAR5": "Apgar 5th minute",
               "IDANOMAL": "Congenital anomaly", "PARTO": "Mode of delivery"}

# Death-timing categories: within the first day, 1-7 days, 8-28 days
DEATH_TIMING_ORDER = ["<1 day", "1-7 days", "8-28 days"]


# ----------------------------------------------------------------------------------
# Reading the raw national file (.DBF format, which pandas cannot read directly)
# ----------------------------------------------------------------------------------
# read_dbf streams selected columns out of the .DBF file, so the national file fits in memory.

def _dbf_meta(path):
    # Parse a DBF header: returns (n_records, header_len, record_len, [(name, offset, length), ...]).
    import struct
    with open(path, "rb") as fh:
        h = fh.read(32)
        nrec = struct.unpack("<I", h[4:8])[0]
        hlen = struct.unpack("<H", h[8:10])[0]
        rlen = struct.unpack("<H", h[10:12])[0]
        fields, off = [], 1
        fh.seek(32)
        while True:
            d = fh.read(32)
            if d[:1] == b"\r" or len(d) < 32:
                break
            name = d[0:11].split(b"\x00")[0].decode("latin1")
            flen = d[16]
            fields.append((name, off, flen))
            off += flen
    return nrec, hlen, rlen, fields


def read_dbf(path, cols, chunk=400_000):
    # Stream selected columns out of a large DBF, dependency-free and memory-light.
    nrec, hlen, rlen, fields = _dbf_meta(path)
    fmap = {n: (o, l) for n, o, l in fields}
    missing = [c for c in cols if c not in fmap]
    if missing:
        raise KeyError("DBF " + os.path.basename(path) + " missing fields: " + str(missing))
    dt = np.dtype(dict(names=list(cols),
                       formats=["S" + str(fmap[c][1]) for c in cols],
                       offsets=[fmap[c][0] for c in cols],
                       itemsize=rlen))
    parts = {c: [] for c in cols}
    with open(path, "rb") as fh:
        fh.seek(hlen)
        read = 0
        while read < nrec:
            buf = fh.read(min(chunk, nrec - read) * rlen)
            if not buf:
                break
            a = np.frombuffer(buf, dtype=dt, count=len(buf) // rlen)
            for c in cols:
                parts[c].append(np.char.strip(a[c].astype("U")))
            read += len(buf) // rlen
    return pd.DataFrame({c: np.concatenate(parts[c]) for c in cols})


# ----------------------------------------------------------------------------------
# Models and hyperparameter search
# ----------------------------------------------------------------------------------

USE_GPU = False    # set True on a machine with a CUDA GPU (XGBoost/CatBoost only)


def search_spaces():
    # Hyperparameter search space for each gradient-boosting model; the default is inside each space.
    from hyperopt import hp
    return {
        "xgboost": {
            "n_estimators": hp.choice("xgb_n", [100, 150, 200, 250, 300]),
            "max_depth": hp.choice("xgb_d", [3, 4, 5, 6, 7, 8]),
            "learning_rate": hp.uniform("xgb_lr", 0.01, 0.3),
            "subsample": hp.uniform("xgb_ss", 0.5, 1.0),
            "colsample_bytree": hp.uniform("xgb_cs", 0.5, 1.0),
            "min_child_weight": hp.choice("xgb_mcw", [1, 3, 5, 10]),
            "reg_lambda": hp.uniform("xgb_l2", 0.0, 5.0),
        },
        "lightgbm": {
            "n_estimators": hp.choice("lgb_n", [100, 150, 200, 250, 300]),
            "num_leaves": hp.choice("lgb_nl", [15, 31, 63, 127]),
            "learning_rate": hp.uniform("lgb_lr", 0.01, 0.3),
            "min_child_samples": hp.choice("lgb_mcs", [20, 50, 100, 200]),
            "colsample_bytree": hp.uniform("lgb_cs", 0.5, 1.0),
            "reg_lambda": hp.uniform("lgb_l2", 0.0, 5.0),
        },
        "catboost": {
            "iterations": hp.choice("cat_it", [100, 150, 200, 250, 300, 1000]),
            "depth": hp.choice("cat_d", [4, 5, 6, 7, 8]),
            "learning_rate": hp.uniform("cat_lr", 0.01, 0.3),
            "l2_leaf_reg": hp.quniform("cat_l2", 1, 10, 1),
            "border_count": hp.choice("cat_bc", [32, 64, 128, 254]),
        },
    }


def make_model(model_name, params, seed=SEED):
    # Build one model. The three gradient-boosting models take tuned params; the logistic regression
    # is the classical (unpenalised) maximum-likelihood baseline and takes no params.
    if model_name == "xgboost":
        from xgboost import XGBClassifier
        return XGBClassifier(**params, objective="binary:logistic", n_jobs=-1, random_state=seed,
                             tree_method="hist", device="cuda" if USE_GPU else "cpu")
    if model_name == "lightgbm":
        import lightgbm as lgb
        return lgb.LGBMClassifier(**params, n_jobs=-1, random_state=seed,
                                  deterministic=True, force_row_wise=True, verbose=-1)
    if model_name == "catboost":
        from catboost import CatBoostClassifier
        return CatBoostClassifier(**params, random_seed=seed, thread_count=-1, verbose=False,
                                  allow_writing_files=False, task_type="GPU" if USE_GPU else "CPU")
    if model_name == "logistic_regression":
        from sklearn.linear_model import LogisticRegression
        return LogisticRegression(penalty=None, solver="lbfgs", max_iter=5000, random_state=seed)
    raise ValueError("unknown model_name: " + model_name)


def _categorical_to_text(frame):
    # Turn category codes into text while keeping missing values missing (so the imputer can fill them).
    frame = pd.DataFrame(frame)
    return frame.apply(lambda column: column.map(lambda v: np.nan if pd.isna(v) else str(v)))


def make_preprocessor(continuous, categorical):
    # Continuous: fill missing with the median, then standardise.
    # Categorical: fill missing with an explicit "Missing" category, then one-hot encode.
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
    continuous_steps = Pipeline([
        ("fill_missing", SimpleImputer(strategy="median")),
        ("standardise", StandardScaler()),
    ])
    categorical_steps = Pipeline([
        ("as_text", FunctionTransformer(_categorical_to_text, feature_names_out="one-to-one")),
        ("fill_missing", SimpleImputer(strategy="constant", fill_value="Missing")),
        ("one_hot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    blocks = []
    if continuous:
        blocks.append(("continuous", continuous_steps, list(continuous)))
    if categorical:
        blocks.append(("categorical", categorical_steps, list(categorical)))
    return ColumnTransformer(blocks, remainder="drop")


def make_pipeline(model_name, params, continuous, categorical, seed=SEED):
    # Preprocessing plus the model as a single fitted object, so the same preparation is applied to
    # training and to every test set with no risk of a mismatch.
    from sklearn.pipeline import Pipeline
    return Pipeline([
        ("preprocessor", make_preprocessor(continuous, categorical)),
        ("model", make_model(model_name, params, seed)),
    ])


def tune_hyperparameters(model_name, X, y, continuous, categorical,
                         n_evaluations=100, n_folds=10, seed=SEED):
    # Bayesian search (Tree Parzen Estimator) maximising cross-validated F1. Inside cross-validation
    # the preprocessor is refit on each training fold, so the validation fold never leaks into the
    # imputation/scaling. Returns the best parameters AND every evaluation (nothing is discarded).
    from hyperopt import fmin, tpe, Trials, space_eval
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    space = search_spaces()[model_name]
    folds = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

    def objective(params):
        pipeline = make_pipeline(model_name, params, continuous, categorical, seed)
        f1 = cross_val_score(pipeline, X, y, cv=folds, scoring="f1", n_jobs=1).mean()
        return 1 - f1

    trials = Trials()
    best = fmin(objective, space=space, algo=tpe.suggest, max_evals=n_evaluations,
                trials=trials, rstate=np.random.default_rng(seed))
    best_parameters = space_eval(space, best)
    history = trials_to_dataframe(trials, space)
    return {"best_parameters": best_parameters, "trials": history}


def trials_to_dataframe(trials, space):
    # One row per hyperopt evaluation: the parameters tried and the mean cross-validated F1.
    from hyperopt import space_eval
    rows = []
    for position, trial in enumerate(trials.trials):
        sampled = {name: values[0] for name, values in trial["misc"]["vals"].items() if len(values) > 0}
        parameters = space_eval(space, sampled)
        loss = trial["result"].get("loss")
        row = {"evaluation": position + 1}
        row.update(parameters)
        row["mean_cross_validated_f1"] = None if loss is None else 1 - loss
        rows.append(row)
    return pd.DataFrame(rows).sort_values("mean_cross_validated_f1", ascending=False).reset_index(drop=True)


# ----------------------------------------------------------------------------------
# Calibration
# ----------------------------------------------------------------------------------

class SigmoidCalibrator:
    def __init__(self, logistic):
        self.logistic = logistic

    def predict(self, probability):
        probability = np.clip(np.asarray(probability, float), 1e-6, 1 - 1e-6)
        logit = np.log(probability / (1 - probability)).reshape(-1, 1)
        return self.logistic.predict_proba(logit)[:, 1]


def fit_sigmoid_calibrator(probability, y):
    from sklearn.linear_model import LogisticRegression
    probability = np.clip(np.asarray(probability, float), 1e-6, 1 - 1e-6)
    logit = np.log(probability / (1 - probability)).reshape(-1, 1)
    return SigmoidCalibrator(LogisticRegression().fit(logit, np.asarray(y).ravel()))


class IsotonicCalibrator:
    def __init__(self, isotonic):
        self.isotonic = isotonic

    def predict(self, probability):
        return self.isotonic.predict(np.asarray(probability, float))


def fit_isotonic_calibrator(probability, y):
    from sklearn.isotonic import IsotonicRegression
    isotonic = IsotonicRegression(out_of_bounds="clip")
    isotonic.fit(np.asarray(probability, float), np.asarray(y).ravel())
    return IsotonicCalibrator(isotonic)


def calibration_curve_points(y_true, probability, n_bins=10):
    from sklearn.calibration import calibration_curve
    fraction_positive, mean_predicted = calibration_curve(
        y_true, probability, n_bins=n_bins, strategy="quantile")
    return pd.DataFrame({"mean_predicted_probability": mean_predicted,
                         "observed_fraction_positive": fraction_positive})


def calibration_in_the_large_and_slope(y_true, probability):
    # Calibration-in-the-large is the intercept of a logistic model with the predicted logit as an
    # offset; the calibration slope is the coefficient of the predicted logit. Both are fit here with
    # a few Newton steps so the computation stays memory-light on the very large external cohort.
    y = np.asarray(y_true).astype(float)
    probability = np.clip(np.asarray(probability, float), 1e-8, 1 - 1e-8)
    logit = np.log(probability / (1 - probability))

    alpha = 0.0
    for _ in range(100):
        p = 1.0 / (1.0 + np.exp(-(alpha + logit)))
        hessian = np.sum(p * (1.0 - p))
        if hessian < 1e-12:
            break
        stepsize = np.sum(y - p) / hessian
        alpha += stepsize
        if abs(stepsize) < 1e-10:
            break

    intercept, slope = 0.0, 1.0
    for _ in range(100):
        p = 1.0 / (1.0 + np.exp(-(intercept + slope * logit)))
        weight = p * (1.0 - p)
        residual = y - p
        score = np.array([np.sum(residual), np.sum(logit * residual)])
        sum_w = np.sum(weight)
        sum_wl = np.sum(weight * logit)
        sum_wll = np.sum(weight * logit * logit)
        hessian = np.array([[sum_w, sum_wl], [sum_wl, sum_wll]])
        try:
            delta = np.linalg.solve(hessian, score)
        except np.linalg.LinAlgError:
            break
        intercept += delta[0]
        slope += delta[1]
        if np.max(np.abs(delta)) < 1e-10:
            break

    return {"calibration_in_the_large": float(alpha), "calibration_slope": float(slope)}


def scaled_brier_score(y_true, probability):
    from sklearn.metrics import brier_score_loss
    y = np.asarray(y_true).astype(int)
    prevalence = y.mean()
    reference = brier_score_loss(y, np.full_like(y, prevalence, dtype=float))
    return 1 - brier_score_loss(y, np.asarray(probability, float)) / reference


# ----------------------------------------------------------------------------------
# Operating points and performance metrics
# ----------------------------------------------------------------------------------

def operating_points(y_true, probability, target_sensitivity=0.80):
    # Three thresholds: the default 0.5, Youden's J, and the one reaching the target sensitivity.
    from sklearn.metrics import roc_curve
    y = np.asarray(y_true).astype(int)
    probability = np.asarray(probability, float)
    false_positive_rate, true_positive_rate, thresholds = roc_curve(y, probability)
    youden_threshold = float(thresholds[np.argmax(true_positive_rate - false_positive_rate)])
    reaching_target = np.where(true_positive_rate >= target_sensitivity)[0]
    target_threshold = (float(thresholds[reaching_target[0]]) if len(reaching_target)
                        else float(thresholds[-1]))
    chosen = {"threshold 0.5": 0.5, "max sens + spec": youden_threshold,
              "fixed sensitivity 0.80": target_threshold}
    rows = []
    for name, threshold in chosen.items():
        counts = confusion_matrix_at(y, probability, threshold)
        rows.append({"operating_point": name, "threshold": threshold,
                     **classification_metrics_from_counts(counts)})
    return pd.DataFrame(rows)


def confusion_matrix_at(y_true, probability, threshold):
    y = np.asarray(y_true).astype(int)
    predicted = (np.asarray(probability, float) >= threshold).astype(int)
    return {"true_positive": int(np.sum((predicted == 1) & (y == 1))),
            "false_positive": int(np.sum((predicted == 1) & (y == 0))),
            "false_negative": int(np.sum((predicted == 0) & (y == 1))),
            "true_negative": int(np.sum((predicted == 0) & (y == 0)))}


def classification_metrics_from_counts(counts):
    true_positive = counts["true_positive"]
    false_positive = counts["false_positive"]
    false_negative = counts["false_negative"]
    true_negative = counts["true_negative"]
    positives = true_positive + false_negative
    negatives = true_negative + false_positive
    sensitivity = true_positive / positives if positives else np.nan
    specificity = true_negative / negatives if negatives else np.nan
    ppv = true_positive / (true_positive + false_positive) if (true_positive + false_positive) else np.nan
    npv = true_negative / (true_negative + false_negative) if (true_negative + false_negative) else np.nan
    f1 = (2 * ppv * sensitivity / (ppv + sensitivity)
          if (ppv and sensitivity and (ppv + sensitivity) > 0) else np.nan)
    balanced_accuracy = np.nanmean([sensitivity, specificity])
    return {"sensitivity": sensitivity, "specificity": specificity, "ppv": ppv, "npv": npv,
            "f1": f1, "balanced_accuracy": balanced_accuracy,
            "alert_rate": (true_positive + false_positive) / max(1, (positives + negatives))}


def all_metrics(y_true, probability, thresholds):
    # Discrimination (threshold-free) plus classification at each operating point, as one flat dict.
    from sklearn.metrics import roc_auc_score, average_precision_score
    y = np.asarray(y_true).astype(int)
    probability = np.asarray(probability, float)
    result = {"auroc": _safe(roc_auc_score, y, probability),
              "auprc": _safe(average_precision_score, y, probability),
              "scaled_brier": _safe(scaled_brier_score, y, probability)}
    for name, threshold in thresholds.items():
        counts = confusion_matrix_at(y, probability, threshold)
        for metric, value in classification_metrics_from_counts(counts).items():
            result[metric + " @ " + name] = value
    return result


def _safe(function, y, probability):
    try:
        return function(y, probability)
    except Exception:
        return np.nan


def evaluate_with_confidence_intervals(y_true, probability, thresholds, n_bootstrap=1000, seed=SEED):
    # Point estimate of every metric plus a stratified bootstrap 95% confidence interval.
    y = np.asarray(y_true).astype(int)
    probability = np.asarray(probability, float)
    point_estimate = all_metrics(y, probability, thresholds)
    positives = np.where(y == 1)[0]
    negatives = np.where(y == 0)[0]
    random_generator = np.random.default_rng(seed)
    bootstrap_values = {metric: [] for metric in point_estimate}
    for _ in range(n_bootstrap):
        resample = np.concatenate([
            random_generator.choice(positives, positives.size, replace=True),
            random_generator.choice(negatives, negatives.size, replace=True)])
        for metric, value in all_metrics(y[resample], probability[resample], thresholds).items():
            bootstrap_values[metric].append(value)
    rows = []
    for metric, value in point_estimate.items():
        low = float(np.nanpercentile(bootstrap_values[metric], 2.5))
        high = float(np.nanpercentile(bootstrap_values[metric], 97.5))
        rows.append({"metric": metric, "value": value, "ci_low": low, "ci_high": high})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------------
# ROC / precision-recall curve points (for plotting)
# ----------------------------------------------------------------------------------

def roc_points(y_true, probability):
    from sklearn.metrics import roc_curve
    false_positive_rate, true_positive_rate, _ = roc_curve(y_true, probability)
    return pd.DataFrame({"false_positive_rate": false_positive_rate,
                         "true_positive_rate": true_positive_rate})


def precision_recall_points(y_true, probability):
    from sklearn.metrics import precision_recall_curve
    precision, recall, _ = precision_recall_curve(y_true, probability)
    return pd.DataFrame({"recall": recall, "precision": precision})


# ----------------------------------------------------------------------------------
# Decision curve (net benefit)
# ----------------------------------------------------------------------------------

def net_benefit_curve(y_true, probability, threshold_probabilities=None):
    y = np.asarray(y_true).astype(int)
    probability = np.asarray(probability, float)
    n = len(y)
    prevalence = y.mean()
    if threshold_probabilities is None:
        threshold_probabilities = np.round(np.arange(0.002, 0.03, 0.001), 4)
    rows = []
    for threshold in threshold_probabilities:
        predicted = (probability >= threshold).astype(int)
        true_positive = int(np.sum((predicted == 1) & (y == 1)))
        false_positive = int(np.sum((predicted == 1) & (y == 0)))
        weight = threshold / (1 - threshold)
        rows.append({"threshold_probability": float(threshold),
                     "net_benefit_model": true_positive / n - (false_positive / n) * weight,
                     "net_benefit_treat_all": prevalence - (1 - prevalence) * weight,
                     "net_benefit_treat_none": 0.0})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------------
# Comparing AUROC between two models on the same patients
# ----------------------------------------------------------------------------------

def paired_bootstrap_auroc_difference(y_true, probability_a, probability_b, n_bootstrap=1000, seed=SEED):
    # Resample the same patients, recompute both AUROCs and their difference (a - b), and take the
    # 95% percentile interval and a two-sided p-value of the difference.
    from sklearn.metrics import roc_auc_score
    y = np.asarray(y_true).astype(int)
    probability_a = np.asarray(probability_a, float)
    probability_b = np.asarray(probability_b, float)
    observed_difference = roc_auc_score(y, probability_a) - roc_auc_score(y, probability_b)
    positives = np.where(y == 1)[0]
    negatives = np.where(y == 0)[0]
    random_generator = np.random.default_rng(seed)
    differences = []
    for _ in range(n_bootstrap):
        resample = np.concatenate([
            random_generator.choice(positives, positives.size, replace=True),
            random_generator.choice(negatives, negatives.size, replace=True)])
        differences.append(roc_auc_score(y[resample], probability_a[resample])
                           - roc_auc_score(y[resample], probability_b[resample]))
    proportion_below_zero = np.mean(np.array(differences) < 0)
    two_sided_p_value = 2 * min(proportion_below_zero, 1 - proportion_below_zero)
    return {"auroc_difference": float(observed_difference),
            "ci_low": float(np.percentile(differences, 2.5)),
            "ci_high": float(np.percentile(differences, 97.5)),
            "p_value": float(two_sided_p_value)}


def delong_test(y_true, probability_a, probability_b):
    # DeLong's test for two correlated AUROCs measured on the same patients.
    from scipy import stats
    y = np.asarray(y_true).astype(int)
    order = np.argsort(-y)
    n_positive = int(y.sum())
    predictions = np.vstack((np.asarray(probability_a, float)[order],
                             np.asarray(probability_b, float)[order]))
    aucs, covariance = _delong_covariance(predictions, n_positive)
    difference = float(aucs[0] - aucs[1])
    variance = covariance[0, 0] + covariance[1, 1] - 2 * covariance[0, 1]
    standard_error = float(np.sqrt(variance)) if variance > 0 else 0.0
    if standard_error == 0:
        p_value = 1.0 if difference == 0 else 0.0
    else:
        p_value = float(2 * stats.norm.sf(abs(difference / standard_error)))
    return {"auroc_a": float(aucs[0]), "auroc_b": float(aucs[1]),
            "auroc_difference": difference, "p_value": p_value}


def _delong_midrank(values):
    order = np.argsort(values)
    sorted_values = values[order]
    n = len(values)
    ranks = np.zeros(n)
    i = 0
    while i < n:
        j = i
        while j < n and sorted_values[j] == sorted_values[i]:
            j += 1
        ranks[i:j] = 0.5 * (i + j - 1)
        i = j
    result = np.empty(n)
    result[order] = ranks + 1
    return result


def _delong_covariance(predictions, n_positive):
    m = n_positive
    n = predictions.shape[1] - n_positive
    n_models = predictions.shape[0]
    tx = np.empty([n_models, m])
    ty = np.empty([n_models, n])
    tz = np.empty([n_models, m + n])
    for r in range(n_models):
        tx[r] = _delong_midrank(predictions[r, :m])
        ty[r] = _delong_midrank(predictions[r, m:])
        tz[r] = _delong_midrank(predictions[r, :])
    aucs = tz[:, :m].sum(axis=1) / m / n - (m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    covariance = np.cov(v01) / m + np.cov(v10) / n
    return aucs, np.atleast_2d(covariance)


# ----------------------------------------------------------------------------------
# External validation across municipalities
# ----------------------------------------------------------------------------------

def safe_auroc(y_true, probability):
    # AUROC, or NaN when a group has only one outcome class (so it cannot be computed). Used for the
    # per-municipality AUROC on the external test set, where many small municipalities have no deaths.
    from sklearn.metrics import roc_auc_score
    y = np.asarray(y_true).astype(int)
    if y.sum() < 1 or (y == 0).sum() < 1:
        return np.nan
    return float(roc_auc_score(y, np.asarray(probability, float)))


def population_size_group(population):
    # Municipality 2016 population bands used in the external transportability analysis.
    if pd.isna(population):
        return "unknown"
    if population <= 5000:
        return "up to 5,000"
    if population <= 10000:
        return "5,001 to 10,000"
    if population <= 20000:
        return "10,001 to 20,000"
    if population <= 100000:
        return "20,001 to 100,000"
    if population <= 500000:
        return "100,001 to 500,000"
    return "more than 500,000"


POPULATION_SIZE_ORDER = ["up to 5,000", "5,001 to 10,000", "10,001 to 20,000",
                         "20,001 to 100,000", "100,001 to 500,000", "more than 500,000"]


# ----------------------------------------------------------------------------------
# SHAP importance (aggregated back to the original predictors)
# ----------------------------------------------------------------------------------

def _explain(model, X_transformed):
    # SHAP values for a fitted model on the already-preprocessed matrix. Tree explainer for the
    # gradient-boosting models; linear explainer for the logistic regression.
    import shap
    try:
        explainer = shap.TreeExplainer(model)
        values = explainer.shap_values(X_transformed)
    except Exception:
        explainer = shap.LinearExplainer(model, X_transformed)
        values = explainer.shap_values(X_transformed)
    if isinstance(values, list):        # some versions return one array per class
        values = values[1]
    return np.asarray(values)


def shap_mean_abs_importance(pipeline, X):
    # Mean absolute SHAP value per ORIGINAL predictor. The preprocessor expands the categoricals into
    # one-hot columns; SHAP is computed on those columns and summed back to the predictor they came
    # from, so importance is reported per predictor, not per dummy.
    preprocessor = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]
    X_transformed = preprocessor.transform(X)
    column_names = preprocessor.get_feature_names_out()
    values = _explain(model, X_transformed)
    mean_abs = np.abs(values).mean(axis=0)
    importance = {}
    for column, value in zip(column_names, mean_abs):
        original = column.split("__", 1)[-1]
        for predictor in FEATURES:
            if original.startswith(predictor):
                importance[predictor] = importance.get(predictor, 0.0) + float(value)
                break
    return pd.Series(importance).reindex(FEATURES).fillna(0.0)


def shap_values_for_beeswarm(pipeline, X):
    # SHAP values on the preprocessed columns, for a beeswarm summary plot (one dot per record).
    # Returns (values, transformed matrix, column names).
    preprocessor = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]
    X_transformed = preprocessor.transform(X)
    column_names = list(preprocessor.get_feature_names_out())
    values = _explain(model, X_transformed)
    return values, X_transformed, column_names


# ----------------------------------------------------------------------------------
# Subgroups (fairness) and death-timing windows
# ----------------------------------------------------------------------------------

def subgroup_masks(data):
    # Subgroup definitions.
    race = data["RACACORMAE"]
    education = data["ESCMAE2010"]
    maternal_age = data["IDADEMAE"]
    return {"race": {"white": race.eq(1), "non_white": race.isin([2, 3, 4, 5])},
            "education": {"high": education.isin([3, 4, 5]), "low": education.isin([0, 1, 2])},
            "maternal_age": {"adult": maternal_age >= 18, "adolescent": maternal_age < 18}}


def _sensitivity_and_fpr(y, probability, threshold):
    counts = confusion_matrix_at(y, probability, threshold)
    metrics = classification_metrics_from_counts(counts)
    return metrics["sensitivity"], 1 - metrics["specificity"]


def fairness_gaps_with_ci(y_true, probability, mask_a, mask_b, threshold,
                          n_bootstrap=1000, seed=SEED):
    # Gap between two subgroups (group A minus group B) at one threshold, with a stratified bootstrap
    # 95% CI. Returns the equal-opportunity gap (sensitivity difference) and the equalized-odds
    # false-positive-rate gap.
    y = np.asarray(y_true).astype(int)
    probability = np.asarray(probability, float)
    index_a = np.where(np.asarray(mask_a))[0]
    index_b = np.where(np.asarray(mask_b))[0]
    sensitivity_a, fpr_a = _sensitivity_and_fpr(y[index_a], probability[index_a], threshold)
    sensitivity_b, fpr_b = _sensitivity_and_fpr(y[index_b], probability[index_b], threshold)
    point = {"sensitivity": sensitivity_a - sensitivity_b, "fpr": fpr_a - fpr_b}
    random_generator = np.random.default_rng(seed)
    draws = {"sensitivity": [], "fpr": []}
    for _ in range(n_bootstrap):
        ra = random_generator.choice(index_a, index_a.size, replace=True)
        rb = random_generator.choice(index_b, index_b.size, replace=True)
        sa, fa = _sensitivity_and_fpr(y[ra], probability[ra], threshold)
        sb, fb = _sensitivity_and_fpr(y[rb], probability[rb], threshold)
        draws["sensitivity"].append(sa - sb)
        draws["fpr"].append(fa - fb)
    result = {}
    for gap_type, value in point.items():
        result[gap_type] = {"gap": value,
                            "ci_low": float(np.nanpercentile(draws[gap_type], 2.5)),
                            "ci_high": float(np.nanpercentile(draws[gap_type], 97.5))}
    return result


def death_timing_window_rows(y_true, death_timing_values, window):
    # Rows for evaluating one death-timing window against all survivors (one-versus-rest):
    # every survivor plus the deaths that happened in that window.
    y = np.asarray(y_true).astype(int)
    death_timing_values = np.asarray(death_timing_values, dtype=object)
    survivor = (y == 0)
    death_in_window = (y == 1) & (death_timing_values == window)
    return np.where(survivor | death_in_window)[0]


# ----------------------------------------------------------------------------------
# Saving and loading fitted objects (joblib)
# ----------------------------------------------------------------------------------

def save_object(obj, path):
    import joblib
    if not str(path).endswith(".pkl"):
        path = str(path) + ".pkl"
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    joblib.dump(obj, path)
    return path


def load_object(path):
    import joblib
    if not str(path).endswith(".pkl"):
        path = str(path) + ".pkl"
    return joblib.load(path)


def load_or_compute(path, compute):
    # Load a saved result if it exists, otherwise compute it and save it (used for the heavy
    # per-municipality external results, so the notebook does not recompute them every run).
    if not str(path).endswith(".pkl"):
        path = str(path) + ".pkl"
    if os.path.exists(path):
        return load_object(path)
    obj = compute()
    save_object(obj, path)
    return obj


# ----------------------------------------------------------------------------------
# Saving tables and figures (file name = table or figure number)
# ----------------------------------------------------------------------------------

def save_table(table, folder, name):
    # A table: written as .xlsx and .csv.
    os.makedirs(folder, exist_ok=True)
    table.to_excel(os.path.join(folder, name + ".xlsx"), index=False)
    table.to_csv(os.path.join(folder, name + ".csv"), index=False)


def save_figure(figure, folder, name, data=None):
    # A figure: written as .pdf. If plot data is given it is written
    # as a .csv next to the figure, so the figure can be recoloured/reformatted without recomputing.
    os.makedirs(folder, exist_ok=True)
    figure.savefig(os.path.join(folder, name + ".pdf"), bbox_inches="tight")
    if data is not None:
        data.to_csv(os.path.join(folder, name + ".csv"), index=False)


def save_sheets(sheets, folder, name):
    # A supplement table that has several sub-tables (one per cohort, model, or subgroup dimension):
    # written as one .xlsx workbook with one sheet each. `sheets` is a dict {sheet name: DataFrame}.
    os.makedirs(folder, exist_ok=True)
    with pd.ExcelWriter(os.path.join(folder, name + ".xlsx")) as writer:
        for sheet_name, table in sheets.items():
            table.to_excel(writer, sheet_name=str(sheet_name)[:31], index=False)


# ----------------------------------------------------------------------------------
# Presentation: table formatting
# ----------------------------------------------------------------------------------

# predictor labels used in the tables and figures
VARIABLE_LABELS = {"LOCNASC": "Place of birth", "PARTO": "Mode of delivery", "IDADEMAE": "Mother age",
                   "SEMAGESTAC": "Gestational age", "APGAR1": "Apgar 1 minute",
                   "IDANOMAL": "Congenital anomaly", "APGAR5": "Apgar 5 minutes", "PESO": "Birth weight"}

# how each metric key is written in the tables
_METRIC_LABELS = {"auroc": "AUROC", "auprc": "AUPRC", "scaled_brier": "Scaled Brier",
                  "sensitivity": "Sensitivity", "specificity": "Specificity", "ppv": "PPV",
                  "npv": "NPV", "f1": "F1", "balanced_accuracy": "Balanced accuracy",
                  "alert_rate": "Alert rate"}


def estimate_with_ci(value, low, high, decimals=3):
    # one cell of a results table, written as "value [low-high]" (empty when the value is missing)
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    return f"{value:.{decimals}f} [{low:.{decimals}f}-{high:.{decimals}f}]"


def _metric_label(metric_key):
    # turn a metric key from all_metrics into its table label:
    # "auroc" -> "AUROC"; "sensitivity @ max sens + spec" -> "Sensitivity (max sens + spec)"
    if " @ " in metric_key:
        metric, operating_point = metric_key.split(" @ ")
        return _METRIC_LABELS.get(metric, metric) + " (" + operating_point + ")"
    return _METRIC_LABELS.get(metric_key, metric_key)


def metric_battery_wide(columns):
    # Build a wide "Metric x columns" table with "value [95% CI]" cells. `columns` is a dict
    # {column label: evaluate_with_confidence_intervals output}. Metrics are listed alphabetically by label.
    wide = None
    for column_label, evaluation in columns.items():
        cells = {_metric_label(row["metric"]): estimate_with_ci(row["value"], row["ci_low"], row["ci_high"])
                 for _, row in evaluation.iterrows()}
        series = pd.Series(cells, name=column_label)
        wide = series.to_frame() if wide is None else wide.join(series)
    wide = wide.sort_index()
    wide.index.name = "Metric"
    return wide.reset_index()


def median_iqr_text(values):
    # "median (q1-q3)" for a descriptive table cell (empty when there are no values)
    values = pd.to_numeric(values, errors="coerce").dropna()
    if not len(values):
        return ""
    return (format(round(values.median()), ",") + " (" + format(round(values.quantile(0.25)), ",")
            + "-" + format(round(values.quantile(0.75)), ",") + ")")


def count_percent_text(series, code_value):
    # "count (percent%)" of one category within a group. Percentages below 1% keep one decimal
    # (e.g. 0.4%), 1% and above are whole numbers (e.g. 54%).
    total = len(series)
    if not total:
        return ""
    n = int((series == code_value).sum())
    percent = 100 * n / total
    percent_text = str(round(percent, 1)) if percent < 1 else str(round(percent))
    return format(n, ",") + " (" + percent_text + "%)"


def missing_percent_text(series):
    # "X%" missing for a descriptive table cell; "<0.1%" when some but under 0.1% is missing
    percent = 100 * series.isna().mean()
    if percent == 0:
        return "0%"
    if percent < 0.1:
        return "<0.1%"
    return str(round(percent, 1)) + "%"


def case_mix_divergence(block_continuous, train_mean, train_std):
    # How far a municipality's case mix is from the training distribution: the mean absolute
    # standardised difference of the continuous predictors' means (0 = identical to training).
    standardised = (block_continuous.mean() - train_mean) / train_std
    return float(np.nanmean(np.abs(standardised)))
