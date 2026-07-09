import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold


def compute_cv_predicate_baseline_scores(train_df, n_splits=5, random_state=42):
    """
    Computes predicate-baseline scores using cross-validation.

    This avoids evaluating a fact using predicate statistics that were
    calculated from that same fact.
    """

    y = train_df["truth_value"].astype(int).values
    cv_scores = np.zeros(len(train_df))

    cv = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )

    for train_index, valid_index in cv.split(train_df, y):
        fold_train = train_df.iloc[train_index]
        fold_valid = train_df.iloc[valid_index]

        predicate_means = (
            fold_train
            .groupby("predicate")["truth_value"]
            .mean()
            .to_dict()
        )

        global_mean = fold_train["truth_value"].mean()

        fold_scores = (
            fold_valid["predicate"]
            .map(predicate_means)
            .fillna(global_mean)
            .values
        )

        cv_scores[valid_index] = fold_scores

    return cv_scores


def tune_blend_weight(y_true, baseline_cv_scores, ml_cv_scores):
    """
    Finds the best blend weight using cross-validation scores.

    final_score = weight * ml_score + (1 - weight) * baseline_score
    """

    rows = []

    for weight in np.arange(0.0, 1.01, 0.05):
        blended_scores = (
            weight * ml_cv_scores
            + (1.0 - weight) * baseline_cv_scores
        )

        auc = roc_auc_score(y_true, blended_scores)

        rows.append(
            {
                "ml_weight": round(float(weight), 2),
                "baseline_weight": round(float(1.0 - weight), 2),
                "cv_auc": auc,
            }
        )

    tuning_df = pd.DataFrame(rows)
    tuning_df = tuning_df.sort_values("cv_auc", ascending=False).reset_index(drop=True)

    best_weight = float(tuning_df.loc[0, "ml_weight"])
    best_auc = float(tuning_df.loc[0, "cv_auc"])

    return best_weight, best_auc, tuning_df