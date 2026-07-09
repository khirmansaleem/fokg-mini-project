import pandas as pd
from sklearn.metrics import roc_auc_score


def train_predicate_baseline(train_df):
    """
    Learns the average truth value for each predicate.
    Example: birthPlace -> 0.65
    """

    predicate_scores = (
        train_df
        .groupby("predicate")["truth_value"]
        .mean()
        .to_dict()
    )

    global_mean = train_df["truth_value"].mean()

    return predicate_scores, global_mean


def apply_predicate_baseline(df, predicate_scores, global_mean):
    """
    Assigns each fact a score based on its predicate.
    If a predicate is unknown, use global mean.
    """

    df = df.copy()

    df["score"] = df["predicate"].map(predicate_scores)
    df["score"] = df["score"].fillna(global_mean)

    return df


def evaluate_auc(df):
    """
    Calculates ROC AUC if truth labels are available.
    """

    if "truth_value" not in df.columns:
        return None

    if df["truth_value"].isna().all():
        return None

    return roc_auc_score(df["truth_value"], df["score"])


def print_predicate_scores(predicate_scores):
    """
    Prints learned predicate scores in readable form.
    """

    rows = []

    for predicate, score in predicate_scores.items():
        predicate_short = predicate.rstrip("/").split("/")[-1]
        rows.append((predicate_short, score))

    rows = sorted(rows, key=lambda x: x[1], reverse=True)

    print("\nPredicate baseline scores:")
    for predicate_short, score in rows:
        print(f"{predicate_short:20s} {score:.4f}")