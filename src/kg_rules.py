from sklearn.metrics import roc_auc_score


FUNCTIONAL_LIKE_PREDICATES = {
    "birthPlace",
    "deathPlace",
    "foundationPlace",
}


def apply_kg_rules(prediction_df, kg_feature_df):
    """
    Combines previous model score with DBpedia KG evidence.
    """

    df = prediction_df.merge(kg_feature_df, on="fact_id", how="left")

    df["kg_exact"] = df["kg_exact"].fillna(0).astype(int)
    df["kg_sp_count"] = df["kg_sp_count"].fillna(0).astype(int)
    df["kg_po_count"] = df["kg_po_count"].fillna(0).astype(int)

    df["score_before_kg"] = df["score"]
    df["score"] = df["score_before_kg"]

    exact_mask = df["kg_exact"] == 1

    # If DBpedia contains exact triple, this is strong evidence.
    df.loc[exact_mask, "score"] = 0.99

    # For functional-like predicates, if subject has this predicate
    # but not this object, the candidate is probably false.
    functional_wrong_mask = (
        (df["kg_exact"] == 0)
        & (df["predicate_short"].isin(FUNCTIONAL_LIKE_PREDICATES))
        & (df["kg_sp_count"] > 0)
    )

    df.loc[functional_wrong_mask, "score"] = 0.02

    # If the object is never used with this predicate in DBpedia,
    # it is suspicious. Example: award -> Belgrade.
    incompatible_object_mask = (
        (df["kg_exact"] == 0)
        & (df["kg_po_count"] == 0)
    )

    df.loc[incompatible_object_mask, "score"] = (
        df.loc[incompatible_object_mask, "score"] * 0.35
    )

    # If object is commonly used with this predicate, slightly boost.
    compatible_object_mask = (
        (df["kg_exact"] == 0)
        & (df["kg_po_count"] > 0)
        & (~functional_wrong_mask)
    )

    df.loc[compatible_object_mask, "score"] = (
        0.85 * df.loc[compatible_object_mask, "score"] + 0.15 * 0.65
    )

    # Keep valid range.
    df["score"] = df["score"].clip(0.0, 1.0)

    return df


def print_kg_analysis(train_predictions):
    print("\n==============================")
    print("KG FEATURE ANALYSIS")
    print("==============================")

    print("\nExact triple counts:")
    print(train_predictions["kg_exact"].value_counts())

    print("\nTruth rate by exact triple existence:")
    print(train_predictions.groupby("kg_exact")["truth_value"].mean())

    exact_auc = roc_auc_score(
        train_predictions["truth_value"].astype(int),
        train_predictions["kg_exact"],
    )

    print(f"\nExact-triple-only ROC AUC: {exact_auc:.4f}")