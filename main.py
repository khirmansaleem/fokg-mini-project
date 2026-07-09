from pathlib import Path

from src.parser import parse_statement_file
from src.baseline import (
    train_predicate_baseline,
    apply_predicate_baseline,
    evaluate_auc,
    print_predicate_scores,
)
from src.improved_model import (
    train_improved_model,
    apply_improved_model,
    blend_with_baseline,
)
from src.tuning import (
    compute_cv_predicate_baseline_scores,
    tune_blend_weight,
)
from src.kg_features import get_dbpedia_features
from src.kg_rules import apply_kg_rules, print_kg_analysis
from src.writer import write_result_ttl
from src.validator import validate_result_file


DATA_DIR = Path("data")
OUTPUT_DIR = Path("outputs")


def print_basic_analysis(train_df, test_df):
    print("\n==============================")
    print("BASIC DATA INFORMATION")
    print("==============================")

    print(f"Train facts: {len(train_df)}")
    print(f"Test facts:  {len(test_df)}")

    print("\nTruth value distribution in train:")
    print(train_df["truth_value"].value_counts(dropna=False))

    print("\nTop train predicates:")
    print(train_df["predicate_short"].value_counts())

    print("\nTop test predicates:")
    print(test_df["predicate_short"].value_counts())

    train_predicates = set(train_df["predicate"])
    test_predicates = set(test_df["predicate"])

    missing_in_train = test_predicates - train_predicates

    print("\nTest predicates not present in train:")
    if missing_in_train:
        for pred in sorted(missing_in_train):
            print(pred)
    else:
        print("None. All test predicates are also present in train.")


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    train_path = DATA_DIR / "train.nt.txt"
    test_path = DATA_DIR / "test.nt.txt"

    print("Parsing train file...")
    train_df = parse_statement_file(train_path)

    print("Parsing test file...")
    test_df = parse_statement_file(test_path)

    train_df.to_csv(OUTPUT_DIR / "train_parsed.csv", index=False)
    test_df.to_csv(OUTPUT_DIR / "test_parsed.csv", index=False)

    print_basic_analysis(train_df, test_df)

    # ----------------------------
    # Predicate baseline
    # ----------------------------
    print("\n==============================")
    print("PREDICATE BASELINE")
    print("==============================")

    predicate_scores, global_mean = train_predicate_baseline(train_df)

    print(f"\nGlobal mean truth value: {global_mean:.4f}")
    print_predicate_scores(predicate_scores)

    baseline_train_predictions = apply_predicate_baseline(
        train_df,
        predicate_scores,
        global_mean,
    )

    baseline_test_predictions = apply_predicate_baseline(
        test_df,
        predicate_scores,
        global_mean,
    )

    baseline_train_auc = evaluate_auc(baseline_train_predictions)

    print("\nBaseline Train ROC AUC:")
    print(f"{baseline_train_auc:.4f}")

    # ----------------------------
    # Improved pattern model
    # ----------------------------
    print("\n==============================")
    print("IMPROVED PATTERN MODEL")
    print("==============================")

    (
        model,
        train_ml_scores,
        train_ml_auc,
        ml_cv_auc,
        ml_cv_scores,
    ) = train_improved_model(train_df)

    print(f"ML model train ROC AUC: {train_ml_auc:.4f}")
    print(f"ML model 5-fold CV ROC AUC: {ml_cv_auc:.4f}")

    # ----------------------------
    # Tune blend weight using CV
    # ----------------------------
    print("\n==============================")
    print("TUNING BLEND WEIGHT")
    print("==============================")

    y_true = train_df["truth_value"].astype(int).values

    baseline_cv_scores = compute_cv_predicate_baseline_scores(train_df)

    baseline_cv_auc = evaluate_auc(
        baseline_train_predictions.assign(score=baseline_cv_scores)
    )

    print(f"Predicate baseline 5-fold CV ROC AUC: {baseline_cv_auc:.4f}")

    best_weight, best_cv_auc, tuning_df = tune_blend_weight(
        y_true=y_true,
        baseline_cv_scores=baseline_cv_scores,
        ml_cv_scores=ml_cv_scores,
    )

    tuning_df.to_csv(OUTPUT_DIR / "blend_tuning.csv", index=False)

    print("\nTop blend weights:")
    print(tuning_df.head(10))

    print("\nBest blend:")
    print(f"ML weight:       {best_weight:.2f}")
    print(f"Baseline weight: {1.0 - best_weight:.2f}")
    print(f"Best CV AUC:     {best_cv_auc:.4f}")

    # ----------------------------
    # Fit final pattern model and blend
    # ----------------------------
    ml_train_predictions = apply_improved_model(model, train_df)
    ml_test_predictions = apply_improved_model(model, test_df)

    improved_train_predictions = blend_with_baseline(
        ml_train_predictions,
        baseline_train_predictions,
        ml_weight=best_weight,
    )

    improved_test_predictions = blend_with_baseline(
        ml_test_predictions,
        baseline_test_predictions,
        ml_weight=best_weight,
    )

    improved_train_auc = evaluate_auc(improved_train_predictions)

    # ----------------------------
    # Add DBpedia KG evidence
    # ----------------------------
    print("\n==============================")
    print("ADDING DBPEDIA KG EVIDENCE")
    print("==============================")

    kg_train_features = get_dbpedia_features(
        train_df,
        OUTPUT_DIR / "kg_features_train.csv",
    )

    kg_test_features = get_dbpedia_features(
        test_df,
        OUTPUT_DIR / "kg_features_test.csv",
    )

    kg_train_predictions = apply_kg_rules(
        improved_train_predictions,
        kg_train_features,
    )

    kg_test_predictions = apply_kg_rules(
        improved_test_predictions,
        kg_test_features,
    )

    print_kg_analysis(kg_train_predictions)

    kg_train_auc = evaluate_auc(kg_train_predictions)

    print("\nKG-enhanced train ROC AUC:")
    print(f"{kg_train_auc:.4f}")

    if kg_train_auc > improved_train_auc:
        print("\nUsing KG-enhanced predictions.")
        final_train_predictions = kg_train_predictions
        final_test_predictions = kg_test_predictions
        final_train_auc = kg_train_auc
    else:
        print("\nKG rules did not improve train AUC. Keeping previous improved model.")
        final_train_predictions = improved_train_predictions
        final_test_predictions = improved_test_predictions
        final_train_auc = improved_train_auc

    # ----------------------------
    # Final evaluation summary
    # ----------------------------
    print("\n==============================")
    print("FINAL EVALUATION SUMMARY")
    print("==============================")
    print(f"Baseline train ROC AUC:       {baseline_train_auc:.4f}")
    print(f"ML train ROC AUC:             {train_ml_auc:.4f}")
    print(f"ML 5-fold CV ROC AUC:         {ml_cv_auc:.4f}")
    print(f"Best CV blended ROC AUC:      {best_cv_auc:.4f}")
    print(f"Blended train ROC AUC:        {improved_train_auc:.4f}")
    print(f"KG-enhanced train ROC AUC:    {kg_train_auc:.4f}")
    print(f"Final train ROC AUC:          {final_train_auc:.4f}")

    # ----------------------------
    # Save CSV outputs
    # ----------------------------
    baseline_train_predictions.to_csv(
        OUTPUT_DIR / "baseline_train_predictions.csv",
        index=False,
    )

    baseline_test_predictions.to_csv(
        OUTPUT_DIR / "baseline_test_predictions.csv",
        index=False,
    )

    final_train_predictions.to_csv(
        OUTPUT_DIR / "train_predictions.csv",
        index=False,
    )

    final_test_predictions.to_csv(
        OUTPUT_DIR / "test_predictions.csv",
        index=False,
    )

    # ----------------------------
    # Save TTL outputs
    # ----------------------------
    write_result_ttl(
        baseline_train_predictions,
        OUTPUT_DIR / "baseline_train_result.ttl",
    )

    write_result_ttl(
        baseline_test_predictions,
        OUTPUT_DIR / "baseline_result.ttl",
    )

    write_result_ttl(
        final_train_predictions,
        OUTPUT_DIR / "train_result.ttl",
    )

    write_result_ttl(
        final_test_predictions,
        OUTPUT_DIR / "result.ttl",
    )

    # ----------------------------
    # Validate result files
    # ----------------------------
    validate_result_file(
        OUTPUT_DIR / "train_result.ttl",
        expected_fact_count=len(final_train_predictions),
    )

    validate_result_file(
        OUTPUT_DIR / "result.ttl",
        expected_fact_count=len(final_test_predictions),
    )

    print("\n==============================")
    print("SAVED FILES")
    print("==============================")
    print("outputs/blend_tuning.csv")
    print("outputs/kg_features_train.csv")
    print("outputs/kg_features_test.csv")
    print("outputs/baseline_train_result.ttl")
    print("outputs/baseline_result.ttl")
    print("outputs/train_result.ttl        <-- final train result")
    print("outputs/result.ttl              <-- final test result")
    print("outputs/train_predictions.csv")
    print("outputs/test_predictions.csv")


if __name__ == "__main__":
    main()