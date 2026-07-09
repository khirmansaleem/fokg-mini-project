import re

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline


def clean_text(value):
    if value is None:
        return ""

    value = str(value)
    value = value.replace("_", " ")
    value = value.replace("-", " ")
    value = re.sub(r"[^A-Za-z0-9,() ]+", " ", value)
    value = re.sub(r"\s+", " ", value)
    return value.strip().lower()


def bucket_token_count(text):
    count = len(text.split())

    if count <= 1:
        return "one_token"
    if count == 2:
        return "two_tokens"
    if count == 3:
        return "three_tokens"

    return "many_tokens"


def build_pattern_features(row):
    predicate = clean_text(row["predicate_short"]).replace(" ", "_")
    subject = clean_text(row["subject_short"])
    obj = clean_text(row["object_short"])
    original_obj = str(row["object_short"])

    features = []

    features.append(f"pred_{predicate}")
    features.append(f"subject_{bucket_token_count(subject)}")
    features.append(f"object_{bucket_token_count(obj)}")

    if "," in original_obj:
        features.append("object_has_comma")

    if "(" in original_obj or ")" in original_obj:
        features.append("object_has_parentheses")

    award_words = [
        "award",
        "prize",
        "medal",
        "nobel",
        "oscar",
        "emmy",
        "grammy",
        "bafta",
        "golden globe",
    ]

    place_words = [
        "city",
        "county",
        "state",
        "province",
        "district",
        "california",
        "new jersey",
        "new york",
        "texas",
        "florida",
        "london",
        "paris",
        "moscow",
        "germany",
        "france",
        "india",
        "canada",
        "russia",
        "england",
        "scotland",
        "ireland",
        "australia",
        "island",
    ]

    organization_words = [
        "company",
        "corporation",
        "inc",
        "ltd",
        "limited",
        "group",
        "lab",
        "university",
        "college",
        "school",
        "airlines",
        "bank",
    ]

    sports_words = [
        "fc",
        "club",
        "basket",
        "lakers",
        "rockets",
        "heat",
        "warriors",
        "wizards",
        "bulls",
        "celtics",
        "knicks",
        "blazers",
        "trail blazers",
    ]

    movie_or_book_words = [
        "film",
        "novel",
        "book",
        "story",
        "worlds",
        "love",
    ]

    if any(word in obj for word in award_words):
        features.append("object_looks_award")

    if any(word in obj for word in place_words):
        features.append("object_looks_place")

    if any(word in obj for word in organization_words):
        features.append("object_looks_org")

    if any(word in obj for word in sports_words):
        features.append("object_looks_team")

    if any(word in obj for word in movie_or_book_words):
        features.append("object_looks_creative_work")

    if "_" in original_obj:
        features.append("object_multiple_uri_tokens")

    # Predicate-specific interactions
    base_features = list(features)
    for feature in base_features:
        if feature != f"pred_{predicate}":
            features.append(f"{predicate}_{feature}")

    # Mismatch-style clues
    if predicate == "award" and "object_looks_place" in features:
        features.append("mismatch_award_place")

    if predicate in ["birthplace", "deathplace", "foundationplace"]:
        if "object_looks_award" in features:
            features.append("mismatch_place_award")
        if "object_looks_team" in features:
            features.append("mismatch_place_team")

    if predicate == "author" and "object_looks_place" in features:
        features.append("mismatch_author_place")

    if predicate == "team" and "object_looks_award" in features:
        features.append("mismatch_team_award")

    if predicate == "subsidiary" and "object_looks_place" in features:
        features.append("mismatch_subsidiary_place")

    return " ".join(features)


def build_text_features(df):
    return pd.Series([build_pattern_features(row) for _, row in df.iterrows()])


def create_text_model():
    return Pipeline(
        steps=[
            (
                "vectorizer",
                CountVectorizer(
                    binary=True,
                    token_pattern=r"(?u)\b[\w_]+\b",
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=2000,
                    solver="liblinear",
                    class_weight="balanced",
                    C=0.5,
                    random_state=42,
                ),
            ),
        ]
    )


def train_improved_model(train_df):
    x_text = build_text_features(train_df)
    y = train_df["truth_value"].astype(int)

    model = create_text_model()

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )

    cv_scores = cross_val_predict(
        model,
        x_text,
        y,
        cv=cv,
        method="predict_proba",
    )[:, 1]

    cv_auc = roc_auc_score(y, cv_scores)

    model.fit(x_text, y)

    train_scores = model.predict_proba(x_text)[:, 1]
    train_auc = roc_auc_score(y, train_scores)

    return model, train_scores, train_auc, cv_auc, cv_scores


def apply_improved_model(model, df):
    df = df.copy()

    x_text = build_text_features(df)
    df["ml_score"] = model.predict_proba(x_text)[:, 1]
    df["score"] = df["ml_score"]

    return df


def blend_with_baseline(ml_df, baseline_df, ml_weight=0.50):
    df = ml_df.copy()

    baseline_weight = 1.0 - ml_weight

    df["baseline_score"] = baseline_df["score"].values

    df["score"] = (
        ml_weight * df["ml_score"]
        + baseline_weight * df["baseline_score"]
    )

    return df