from pathlib import Path
import time

import pandas as pd
import requests


DBPEDIA_ENDPOINT = "https://dbpedia.org/sparql"


def iri(value):
    return f"<{value}>"


def run_sparql(query, timeout=60):
    response = requests.get(
        DBPEDIA_ENDPOINT,
        params={
            "query": query,
            "format": "application/sparql-results+json",
        },
        headers={
            "User-Agent": "FOKG-mini-project/1.0",
        },
        timeout=timeout,
    )

    response.raise_for_status()
    return response.json()


def make_values_exact(batch_df):
    rows = []

    for _, row in batch_df.iterrows():
        rows.append(
            f"({iri(row['fact_id'])} {iri(row['subject'])} "
            f"{iri(row['predicate'])} {iri(row['object'])})"
        )

    return "\n".join(rows)


def make_values_sp(batch_df):
    rows = []

    for _, row in batch_df.iterrows():
        rows.append(
            f"({iri(row['fact_id'])} {iri(row['subject'])} {iri(row['predicate'])})"
        )

    return "\n".join(rows)


def make_values_po(batch_df):
    rows = []

    for _, row in batch_df.iterrows():
        rows.append(
            f"({iri(row['fact_id'])} {iri(row['predicate'])} {iri(row['object'])})"
        )

    return "\n".join(rows)


def query_exact_matches(batch_df):
    values = make_values_exact(batch_df)

    query = f"""
    SELECT ?fact WHERE {{
      VALUES (?fact ?s ?p ?o) {{
        {values}
      }}
      ?s ?p ?o .
    }}
    """

    data = run_sparql(query)

    exact_facts = set()

    for binding in data["results"]["bindings"]:
        exact_facts.add(binding["fact"]["value"])

    return exact_facts


def query_subject_predicate_counts(batch_df):
    values = make_values_sp(batch_df)

    query = f"""
    SELECT ?fact (COUNT(DISTINCT ?x) AS ?count) WHERE {{
      VALUES (?fact ?s ?p) {{
        {values}
      }}
      OPTIONAL {{
        ?s ?p ?x .
      }}
    }}
    GROUP BY ?fact
    """

    data = run_sparql(query)

    counts = {}

    for binding in data["results"]["bindings"]:
        fact = binding["fact"]["value"]
        count = int(binding["count"]["value"])
        counts[fact] = count

    return counts


def query_predicate_object_counts(batch_df):
    values = make_values_po(batch_df)

    query = f"""
    SELECT ?fact (COUNT(DISTINCT ?y) AS ?count) WHERE {{
      VALUES (?fact ?p ?o) {{
        {values}
      }}
      OPTIONAL {{
        ?y ?p ?o .
      }}
    }}
    GROUP BY ?fact
    """

    data = run_sparql(query)

    counts = {}

    for binding in data["results"]["bindings"]:
        fact = binding["fact"]["value"]
        count = int(binding["count"]["value"])
        counts[fact] = count

    return counts


def get_dbpedia_features(df, cache_path, batch_size=25):
    """
    Gets DBpedia evidence features.

    Features:
    - kg_exact: candidate triple exists in DBpedia
    - kg_sp_count: number of DBpedia objects for same subject + predicate
    - kg_po_count: number of DBpedia subjects for same predicate + object
    """

    cache_path = Path(cache_path)

    if cache_path.exists():
        print(f"Loading cached KG features: {cache_path}")
        return pd.read_csv(cache_path)

    rows = []

    total = len(df)

    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        batch_df = df.iloc[start:end]

        print(f"Querying DBpedia KG features: {start + 1}-{end} / {total}")

        try:
            exact_facts = query_exact_matches(batch_df)
            sp_counts = query_subject_predicate_counts(batch_df)
            po_counts = query_predicate_object_counts(batch_df)

            for _, row in batch_df.iterrows():
                fact_id = row["fact_id"]

                rows.append(
                    {
                        "fact_id": fact_id,
                        "kg_exact": 1 if fact_id in exact_facts else 0,
                        "kg_sp_count": sp_counts.get(fact_id, 0),
                        "kg_po_count": po_counts.get(fact_id, 0),
                    }
                )

        except Exception as error:
            print(f"WARNING: DBpedia query failed for batch {start}-{end}: {error}")

            for _, row in batch_df.iterrows():
                rows.append(
                    {
                        "fact_id": row["fact_id"],
                        "kg_exact": 0,
                        "kg_sp_count": 0,
                        "kg_po_count": 0,
                    }
                )

        time.sleep(0.5)

    feature_df = pd.DataFrame(rows)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    feature_df.to_csv(cache_path, index=False)

    return feature_df