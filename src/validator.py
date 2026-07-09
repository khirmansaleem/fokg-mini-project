from pathlib import Path

from rdflib import Graph, URIRef


HAS_TRUTH_VALUE = URIRef("http://swc2017.aksw.org/hasTruthValue")


def validate_result_file(result_path, expected_fact_count=None):
    """
    Validates whether the result file can be parsed as RDF
    and contains one hasTruthValue statement per fact.
    """

    result_path = Path(result_path)

    if not result_path.exists():
        raise FileNotFoundError(f"Result file not found: {result_path}")

    graph = Graph()

    try:
        graph.parse(str(result_path), format="turtle")
    except Exception as error:
        raise ValueError(f"RDF parsing failed: {error}")

    truth_statements = list(graph.triples((None, HAS_TRUTH_VALUE, None)))

    print("\n==============================")
    print("RESULT FILE VALIDATION")
    print("==============================")
    print(f"File: {result_path}")
    print(f"RDF triples found: {len(graph)}")
    print(f"Truth value statements found: {len(truth_statements)}")

    if expected_fact_count is not None:
        print(f"Expected facts: {expected_fact_count}")

        if len(truth_statements) != expected_fact_count:
            raise ValueError(
                f"Wrong number of result lines. "
                f"Expected {expected_fact_count}, found {len(truth_statements)}."
            )

    for fact_id, _, score_literal in truth_statements:
        score = float(score_literal)

        if score < 0.0 or score > 1.0:
            raise ValueError(f"Invalid score {score} for fact {fact_id}")

    print("Validation successful.")