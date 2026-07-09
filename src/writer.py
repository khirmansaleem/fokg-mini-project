from pathlib import Path


HAS_TRUTH_VALUE = "http://swc2017.aksw.org/hasTruthValue"
XSD_DOUBLE = "http://www.w3.org/2001/XMLSchema#double"


def write_result_ttl(df, output_path):
    """
    Writes result.ttl in the required GERBIL format.

    Format:
    <Fact-URI> <hasTruthValue> "score"^^<xsd:double> .
    """

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as file:
        for _, row in df.iterrows():
            fact_id = row["fact_id"]
            score = float(row["score"])

            # Keep score inside valid range [0, 1]
            score = max(0.0, min(1.0, score))

            line = (
                f"<{fact_id}> "
                f"<{HAS_TRUTH_VALUE}> "
                f"\"{score:.6f}\"^^<{XSD_DOUBLE}> .\n"
            )

            file.write(line)