from pathlib import Path

import pandas as pd
from rdflib import Graph, URIRef


RDF_STATEMENT = URIRef("http://www.w3.org/1999/02/22-rdf-syntax-ns#Statement")
RDF_TYPE = URIRef("http://www.w3.org/1999/02/22-rdf-syntax-ns#type")
RDF_SUBJECT = URIRef("http://www.w3.org/1999/02/22-rdf-syntax-ns#subject")
RDF_PREDICATE = URIRef("http://www.w3.org/1999/02/22-rdf-syntax-ns#predicate")
RDF_OBJECT = URIRef("http://www.w3.org/1999/02/22-rdf-syntax-ns#object")

HAS_TRUTH_VALUE = URIRef("http://swc2017.aksw.org/hasTruthValue")


def short_uri(uri):
    """
    Converts full URI into a readable name.
    Example:
    http://dbpedia.org/resource/Walt_Whitman -> Walt_Whitman
    """
    if uri is None:
        return None

    uri = str(uri)

    if "#" in uri:
        return uri.split("#")[-1]

    return uri.rstrip("/").split("/")[-1]


def parse_statement_file(file_path):
    """
    Parses train/test RDF statement file.

    Returns a pandas DataFrame with:
    fact_id, subject, predicate, object, truth_value
    """

    file_path = Path(file_path)

    graph = Graph()
    graph.parse(str(file_path), format="nt")

    rows = []

    for fact_id in graph.subjects(RDF_TYPE, RDF_STATEMENT):
        subject = graph.value(fact_id, RDF_SUBJECT)
        predicate = graph.value(fact_id, RDF_PREDICATE)
        obj = graph.value(fact_id, RDF_OBJECT)
        truth_literal = graph.value(fact_id, HAS_TRUTH_VALUE)

        if subject is None or predicate is None or obj is None:
            continue

        truth_value = None
        if truth_literal is not None:
            truth_value = float(truth_literal)

        rows.append(
            {
                "fact_id": str(fact_id),
                "subject": str(subject),
                "predicate": str(predicate),
                "object": str(obj),
                "truth_value": truth_value,
                "subject_short": short_uri(subject),
                "predicate_short": short_uri(predicate),
                "object_short": short_uri(obj),
            }
        )

    df = pd.DataFrame(rows)
    df = df.sort_values("fact_id").reset_index(drop=True)

    return df