import io
from typing import AbstractSet, Tuple

import numpy as np

from ..probabilistic_frontend import ProbabilisticFrontend


def test_add_uniform_probabilistic_choice_set():
    nl = ProbabilisticFrontend()

    prob = [(a,) for a in range(10)]
    prob_set = nl.add_uniform_probabilistic_choice_over_set(prob, "prob")
    res = nl[prob_set]

    assert prob_set.type is AbstractSet[Tuple[float, int]]
    assert res.type is AbstractSet[Tuple[float, int]]
    assert res.value == frozenset((1 / len(prob), a) for a in range(10))

    d = [("a",), ("b",), ("c",), ("d",)]
    data = nl.add_uniform_probabilistic_choice_over_set(d, name="data")
    res_d = nl[data]

    assert data.type is AbstractSet[Tuple[float, str]]
    assert res_d.type is AbstractSet[Tuple[float, str]]


def test_deterministic_query():
    nl = ProbabilisticFrontend()
    d1 = [(1,), (2,), (3,), (4,), (5,)]
    data1 = nl.add_tuple_set(d1, name="data1")

    d2 = [(2, "a"), (3, "b"), (4, "d"), (5, "c"), (7, "z")]
    data2 = nl.add_tuple_set(d2, name="data2")

    with nl.scope as e:
        e.query1[e.y] = data1[e.x] & data2[e.x, e.y]
        res = nl.solve_all()

    assert "query1" in res.keys()
    q1 = res["query1"].as_pandas_dataframe().values
    assert len(q1) == 4
    for elem in q1:
        assert elem[0] in ["a", "b", "c", "d"]


def test_probabilistic_query():
    nl = ProbabilisticFrontend()
    d1 = [(1,), (2,), (3,), (4,), (5,)]
    data1 = nl.add_uniform_probabilistic_choice_over_set(d1, name="data1")

    d2 = [(2, "a"), (3, "b"), (4, "d"), (5, "c"), (7, "z")]
    data2 = nl.add_uniform_probabilistic_choice_over_set(d2, name="data2")

    d3 = [("a",), ("b",), ("c",)]
    data3 = nl.add_uniform_probabilistic_choice_over_set(d3, name="data3")

    with nl.scope as e:
        e.query1[e.y, e.PROB[e.y]] = data1[e.x] & data2[e.x, e.y]
        e.query2[e.y, e.PROB[e.y]] = e.query1[e.y] & data3[e.y]
        res = nl.solve_all()

    assert "query1" in res.keys()
    assert "query2" in res.keys()
    assert len(res["query1"].columns) == 2
    assert len(res["query2"].columns) == 2
    q2 = res["query2"].as_pandas_dataframe().values
    assert len(q2) == 3
    for elem in q2:
        assert elem[0] in ["a", "b", "c"]
        assert np.isclose(elem[1], 1 / 3 / 5 / 5)


def test_mixed_queries():
    nl = ProbabilisticFrontend()
    d1 = [(1,), (2,), (3,), (4,), (5,)]
    data1 = nl.add_tuple_set(d1, name="data1")

    d2 = [(2, "a"), (3, "b"), (4, "d"), (5, "c"), (7, "z")]
    data2 = nl.add_tuple_set(d2, name="data2")

    d3 = [("a",), ("b",), ("c",), ("d",)]
    data3 = nl.add_uniform_probabilistic_choice_over_set(d3, name="data3")

    with nl.scope as e:
        e.tmp[e.x, e.y] = data1[e.x] & data2[e.x, e.y]
        e.query1[e.x, e.y, e.PROB[e.x, e.y]] = e.tmp[e.x, e.y]
        e.query2[e.y, e.PROB[e.y]] = e.tmp[e.x, e.y] & data3[e.y]
        res = nl.solve_all()

    assert "query1" in res.keys()
    assert "query2" in res.keys()
    assert len(res["query2"].columns) == 2
    q2 = res["query2"].as_pandas_dataframe().values
    assert len(q2) == 4
    for elem in q2:
        assert elem[1] == 0.25
        assert elem[0] in ["a", "b", "c", "d"]


def test_ontology_query():

    test_case = """
    <rdf:RDF
        xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
        xmlns:rdfs="http://www.w3.org/2000/01/rdf-schema#"
        xmlns:owl="http://www.w3.org/2002/07/owl#"
        xmlns:first="http://www.w3.org/2002/03owlt/hasValue/premises001#"
        xml:base="http://www.w3.org/2002/03owlt/hasValue/premises001" >
        <owl:Ontology/>
        <owl:Class rdf:ID="r">
        <rdfs:subClassOf>
            <owl:Restriction>
                <owl:onProperty rdf:resource="#p2"/>
                <owl:hasValue rdf:datatype="http://www.w3.org/2001/XMLSchema#boolean">true</owl:hasValue>
            </owl:Restriction>
        </rdfs:subClassOf>
        </owl:Class>
        <owl:ObjectProperty rdf:ID="p"/>
        <owl:ObjectProperty rdf:ID="p2"/>
        <owl:Class rdf:ID="c"/>
        <first:r rdf:ID="i">
        <rdf:type rdf:resource="http://www.w3.org/2002/07/owl#Thing"/>
        <first:p>
            <owl:Thing rdf:ID="o" />
        </first:p>
        </first:r>
    </rdf:RDF>
    """

    nl = ProbabilisticFrontend()
    nl.load_ontology(io.StringIO(test_case))

    p2 = nl.new_symbol(
        name="http://www.w3.org/2002/03owlt/hasValue/premises001#p2"
    )

    with nl.scope as e:
        e.answer[e.x, e.y] = p2[e.x, e.y]
        solution_instance = nl.solve_all()

    resp = list(solution_instance["answer"].unwrapped_iter())
    assert (
        "http://www.w3.org/2002/03owlt/hasValue/premises001#i",
        "true",
    ) in resp


def test_simple_within_language_succ_query():
    nl = ProbabilisticFrontend()
    P = nl.add_uniform_probabilistic_choice_over_set(
        [("a",), ("b",), ("c",)], name="P"
    )
    Q = nl.add_uniform_probabilistic_choice_over_set(
        [("a",), ("d",), ("e",)], name="Q"
    )
    with nl.scope as e:
        e.Z[e.x, e.PROB[e.x]] = P[e.x] & Q[e.x]
        res = nl.solve_all()
    assert "Z" in res.keys()
    df = res["Z"].as_pandas_dataframe()
    assert len(df) == 1
    assert tuple(df.values[0])[0] == "a"
    assert np.isclose(tuple(df.values[0])[1], 1 / 9)


def test_within_language_succ_query():
    nl = ProbabilisticFrontend()
    P = nl.add_uniform_probabilistic_choice_over_set(
        [("a", "b",), ("b", "c",), ("b", "d",)], name="P"
    )
    Q = nl.add_uniform_probabilistic_choice_over_set(
        [("a",), ("b",)], name="Q"
    )
    with nl.scope as e:
        e.Z[e.x, e.PROB[e.x]] = P[e.x, e.y] & Q[e.x]
        res = nl.solve_all()
    assert "Z" in res.keys()
    df = res["Z"].as_pandas_dataframe()
    assert np.isclose(df.loc[df["x"] == "b"].iloc[0][1], 2 / 3 / 2)
    assert np.isclose(df.loc[df["x"] == "a"].iloc[0][1], 1 / 3 / 2)
