import io
from typing import AbstractSet, Tuple

import numpy as np
import pytest

from ...exceptions import UnsupportedProgramError
from ...probabilistic.exceptions import UnsupportedProbabilisticQueryError
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
        with pytest.raises(UnsupportedProgramError):
            nl.solve_all()


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
    assert np.isclose(df.loc[df[0] == "b"].iloc[0][1], 2 / 3 / 2)
    assert np.isclose(df.loc[df[0] == "a"].iloc[0][1], 1 / 3 / 2)


def test_solve_query():
    nl = ProbabilisticFrontend()
    P = nl.add_uniform_probabilistic_choice_over_set(
        [("a",), ("b",), ("c",)], name="P"
    )
    Q = nl.add_uniform_probabilistic_choice_over_set(
        [("a",), ("d",), ("c",)], name="Q"
    )
    with nl.scope as e:
        e.Z[e.x, e.PROB[e.x]] = P[e.x] & Q[e.x]
        res = nl.query((e.x, e.p), e.Z[e.x, e.p])
    df = res.as_pandas_dataframe()
    assert len(df) == 2
    assert all(c1 == c2 for c1, c2 in zip(df.columns, ["x", "p"]))
    assert len(df.loc[df["x"] == "a"]) == 1
    assert len(df.loc[df["x"] == "c"]) == 1
    assert np.isclose(df.loc[df["x"] == "a"].iloc[0]["p"], 1 / 9)
    assert np.isclose(df.loc[df["x"] == "c"].iloc[0]["p"], 1 / 9)


def test_solve_query_prob_col_not_last():
    nl = ProbabilisticFrontend()
    P = nl.add_uniform_probabilistic_choice_over_set(
        [("a",), ("b",), ("c",)], name="P"
    )
    Q = nl.add_uniform_probabilistic_choice_over_set(
        [("a",), ("d",), ("c",)], name="Q"
    )
    with nl.scope as e:
        e.Z[e.PROB[e.x], e.x] = P[e.x] & Q[e.x]
        res = nl.query((e.p, e.x), e.Z[e.p, e.x])
    df = res.as_pandas_dataframe()
    assert len(df) == 2
    assert all(c1 == c2 for c1, c2 in zip(df.columns, ["p", "x"]))
    assert len(df.loc[df["x"] == "a"]) == 1
    assert len(df.loc[df["x"] == "c"]) == 1
    assert np.isclose(df.loc[df["x"] == "a"].iloc[0]["p"], 1 / 9)
    assert np.isclose(df.loc[df["x"] == "c"].iloc[0]["p"], 1 / 9)


def test_solve_boolean_query():
    nl = ProbabilisticFrontend()
    P = nl.add_uniform_probabilistic_choice_over_set(
        [("a",), ("b",), ("c",)], name="P"
    )
    Q = nl.add_uniform_probabilistic_choice_over_set(
        [("a",), ("d",), ("c",)], name="Q"
    )
    with pytest.raises(UnsupportedProbabilisticQueryError):
        with nl.scope as e:
            e.ans[e.PROB()] = P[e.x] & Q[e.x]
            nl.query((e.p), e.ans[e.p])


def test_solve_complex_stratified_query():
    """
    R(1, 2) : 0.3
    R(1, 4) : 0.7
    R(2, 2) : 0.2
    R(2, 4) : 0.6
    R(2, 6) : 0.8
    Q(4) : 0.2 v Q(6) : 0.8
    A(x, PROB[x])       :- Q(x), R(1, x), R(2, x)
    B(x, y, PROB[x, y]) :- Q(y), R(1, x), R(2, x)
    C(x, y, p1, p2)     :- A(x, p1), B(x, y, p2)

    """
    nl = ProbabilisticFrontend()
    R = nl.add_probabilistic_facts_from_tuples(
        [(0.3, 1, 2), (0.7, 1, 4), (0.2, 2, 2), (0.6, 2, 4), (0.8, 2, 6)],
        name="R",
    )
    Q = nl.add_probabilistic_choice_from_tuples([(0.2, 4), (0.8, 6)], name="Q")
    with nl.scope as e:
        e.A[e.x, e.PROB[e.x]] = Q[e.x] & R[1, e.x] & R[2, e.x]
        e.B[e.x, e.y, e.PROB[e.x, e.y]] = Q[e.y] & R[1, e.x] & R[2, e.x]
        e.C[e.x, e.y, e.p1, e.p2] = e.A[e.x, e.p1] & e.B[e.x, e.y, e.p2]
        res = nl.query((e.x, e.y, e.h, e.z), e.C[e.x, e.y, e.h, e.z])
    df = res.as_pandas_dataframe()
    assert set(df.columns) == {"x", "y", "h", "z"}
    assert len(df.loc[(df["x"] == 4) & (df["y"] == 4)]) == 1
    assert np.isclose(
        df.loc[(df["x"] == 4) & (df["y"] == 4)].iloc[0]["h"], 0.2 * 0.7 * 0.6,
    )
    assert np.isclose(
        df.loc[(df["x"] == 4) & (df["y"] == 4)].iloc[0]["z"], 0.2 * 0.7 * 0.6,
    )
    assert len(df.loc[(df["x"] == 4) & (df["y"] == 6)]) == 1
    assert np.isclose(
        df.loc[(df["x"] == 4) & (df["y"] == 6)].iloc[0]["h"], 0.2 * 0.7 * 0.6,
    )
    assert np.isclose(
        df.loc[(df["x"] == 4) & (df["y"] == 6)].iloc[0]["z"], 0.8 * 0.7 * 0.6,
    )


def test_solve_complex_stratified_query_with_deterministic_part():
    nl = ProbabilisticFrontend()
    A = nl.add_tuple_set([("a",), ("b",), ("c",)], name="A")
    B = nl.add_tuple_set([("a",), ("b",)], name="B")
    P = nl.add_probabilistic_facts_from_tuples(
        [(0.2, "a"), (0.8, "b")], name="P",
    )
    with nl.scope as e:
        e.C[e.x, e.y] = A[e.x] & A[e.y] & B[e.x]
        e.D[e.x, e.y, e.PROB[e.x, e.y]] = e.C[e.x, e.y] & P[e.y]
        res = nl.query((e.x, e.y, e.p), e.D[e.x, e.y, e.p])
    df = res.as_pandas_dataframe()
    assert df.loc[(df["x"] == "a") & (df["y"] == "b")].iloc[0]["p"] == 0.8
    assert df.loc[(df["x"] == "b") & (df["y"] == "b")].iloc[0]["p"] == 0.8
    assert df.loc[(df["x"] == "a") & (df["y"] == "a")].iloc[0]["p"] == 0.2
    assert df.loc[(df["x"] == "b") & (df["y"] == "a")].iloc[0]["p"] == 0.2
