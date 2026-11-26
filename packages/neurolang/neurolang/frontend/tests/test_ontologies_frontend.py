"""Ontologies loaded into a frontend: the parser's tests are in
``neurolang/datalog/tests/test_ontologies_parser.py``."""

import io
import re

import pytest
from typing import AbstractSet, Callable, Tuple
from nl_core.datalog.wrapped_collections import WrappedRelationalAlgebraSet
from nl_expr.logic import Conjunction, Implication

from nl_expr.exceptions import NeuroLangException
from nl_expr.expressions import Constant, Symbol
from ...frontend import NeurolangPDL


KNOWLEDGE_GRAPH = """
@prefix pv: <http://ld.company.org/prod-vocab/> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix owl: <http://www.w3.org/2002/07/owl#> .
@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .
@prefix ex: <http://example.org/> .

pv:Employee a owl:Class ; rdfs:label "Employee"@en .
pv:Manager a owl:Class ; rdfs:subClassOf pv:Employee ; rdfs:label "Manager" .
ex:e1 a pv:Employee ; pv:name "Ann"@en ; pv:age 31 ;
    pv:hired "2020-01-02"^^xsd:date ; pv:price [ pv:amount 2.5 ] .
ex:m1 a pv:Manager ; pv:name "Bob" ; pv:age 40.5 ;
    pv:manages ex:e1 .
<http://other.org/thing> <http://other.org/rel> "x" .
"""



def test_open_world_example():
    owl = '''<?xml version="1.0"?>
    <rdf:RDF xmlns="http://www.w3.org/2002/07/owl#"
        xmlns:owl="http://www.w3.org/2002/07/owl#"
        xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
        xmlns:rdfs="http://www.w3.org/2000/01/rdf-schema#">
        <Ontology>
            <versionInfo>0.3.1</versionInfo>
        </Ontology>

        <owl:Class rdf:ID="Dean">
            <rdfs:label>dean</rdfs:label>
            <rdfs:subClassOf>
                <owl:Class>
                    <owl:intersectionOf rdf:parseType="Collection">
                        <owl:Class rdf:about="#Person" />
                        <owl:Restriction>
                            <owl:onProperty rdf:resource="#headOf" />
                            <owl:someValuesFrom>
                                <owl:Class rdf:about="#College" />
                            </owl:someValuesFrom>
                        </owl:Restriction>
                    </owl:intersectionOf>
                </owl:Class>
            </rdfs:subClassOf>
            <rdfs:subClassOf rdf:resource="#Professor" />
        </owl:Class>

        <owl:Class rdf:ID="Chair">
            <rdfs:label>chair</rdfs:label>
            <rdfs:subClassOf>
                <owl:Class>
                    <owl:intersectionOf rdf:parseType="Collection">
                        <owl:Class rdf:about="#Person" />
                        <owl:Restriction>
                            <owl:onProperty rdf:resource="#headOf" />
                            <owl:someValuesFrom>
                                <owl:Class rdf:about="#Department" />
                            </owl:someValuesFrom>
                        </owl:Restriction>
                    </owl:intersectionOf>
                </owl:Class>
            </rdfs:subClassOf>
            <rdfs:subClassOf rdf:resource="#Professor" />
        </owl:Class>
    </rdf:RDF>'''

    nl = NeurolangPDL()
    nl.load_ontology(io.StringIO(owl))

    nl.add_tuple_set([('Juan',), ('Manuel',)], name='Dean')
    nl.add_tuple_set([('Miguel',), ('Alberto',)], name='Chair')
    nl.add_tuple_set([('College A',), ('College B',)], name='College')
    nl.add_tuple_set([('Department A',), ('Department B',)], name='Department')

    with nl.scope as e:
        e.answer[e.a] = (
            e.headOf[e.a, e.b] &
            e.College[e.b]
        )

        f_term = nl.solve_all()

    res = f_term['answer'].as_pandas_dataframe().values
    assert (res == [['Juan'], ['Manuel']]).all()


def test_retrieve_property():
    owl = '''<?xml version="1.0"?>
    <rdf:RDF xmlns="http://www.w3.org/2002/07/owl#"
        xmlns:owl="http://www.w3.org/2002/07/owl#"
        xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
        xmlns:rdfs="http://www.w3.org/2000/01/rdf-schema#">
        <Ontology>
            <versionInfo>0.3.1</versionInfo>
        </Ontology>

        <owl:Class rdf:ID="Dean">
            <rdfs:label>dean</rdfs:label>
            <rdfs:subClassOf>
                <owl:Class>
                    <owl:intersectionOf rdf:parseType="Collection">
                        <owl:Class rdf:about="#Person" />
                        <owl:Restriction>
                            <owl:onProperty rdf:resource="#headOf" />
                            <owl:someValuesFrom>
                                <owl:Class rdf:about="#College" />
                            </owl:someValuesFrom>
                        </owl:Restriction>
                    </owl:intersectionOf>
                </owl:Class>
            </rdfs:subClassOf>
            <rdfs:subClassOf rdf:resource="#Professor" />
        </owl:Class>

        <owl:Class rdf:ID="Chair">
            <rdfs:label>chair</rdfs:label>
            <rdfs:subClassOf>
                <owl:Class>
                    <owl:intersectionOf rdf:parseType="Collection">
                        <owl:Class rdf:about="#Person" />
                        <owl:Restriction>
                            <owl:onProperty rdf:resource="#headOf" />
                            <owl:someValuesFrom>
                                <owl:Class rdf:about="#Department" />
                            </owl:someValuesFrom>
                        </owl:Restriction>
                    </owl:intersectionOf>
                </owl:Class>
            </rdfs:subClassOf>
            <rdfs:subClassOf rdf:resource="#Professor" />
        </owl:Class>
    </rdf:RDF>'''

    nl = NeurolangPDL()
    nl.load_ontology(io.StringIO(owl))

    nl.add_tuple_set([('Juan',), ('Manuel',)], name='Dean')
    nl.add_tuple_set([('Miguel',), ('Alberto',)], name='Chair')
    nl.add_tuple_set([('College A',), ('College B',)], name='College')
    nl.add_tuple_set([('Department A',), ('Department B',)], name='Department')

    label = nl.new_symbol(name='rdf-schema:label')

    with nl.scope as e:
        e.answer[e.a] = (
            label[e.a, 'dean']
        )

        f_term = nl.solve_all()

    res = f_term['answer'].as_pandas_dataframe().values
    assert (res == [['Juan'], ['Manuel']]).all()


def test_retrieve_subclass():
    owl = '''<?xml version="1.0"?>
    <rdf:RDF xmlns="http://www.w3.org/2002/07/owl#"
        xmlns:owl="http://www.w3.org/2002/07/owl#"
        xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
        xmlns:rdfs="http://www.w3.org/2000/01/rdf-schema#">
        <Ontology>
            <versionInfo>0.3.1</versionInfo>
        </Ontology>

        <owl:Class rdf:ID="Dean">
            <rdfs:label>dean</rdfs:label>
            <rdfs:subClassOf>
                <owl:Class>
                    <owl:intersectionOf rdf:parseType="Collection">
                        <owl:Class rdf:about="#Person" />
                        <owl:Restriction>
                            <owl:onProperty rdf:resource="#headOf" />
                            <owl:someValuesFrom>
                                <owl:Class rdf:about="#College" />
                            </owl:someValuesFrom>
                        </owl:Restriction>
                    </owl:intersectionOf>
                </owl:Class>
            </rdfs:subClassOf>
            <rdfs:subClassOf rdf:resource="#Professor" />
        </owl:Class>

        <owl:Class rdf:ID="Chair">
            <rdfs:label>chair</rdfs:label>
            <rdfs:subClassOf>
                <owl:Class>
                    <owl:intersectionOf rdf:parseType="Collection">
                        <owl:Class rdf:about="#Person" />
                        <owl:Restriction>
                            <owl:onProperty rdf:resource="#headOf" />
                            <owl:someValuesFrom>
                                <owl:Class rdf:about="#Department" />
                            </owl:someValuesFrom>
                        </owl:Restriction>
                    </owl:intersectionOf>
                </owl:Class>
            </rdfs:subClassOf>
            <rdfs:subClassOf rdf:resource="#Professor" />
        </owl:Class>
    </rdf:RDF>'''

    nl = NeurolangPDL()
    nl.load_ontology(io.StringIO(owl))

    nl.add_tuple_set([('Juan',), ('Manuel',)], name='Dean')
    nl.add_tuple_set([('Miguel',), ('Alberto',)], name='Chair')
    nl.add_tuple_set([('College A',), ('College B',)], name='College')
    nl.add_tuple_set([('Department A',), ('Department B',)], name='Department')

    with nl.scope as e:
        e.answer[e.a] = (
            e.Professor[e.a]
        )

        f_term = nl.solve_all()

    res = f_term['answer'].as_pandas_dataframe().values
    assert ['Miguel'] in res
    assert ['Alberto'] in res
    assert ['Juan'] in res
    assert ['Manuel'] in res
    assert len(res) == 4


def test_knowledge_subclassof():
    owl = '''<?xml version="1.0"?>
    <rdf:RDF xmlns="http://www.w3.org/2002/07/owl#"
        xmlns:owl="http://www.w3.org/2002/07/owl#"
        xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
        xmlns:rdfs="http://www.w3.org/2000/01/rdf-schema#">
        <Ontology>
            <versionInfo>0.3.1</versionInfo>
        </Ontology>

        <owl:Class rdf:ID="Chair">
            <rdfs:label>chair</rdfs:label>
            <rdfs:subClassOf>
                <owl:Class>
                    <owl:intersectionOf rdf:parseType="Collection">
                        <owl:Class rdf:about="#Person" />
                        <owl:Restriction>
                            <owl:onProperty rdf:resource="#headOf" />
                            <owl:someValuesFrom>
                                <owl:Class rdf:about="#Department" />
                            </owl:someValuesFrom>
                        </owl:Restriction>
                    </owl:intersectionOf>
                </owl:Class>
            </rdfs:subClassOf>
            <rdfs:subClassOf rdf:resource="#Professor" />
        </owl:Class>
    </rdf:RDF>'''

    nl = NeurolangPDL()
    nl.load_ontology(io.StringIO(owl))

    subClassOf = nl.new_symbol(name='neurolang:subClassOf')
    with nl.scope as e:
        e.answer[e.a] = (
            subClassOf[e.a, 'Professor']
        )

        f_term = nl.query((e.a,), e.answer(e.a))

    res = f_term.as_pandas_dataframe().values
    assert (res == [['Chair']]).all()


def test_knowledge_property():
    owl = '''<?xml version="1.0"?>
    <rdf:RDF xmlns="http://www.w3.org/2002/07/owl#"
        xmlns:owl="http://www.w3.org/2002/07/owl#"
        xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
        xmlns:rdfs="http://www.w3.org/2000/01/rdf-schema#">
        <Ontology>
            <versionInfo>0.3.1</versionInfo>
        </Ontology>

        <owl:Class rdf:ID="Chair">
            <rdfs:label>chair</rdfs:label>
            <rdfs:subClassOf>
                <owl:Class>
                    <owl:intersectionOf rdf:parseType="Collection">
                        <owl:Class rdf:about="#Person" />
                        <owl:Restriction>
                            <owl:onProperty rdf:resource="#headOf" />
                            <owl:someValuesFrom>
                                <owl:Class rdf:about="#Department" />
                            </owl:someValuesFrom>
                        </owl:Restriction>
                    </owl:intersectionOf>
                </owl:Class>
            </rdfs:subClassOf>
            <rdfs:subClassOf rdf:resource="#Professor" />
        </owl:Class>
    </rdf:RDF>'''

    nl = NeurolangPDL()
    nl.load_ontology(io.StringIO(owl))

    label = nl.new_symbol(name='neurolang:label')
    with nl.scope as e:
        e.answer[e.a] = (
            label[e.a, 'chair']
        )

        f_term = nl.query((e.a,), e.answer[e.a])

    res = f_term.as_pandas_dataframe().values
    assert (res == [['Chair']]).all()


def test_entity_rules_inclusion():
    owl = '''<?xml version="1.0"?>
    <rdf:RDF xmlns="http://www.w3.org/2002/07/owl#"
        xmlns:owl="http://www.w3.org/2002/07/owl#"
        xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
        xmlns:rdfs="http://www.w3.org/2000/01/rdf-schema#">
        <Ontology>
            <versionInfo>0.3.1</versionInfo>
        </Ontology>
        <owl:Class rdf:ID="Chair">
            <rdfs:label>chair</rdfs:label>
            <rdfs:subClassOf>
                <owl:Class>
                    <owl:intersectionOf rdf:parseType="Collection">
                        <owl:Class rdf:about="#Person" />
                        <owl:Restriction>
                            <owl:onProperty rdf:resource="#headOf" />
                            <owl:someValuesFrom>
                                <owl:Class rdf:about="#Department" />
                            </owl:someValuesFrom>
                        </owl:Restriction>
                    </owl:intersectionOf>
                </owl:Class>
            </rdfs:subClassOf>
            <rdfs:subClassOf rdf:resource="#Professor" />
        </owl:Class>
    </rdf:RDF>'''

    nl = NeurolangPDL()

    entity = nl.load_ontology(io.StringIO(owl), connector_symbol_name='Entity')

    chair = Symbol('Chair')
    assert chair in nl.symbol_table

    assert len(nl.symbol_table[chair].formulas) == 1
    rule = nl.symbol_table[chair].formulas[0]

    assert isinstance(rule, Implication)
    assert rule.consequent.functor == chair
    assert len(rule.consequent.args) == 1 and rule.consequent.args[0].is_fresh

    fresh_var = rule.consequent.args[0]
    assert rule == Implication(chair(fresh_var), entity(fresh_var, Constant('chair')))


def test_connector_edb_populated_for_word_lower():
    """
    Regression test: loading an ontology should populate the connector
    symbol's EDB with (entity_name, lower_label) tuples so that queries
    like ``e.term == word_lower[e.onto_term]`` (which compile to an
    ExtendedProjection) can find matching rows via the connector.

    Before the fix, ``_parse_property`` created entity rules that used
    the connector symbol in their body but never inserted corresponding
    EDB facts for that symbol, causing ``lower_terms`` to return empty.
    """
    owl = '''<?xml version="1.0"?>
    <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
             xmlns:rdfs="http://www.w3.org/2000/01/rdf-schema#"
             xmlns:owl="http://www.w3.org/2002/07/owl#">
        <owl:Class rdf:about="http://example.org/Cognition">
            <rdfs:label>Cognition</rdfs:label>
        </owl:Class>
        <owl:Class rdf:about="http://example.org/Memory">
            <rdfs:label>Memory</rdfs:label>
        </owl:Class>
    </rdf:RDF>'''

    nl = NeurolangPDL()
    connector = nl.load_ontology(io.StringIO(owl))

    word_lower = nl.add_symbol(
        lambda x: x.lower(),
        name='word_lower',
        type_=Callable[[str], str],
    )

    with nl.scope as e:
        e.ontology_terms[e.term, e.onto_term] = connector(e.onto_term, e.term)
        e.lower_terms[e.term, e.onto_term] = (
            e.ontology_terms(e.term, e.onto_term)
            & (e.term == word_lower[e.onto_term])
        )
        result = nl.query(
            (e.term, e.onto_term), e.lower_terms(e.term, e.onto_term)
        )

    df = result.as_pandas_dataframe()
    rows = set(zip(df['term'], df['onto_term']))
    assert ('cognition', 'Cognition') in rows
    assert ('memory', 'Memory') in rows
    assert ('cognition', 'Cognition') in rows
    assert ('memory', 'Memory') in rows


def test_load_individuals_with_subclass_reasoning():
    nl = NeurolangPDL()
    nl.load_ontology(
        io.StringIO(KNOWLEDGE_GRAPH), format="turtle", individuals=True,
        names="prefixed",
    )
    employees = nl.execute_datalog_program("ans(e) :- pv:Employee(e)")
    assert set(employees) == {
        ("http://example.org/e1",), ("http://example.org/m1",)
    }
    names = nl.execute_datalog_program(
        "ans(n) :- pv:Employee(e), pv:name(e, n), (n != 'Ann')"
    )
    assert set(names) == {("Bob",)}
    managed = nl.execute_datalog_program(
        "ans(n) :- pv:manages(<http://example.org/m1>, e), pv:name(e, n)"
    )
    assert set(managed) == {("Ann",)}
    assert not any(
        getattr(s, "name", "").startswith("fresh") for s in nl.symbol_table
    )


def test_ontology_predicates_are_recorded():
    nl = NeurolangPDL()
    nl.add_tuple_set([("a", 1)], name="table")
    assert nl.ontology_predicates == set()
    nl.load_ontology(
        io.StringIO(KNOWLEDGE_GRAPH), format="turtle", individuals=True,
        names="prefixed",
    )
    assert {"pv:Employee", "pv:name", "pv:manages"} <= nl.ontology_predicates
    assert "table" not in nl.ontology_predicates


def test_negation_beside_a_class_with_subclasses():
    nl = NeurolangPDL()
    nl.load_ontology(
        io.StringIO(KNOWLEDGE_GRAPH), format="turtle", individuals=True,
        names="prefixed",
    )
    unmanaged = nl.execute_datalog_program(
        "managed(x) :- pv:manages(m, x)\n"
        "ans(e) :- pv:Employee(e), ~managed(e)"
    )
    assert set(unmanaged) == {("http://example.org/m1",)}


def test_query_a_property_the_ontology_only_constrains():
    """``headOf`` has a constraint, no rule and no fact: it is queried as
    an EDB predicate, not by magic sets, which found no rule to adorn. A
    Dean heads something, so is an answer when ``b`` is projected out,
    once there are Deans."""
    owl = '''<?xml version="1.0"?>
    <rdf:RDF xmlns:owl="http://www.w3.org/2002/07/owl#"
        xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
        xmlns:rdfs="http://www.w3.org/2000/01/rdf-schema#">
        <owl:Class rdf:ID="Dean">
            <rdfs:subClassOf>
                <owl:Restriction>
                    <owl:onProperty rdf:resource="#headOf" />
                    <owl:someValuesFrom rdf:resource="#College" />
                </owl:Restriction>
            </rdfs:subClassOf>
        </owl:Class>
    </rdf:RDF>'''
    nl = NeurolangPDL()
    nl.load_ontology(io.StringIO(owl))
    assert not set(nl.execute_datalog_program("ans(a) :- headOf(a, b)"))
    nl.add_tuple_set([('Juan',)], name='Dean')
    with nl.scope as e:
        e.answer[e.a, e.b] = e.headOf[e.a, e.b]
        expected = set(nl.solve_all().get('answer', ()))

    res = nl.execute_datalog_program("ans(a, b) :- headOf(a, b)")
    assert set(res) == expected
    res = nl.execute_datalog_program("ans(a) :- headOf(a, b)")
    assert set(res) == {('Juan',)}
    assert nl.execute_datalog_program("ans() :- headOf(a, b)")
    res = nl.execute_datalog_program("ans(count(a)) :- headOf(a, b)")
    assert set(res) == {(1,)}
    # b is distinguished by the head's term: the constraint does not apply.
    res = nl.execute_datalog_program("ans(count(b)) :- headOf(a, b)")
    assert set(res) == set()
    # the same projection written with exists, in the query's own rule
    res = nl.execute_datalog_program("ans(a) :- exists(b; headOf(a, b))")
    assert set(res) == {('Juan',)}
    res = nl.execute_datalog_program(
        "ans(a) :- Dean(a), exists(b; headOf(a, b))")
    assert set(res) == {('Juan',)}


DEAN_DEPT_UNIT = """
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix owl: <http://www.w3.org/2002/07/owl#> .
@prefix : <http://ex.org/> .
:Dean a owl:Class ; rdfs:subClassOf [
    a owl:Restriction ; owl:onProperty :headOf ; owl:someValuesFrom :Dept
] .
:Dept a owl:Class ; rdfs:subClassOf :Unit .
:Unit a owl:Class .
"""


def test_full_rule_on_the_value_an_existential_introduces():
    """``Dept ⊑ Unit`` is applied by the chase to named departments and by
    the rewriting to the unnamed one each Dean heads."""
    nl = NeurolangPDL()
    nl.load_ontology(io.StringIO(DEAN_DEPT_UNIT), format="turtle")
    nl.add_tuple_set([('Juan',)], name='Dean')
    nl.add_tuple_set([('Ana', 'Physics')], name='headOf')
    nl.add_tuple_set([('Physics',)], name='Dept')
    res = nl.execute_datalog_program("ans(x) :- headOf(x, y), Unit(y)")
    assert set(res) == {('Juan',), ('Ana',)}
    res = nl.execute_datalog_program("ans(x, y) :- headOf(x, y), Unit(y)")
    assert set(res) == {('Ana', 'Physics')}
    assert set(nl.execute_datalog_program("ans(y) :- Unit(y)")) == {
        ('Physics',)
    }
    # The count is over the union of the rewritings of its body.
    res = nl.execute_datalog_program("ans(count(x)) :- headOf(x, y)")
    assert set(res) == {(2,)}
    res = nl.execute_datalog_program(
        "ans(x, w) :- headOf(x, y), Unit(y), headOf(w, z), Unit(z)"
    )
    names = ('Ana', 'Juan')
    assert set(res) == {(a, b) for a in names for b in names}


def test_negation_and_solve_all_over_chased_constraints():
    """Negation is closed world over the derived facts, a negated atom
    does not stop the rewriting, and the predicates the chase adds stay
    out of the solution."""
    nl = NeurolangPDL()
    nl.load_ontology(io.StringIO(DEAN_DEPT_UNIT), format="turtle")
    nl.add_tuple_set([('Juan',)], name='Dean')
    nl.add_tuple_set([('Ana', 'Physics')], name='headOf')
    nl.add_tuple_set([('Physics',)], name='Dept')
    nl.add_tuple_set([('Physics',), ('Bio',), ('Juan',)], name='thing')
    res = nl.execute_datalog_program("ans(y) :- thing(y), ~Unit(y)")
    assert set(res) == {('Bio',), ('Juan',)}
    res = nl.execute_datalog_program("ans(x) :- headOf(x, y), ~thing(x)")
    assert set(res) == {('Ana',)}
    nl.execute_datalog_program(
        "leads(x) :- headOf(x, y), Unit(y)\n"
        "c(count(x)) :- headOf(x, y)"
    )
    solution = nl.solve_all()
    assert not any(name.startswith('fresh') for name in solution)
    assert set(solution['leads']) == {('Ana',), ('Juan',)}
    assert set(solution['c']) == {(2,)}


def test_intersection_class_holds_on_named_values():
    """``Chair ⊑ Person ⊓ ∃headOf.Department`` gives ``Chair ⊑ Person``,
    which the chase applies before ``Person ⊑ Agent``."""
    nl = NeurolangPDL()
    nl.load_ontology(io.StringIO("""
    @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
    @prefix owl: <http://www.w3.org/2002/07/owl#> .
    @prefix : <http://ex.org/> .
    :Chair a owl:Class ; rdfs:subClassOf [ a owl:Class ; owl:intersectionOf (
        :Person
        [ a owl:Restriction ; owl:onProperty :headOf ;
          owl:someValuesFrom :Department ]
    ) ] .
    :Person a owl:Class ; rdfs:subClassOf :Agent .
    """), format="turtle")
    nl.add_tuple_set([('Miguel',)], name='Chair')
    assert set(nl.execute_datalog_program("ans(x) :- Agent(x)")) == {
        ('Miguel',)
    }
    assert set(nl.execute_datalog_program(
        "ans(x) :- headOf(x, y), Department(y)"
    )) == {('Miguel',)}


def test_query_over_a_high_class_instance_ends():
    """A chain of 60 classes, each ``Ci ⊑ ∀p.Ci+1``, with an instance of
    the top: the full rules go to the chase, so the rewriting of ``p(x, y)``
    does not descend the hierarchy."""
    n = 60
    ttl = "".join(
        f":C{i} a owl:Class ; rdfs:subClassOf :C{i + 1} , [ a owl:Restriction"
        f" ; owl:onProperty :p ; owl:allValuesFrom :C{i + 1} ] , [ a "
        f"owl:Restriction ; owl:onProperty :p ; owl:someValuesFrom :C{i} ] .\n"
        for i in range(n)
    )
    nl = NeurolangPDL()
    nl.load_ontology(io.StringIO(
        "@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .\n"
        "@prefix owl: <http://www.w3.org/2002/07/owl#> .\n"
        "@prefix : <http://ex.org/> .\n" + ttl
    ), format="turtle")
    nl.add_tuple_set([('a',)], name='C0')
    assert set(nl.execute_datalog_program("ans(x) :- p(x, y)")) == {('a',)}
    assert set(nl.execute_datalog_program(f"ans(x) :- C{n}(x)")) == {('a',)}


def _random_horn_ontology(rng, n_classes=4, props=("p", "r")):
    """Axioms ``(kind, C, p, D)``: ``sub`` C ⊑ D, ``some`` C ⊑ ∃p.D,
    ``all`` C ⊑ ∀p.D, ``and`` C ⊑ B ⊓ ∃r.D with B as ``p``. The rules
    without existentials go from a class to one of higher index, so are
    not recursive: with recursive ones, Xrewrite need not end."""
    classes = [f"C{i}" for i in range(n_classes)]
    axioms = set()
    for _ in range(rng.randint(2, 6)):
        kind = rng.choice(("sub", "some", "all", "and"))
        c, d = sorted(rng.sample(classes, 2))
        if kind == "some":
            c, d = rng.choice(classes), rng.choice(classes)
        p = rng.choice(props)
        if kind == "and":
            c, p = sorted(rng.sample(classes, 2))
        axioms.add((kind, c, None if kind == "sub" else p, d))
    return classes, sorted(axioms, key=str)


PROPERTY_TURTLE = {
    "subp": ":{p} rdfs:subPropertyOf :{d} .",
    "eqp": ":{p} owl:equivalentProperty :{d} .",
    "inv": ":{p} owl:inverseOf :{d} .",
    "dom": ":{p} rdfs:domain :{d} .",
    "rng": ":{p} rdfs:range :{d} .",
    "sym": ":{p} a owl:SymmetricProperty .",
    "trans": ":{p} a owl:TransitiveProperty .",
}


def _random_property_axioms(rng, classes, props=("p", "r"), kinds=None):
    """1 to 3 property axioms ``(kind, None, p, d)``, ``d`` a property or,
    for ``dom`` and ``rng``, a class."""
    axioms = set()
    for _ in range(rng.randint(1, 3)):
        kind = rng.choice(kinds or sorted(PROPERTY_TURTLE))
        p, q = rng.sample(props, 2)
        d = rng.choice(classes) if kind in ("dom", "rng") else q
        axioms.add((kind, None, p, None if kind in ("sym", "trans") else d))
    return axioms


def _signature(axioms):
    """The predicates of the ``axioms``, some with no entailed tuple."""
    return {q for _, c, p, d in axioms for q in (c, p, d) if q} | {
        "r" for k, *_ in axioms if k == "and"
    }


def _property_consequences(kind, p, d, facts):
    """The facts a property axiom derives from the binary ``facts``."""
    pairs = {f[1:] for f in facts if f[0] == p and len(f) == 3}
    if kind == "subp":
        return {(d, s, o) for s, o in pairs}
    if kind == "eqp":
        return {(d, s, o) for s, o in pairs} | {
            (p,) + f[1:] for f in facts if f[0] == d and len(f) == 3
        }
    if kind == "inv":
        return {(d, o, s) for s, o in pairs} | {
            (p, f[2], f[1]) for f in facts if f[0] == d and len(f) == 3
        }
    if kind == "dom":
        return {(d, s) for s, _ in pairs}
    if kind == "rng":
        return {(d, o) for _, o in pairs}
    if kind == "sym":
        return {(p, o, s) for s, o in pairs}
    return {(p, s, u) for s, o in pairs for o2, u in pairs if o == o2}


def _turtle(axioms):
    lines = [
        "@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .",
        "@prefix owl: <http://www.w3.org/2002/07/owl#> .",
        "@prefix : <http://ex.org/> .",
    ]
    for kind, c, p, d in axioms:
        if kind in PROPERTY_TURTLE:
            lines.append(PROPERTY_TURTLE[kind].format(p=p, d=d))
            continue
        if kind == "sub":
            sup = f":{d}"
        elif kind == "and":
            sup = (
                f"[ a owl:Class ; owl:intersectionOf ( :{p} [ a "
                f"owl:Restriction ; owl:onProperty :r ; owl:someValuesFrom "
                f":{d} ] ) ]"
            )
        else:
            values = "someValuesFrom" if kind == "some" else "allValuesFrom"
            sup = (
                f"[ a owl:Restriction ; owl:onProperty :{p} ; "
                f"owl:{values} :{d} ]"
            )
        lines.append(f":{c} a owl:Class ; rdfs:subClassOf {sup} .")
    return "\n".join(lines)


def _certain_answers(axioms, facts, head, body, depth=6):
    """Chase ``facts`` with labelled nulls up to ``depth``
    nulls deep, match ``body`` and keep the answers without nulls."""
    facts = set(facts)
    depths = {}
    while True:
        new = set()
        for kind, c, p, d in axioms:
            if kind in PROPERTY_TURTLE:
                new |= _property_consequences(kind, p, d, facts)
            for f in facts:
                if f[0] != c:
                    continue
                t = f[1]
                if kind == "sub":
                    new.add((d, t))
                elif kind == "all":
                    new |= {(d, g[2]) for g in facts if g[:2] == (p, t)}
                elif kind == "has":
                    new.add((p, t, d))
                elif depths.get(t, 0) < depth:
                    if kind == "and":
                        new.add((p, t))
                    null = ("null", kind, c, p, d, t)
                    depths[null] = depths.get(t, 0) + 1
                    prop = "r" if kind == "and" else p
                    new |= {(prop, t, null), (d, null)}
        if new <= facts:
            break
        facts |= new

    def match(atoms, sub):
        if not atoms:
            yield sub
            return
        functor, *args = atoms[0]
        for f in facts:
            if f[0] != functor or len(f) != len(atoms[0]):
                continue
            s = dict(sub)
            if all(s.setdefault(a, v) == v for a, v in zip(args, f[1:])):
                yield from match(atoms[1:], s)

    return {
        tuple(s[v] for v in head) for s in match(body, {})
        if not any(isinstance(s[v], tuple) for v in head)
    }


def test_certain_answers_on_random_horn_ontologies():
    """NeuroLang's answers are the certain answers, computed by a chase
    with labelled nulls, on random ontologies, data and queries."""
    import random

    rng = random.Random(0)
    constants = ("a", "b", "c")
    for _ in range(40):
        classes, axioms = _random_horn_ontology(rng)
        facts = {(rng.choice(classes), rng.choice(constants))
                 for _ in range(rng.randint(1, 3))}
        facts |= {(rng.choice(("p", "r")),) + tuple(rng.sample(constants, 2))
                  for _ in range(rng.randint(0, 3))}
        predicates = {f[0] for f in facts} | _signature(axioms)
        nl = NeurolangPDL()
        nl.load_ontology(io.StringIO(_turtle(axioms)), format="turtle")
        for name in {f[0] for f in facts}:
            nl.add_tuple_set(
                [f[1:] for f in facts if f[0] == name], name=name
            )
        for _ in range(3):
            body = []
            for _ in range(rng.randint(1, 3)):
                name = rng.choice(sorted(predicates))
                arity = 2 if name in ("p", "r") else 1
                body.append((name,) + tuple(rng.choices("xyz", k=arity)))
            variables = sorted({v for atom in body for v in atom[1:]})
            head = rng.sample(variables, rng.randint(1, len(variables)))
            program = "ans({}) :- {}".format(", ".join(head), ", ".join(
                f"{a[0]}({', '.join(a[1:])})" for a in body
            ))
            expected = _certain_answers(axioms, facts, head, body)
            res = set(nl.execute_datalog_program(program))
            assert res == expected, (axioms, facts, program)


def test_certain_answers_with_property_axioms():
    """As above, with 1 to 3 property axioms: the answers are the certain
    ones, and only sound with a transitive property."""
    import random

    rng = random.Random(1)
    constants = ("a", "b", "c")
    for case in range(24):
        classes, axioms = _random_horn_ontology(rng)
        axioms = sorted(
            set(axioms) | _random_property_axioms(rng, classes), key=str
        )
        facts = {(rng.choice(classes), rng.choice(constants))
                 for _ in range(rng.randint(1, 3))}
        facts |= {(rng.choice(("p", "r")),) + tuple(rng.sample(constants, 2))
                  for _ in range(rng.randint(0, 3))}
        predicates = {f[0] for f in facts} | _signature(axioms)
        nl = NeurolangPDL()
        nl.load_ontology(io.StringIO(_turtle(axioms)), format="turtle")
        for name in {f[0] for f in facts}:
            nl.add_tuple_set(
                [f[1:] for f in facts if f[0] == name], name=name
            )
        for _ in range(2):
            body = []
            for _ in range(rng.randint(1, 2)):
                name = rng.choice(sorted(predicates))
                arity = 2 if name in ("p", "r") else 1
                body.append((name,) + tuple(rng.choices("xyz", k=arity)))
            variables = sorted({v for atom in body for v in atom[1:]})
            head = rng.sample(variables, rng.randint(1, len(variables)))
            program = "ans({}) :- {}".format(", ".join(head), ", ".join(
                f"{a[0]}({', '.join(a[1:])})" for a in body
            ))
            expected = _certain_answers(axioms, facts, head, body, depth=4)
            res = {tuple(r) for r in nl.execute_datalog_program(program)}
            if any(a[0] == "trans" for a in axioms):
                assert res <= expected, (axioms, facts, program)
            else:
                assert res == expected, (axioms, facts, program)


def test_has_value_binds_the_variable_in_all_the_query():
    """Rewriting ``p(x, w)`` with ``C0(x) → p(x, 'v')`` binds ``w`` to
    ``'v'`` in all of the query, so ``C0(w)`` is ``C0('v')``, false."""
    nl = NeurolangPDL()
    nl.load_ontology(io.StringIO("""
    @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
    @prefix owl: <http://www.w3.org/2002/07/owl#> .
    @prefix : <http://ex.org/> .
    :C0 a owl:Class ; rdfs:subClassOf [
        a owl:Restriction ; owl:onProperty :p ; owl:hasValue :v
    ] .
    """), format="turtle")
    nl.add_tuple_set([('a',)], name='C0')
    assert set(nl.execute_datalog_program("ans(x, w) :- p(x, w)")) == {
        ('a', 'v')
    }
    assert not set(nl.execute_datalog_program(
        "ans(x) :- p(x, w), C0(w)"
    ))


def _deans(axioms):
    """(I) with the property ``axioms``, Juan a Dean, Ana head of Physics
    and Physics head of Lab."""
    nl = NeurolangPDL()
    nl.load_ontology(
        io.StringIO(DEAN_DEPT_UNIT + axioms), format="turtle"
    )
    nl.add_tuple_set([('Juan',)], name='Dean')
    nl.add_tuple_set([('Ana', 'Physics'), ('Physics', 'Lab')], name='headOf')
    return nl


def _answers(nl, query):
    return {tuple(row) for row in nl.execute_datalog_program(query)}


def test_predicate_only_in_constraint_bodies_answers_empty():
    """``Dean`` and ``leads`` are in the signature but nothing entails a
    tuple of theirs: their certain answers are empty. ``headOf``, derived
    from ``leads``, has its answers once ``leads`` has facts."""
    nl = NeurolangPDL()
    nl.load_ontology(io.StringIO(
        DEAN_DEPT_UNIT + ":leads rdfs:subPropertyOf :headOf ."
    ), format="turtle")
    nl.add_tuple_set([('Ana', 'Physics')], name='headOf')
    assert _answers(nl, "ans(x) :- Dean(x)") == set()
    assert _answers(nl, "ans(x, y) :- leads(x, y), headOf(x, y)") == set()
    assert _answers(nl, "ans(x) :- headOf(x, y)") == {('Ana',)}
    nl.execute_datalog_program("lead(x) :- leads(x, y)")
    assert 'lead' not in nl.solve_all()
    nl.add_tuple_set([('Eva', 'Lab')], name='leads')
    nl.add_tuple_set([('Juan',)], name='Dean')
    assert _answers(nl, "ans(x) :- leads(x, y)") == {('Eva',)}
    assert _answers(nl, "ans(x) :- headOf(x, y)") == {
        ('Ana',), ('Eva',), ('Juan',)
    }


def test_sub_property_and_range_on_the_value_an_existential_introduces():
    nl = _deans(":headOf rdfs:subPropertyOf :worksFor .\n"
                ":worksFor rdfs:range :Org .")
    assert _answers(nl, "ans(x) :- worksFor(x, y), Org(y)") == {
        ('Ana',), ('Juan',), ('Physics',)
    }
    assert _answers(nl, "ans(y) :- Org(y)") == {('Physics',), ('Lab',)}
    assert _answers(nl, "ans(x, y) :- worksFor(x, y)") == {
        ('Ana', 'Physics'), ('Physics', 'Lab')
    }


def test_domain_holds_of_a_named_value_through_an_unnamed_one():
    """``headOf(x, y) → Manager(x)`` derives ``Manager('Juan')`` from the
    unnamed department Juan heads."""
    nl = _deans(":headOf rdfs:domain :Manager .")
    assert _answers(nl, "ans(x) :- Manager(x)") == {
        ('Ana',), ('Juan',), ('Physics',)
    }


def test_domain_fact_through_an_unnamed_value_is_chased():
    """The rewritings of ``headOf(x, y) → Manager(x)`` are chased, so the
    full constraints apply to ``Manager('Juan')``: ``Manager ⊑ Person``,
    and ``Manager ⊑ ∃runs.Team`` with ``runs``'s domain ``Lead``. Eva, a
    Manager by a fact, keeps it."""
    nl = _deans(
        ":headOf rdfs:domain :Manager .\n"
        ":Manager a owl:Class ; rdfs:subClassOf :Person , [ a "
        "owl:Restriction ; owl:onProperty :runs ; owl:someValuesFrom :Team"
        " ] .\n:runs rdfs:domain :Lead ."
    )
    nl.add_tuple_set([('Eva',)], name='Manager')
    named = {('Ana',), ('Juan',), ('Physics',), ('Eva',)}
    assert _answers(nl, "ans(x) :- Manager(x)") == named
    assert _answers(nl, "ans(x) :- Person(x)") == named
    assert _answers(nl, "ans(x) :- Lead(x)") == named


def test_all_values_through_an_inverse_is_not_found():
    """``Dept ⊑ ∀headedBy.Boss`` with ``headedBy`` the inverse of
    ``headOf`` makes Juan a Boss through his unnamed department: a rule of
    two atoms from an unnamed value to a named one, outside the
    completeness fragment of doc/ontologies.rst."""
    nl = _deans(
        ":headedBy owl:inverseOf :headOf .\n:Dept rdfs:subClassOf [ a "
        "owl:Restriction ; owl:onProperty :headedBy ; owl:allValuesFrom "
        ":Boss ] ."
    )
    assert _answers(nl, "ans(x) :- Boss(x)") == set()


def test_inverse_and_symmetric_properties_on_an_unnamed_value():
    nl = _deans(":headedBy owl:inverseOf :headOf ; rdfs:range :Boss .")
    assert _answers(nl, "ans(x) :- headedBy(y, x)") == {
        ('Ana',), ('Juan',), ('Physics',)
    }
    assert _answers(nl, "ans(x) :- Boss(x)") == {
        ('Ana',), ('Juan',), ('Physics',)
    }
    assert _answers(nl, "ans(x, y) :- headedBy(x, y)") == {
        ('Physics', 'Ana'), ('Lab', 'Physics')
    }
    nl = _deans(":headOf a owl:SymmetricProperty ; rdfs:range :R .")
    assert _answers(nl, "ans(x) :- headOf(y, x)") == {
        ('Ana',), ('Juan',), ('Physics',), ('Lab',)
    }
    assert _answers(nl, "ans(x) :- R(x)") == {
        ('Ana',), ('Juan',), ('Physics',), ('Lab',)
    }
    nl = _deans(":headOf owl:equivalentProperty :leads .")
    assert _answers(nl, "ans(x) :- leads(x, y)") == {
        ('Ana',), ('Juan',), ('Physics',)
    }


def test_dead_rewritings_with_property_axioms_keep_a_shared_unnamed_value():
    """A shared variable reaches the existential position of an atom through
    an inverse, a sub-property or a range, so the rewritings that join on
    the unnamed value are not dropped as dead."""
    nl = _deans(":headedBy owl:inverseOf :headOf .\n"
                ":headOf rdfs:subPropertyOf :worksFor .\n"
                ":worksFor rdfs:range :Org .")
    assert _answers(nl, "ans(x) :- headedBy(y, x), Org(y)") == {
        ('Ana',), ('Juan',), ('Physics',)
    }
    assert _answers(nl, "ans(x) :- worksFor(x, y), headedBy(y, z)") == {
        ('Ana',), ('Juan',), ('Physics',)
    }


def test_transitive_property_is_chased():
    """Transitivity is a Datalog rule of the chase, never a rewriting
    step, which would not end. It is not applied through unnamed values:
    with symmetry, ``headOf('Juan', 'Juan')`` holds through Juan's unnamed
    department, and is not found."""
    nl = _deans(":headOf a owl:TransitiveProperty .")
    assert _answers(nl, "ans(x, y) :- headOf(x, y)") == {
        ('Ana', 'Physics'), ('Physics', 'Lab'), ('Ana', 'Lab')
    }
    assert _answers(nl, "ans(x) :- headOf(x, y)") == {
        ('Ana',), ('Juan',), ('Physics',)
    }
    nl = _deans(":headOf a owl:TransitiveProperty, owl:SymmetricProperty .")
    assert _answers(nl, "ans(x) :- headOf(x, x)") == {
        ('Ana',), ('Physics',), ('Lab',)
    }


CONTRAST_MEASURES = """
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix owl: <http://www.w3.org/2002/07/owl#> .
@prefix c: <http://www.cognitiveatlas.org/ontology/cogat.owl#> .
c:K a owl:Class ; rdfs:subClassOf [
    a owl:Restriction ; owl:onProperty c:measures ; owl:someValuesFrom c:WM
] .
c:WM a owl:Class ; rdfs:subClassOf c:CC .
c:CC a owl:Class .
"""


def _contrast_measures():
    nl = NeurolangPDL()
    nl.load_ontology(
        io.StringIO(CONTRAST_MEASURES), format="turtle", individuals=True,
        names="prefixed",
    )
    nl.add_tuple_set([("s1",)], name="two_back")
    return nl


def test_constraints_apply_to_the_instances_a_rule_derives():
    """``c:K(s) :- two_back(s)`` makes ``s1`` a ``c:K``, which measures
    some ``c:WM``, a ``c:CC``: the rule is solved with the query, in the
    same program or an earlier one."""
    queries = {
        "ans(s) :- c:measures(s, c)": {("s1",)},
        "ans(s) :- c:measures(s, c), c:CC(c)": {("s1",)},
        "ans(s, c) :- c:measures(s, c), c:CC(c)": set(),
    }
    nl = _contrast_measures()
    for query, expected in queries.items():
        with nl.scope:
            assert set(nl.execute_datalog_program(
                "c:K(s) :- two_back(s)\n" + query
            )) == expected, query
    nl = _contrast_measures()
    nl.execute_datalog_program("mid(s) :- two_back(s)\nc:K(s) :- mid(s)")
    for query, expected in queries.items():
        assert set(nl.execute_datalog_program(query)) == expected, query
    assert nl.execute_datalog_program("ans() :- c:measures('s1', c)")
    assert not nl.execute_datalog_program("ans() :- c:measures('s2', c)")


def test_full_constraints_apply_to_the_instances_a_rule_derives():
    nl = _contrast_measures()
    nl.add_tuple_set([("w",)], name="f")
    assert set(nl.execute_datalog_program(
        "c:WM(x) :- f(x)\nans(x) :- c:CC(x)"
    )) == {("w",)}


def test_a_class_named_by_its_iri():
    """Without a prefix for their namespace, the classes and properties
    are named by their IRI, written ``<iri>`` as the functor of an atom;
    ``prefixes`` names them with a prefix."""
    cogat = "http://www.cognitiveatlas.org/ontology/cogat.owl#"
    ttl = re.sub(
        r"\bc:(\w+)", f"<{cogat}\\1>",
        CONTRAST_MEASURES.replace(f"@prefix c: <{cogat}> .", ""),
    )
    nl = NeurolangPDL()
    nl.load_ontology(
        io.StringIO(ttl), format="turtle", individuals=True,
        names="prefixed",
    )
    nl.add_tuple_set([("s1",)], name="two_back")
    assert set(nl.execute_datalog_program(
        f"<{cogat}K>(s) :- two_back(s)\n"
        f"ans(s) :- <{cogat}measures>(s, c), <{cogat}CC>(c)"
    )) == {("s1",)}
    nl = NeurolangPDL()
    nl.load_ontology(
        io.StringIO(ttl), format="turtle", individuals=True,
        names="prefixed", prefixes={"cogat": cogat},
    )
    nl.add_tuple_set([("s1",)], name="two_back")
    assert set(nl.execute_datalog_program(
        "cogat:K(s) :- two_back(s)\n"
        "ans(s) :- cogat:measures(s, c), cogat:CC(c)"
    )) == {("s1",)}


CONTRAST_OF_A_TASK = """
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix owl: <http://www.w3.org/2002/07/owl#> .
@prefix c: <http://www.cognitiveatlas.org/ontology/cogat.owl#> .
c:Cond a owl:Class ; rdfs:subClassOf [
    a owl:Restriction ; owl:onProperty c:part_of ; owl:someValuesFrom c:T
] .
c:T a owl:Class ; rdfs:subClassOf [
    a owl:Restriction ; owl:onProperty c:has_contrast ;
    owl:someValuesFrom c:K
] .
c:K a owl:Class ; rdfs:subClassOf [
    a owl:Restriction ; owl:onProperty c:measures ; owl:someValuesFrom c:WM
] .
c:WM a owl:Class ; rdfs:subClassOf c:CC .
c:CC a owl:Class .
"""


def test_a_chain_of_existentials_to_what_a_task_measures():
    """The Cognitive Atlas's condition, part of a task, which has a
    contrast that measures a concept: the condition itself measures
    nothing (an N-back condition is not its contrast)."""
    for program in ("", "c:Cond(s) :- d(s)\n"):
        nl = NeurolangPDL()
        nl.load_ontology(
            io.StringIO(CONTRAST_OF_A_TASK), format="turtle",
            individuals=True, names="prefixed",
        )
        nl.add_tuple_set([("s1",)], name="d" if program else "c:Cond")
        assert set(nl.execute_datalog_program(
            program + "ans(s) :- c:part_of(s, t), c:has_contrast(t, k), "
            "c:measures(k, c), c:CC(c)"
        )) == {("s1",)}
        assert not set(nl.execute_datalog_program(
            program + "ans(s) :- c:measures(s, c)"
        ))


DEANS_WITH_LABELS = """
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix owl: <http://www.w3.org/2002/07/owl#> .
@prefix : <http://ex.org/> .
:Dean a owl:Class ; rdfs:label "dean" ; rdfs:subClassOf :Professor , [
    a owl:Restriction ; owl:onProperty :headOf ; owl:someValuesFrom :Unit ] .
:Professor a owl:Class ; rdfs:label "professor" .
:Unit a owl:Class ; rdfs:label "unit" .
:Juan a :Dean .
"""
TYPED_BY_LABEL = "rdf:type(p, c) :- staff(p, n), rdfs:label(c, n)\n"


def _labelled_deans():
    nl = NeurolangPDL()
    nl.load_ontology(
        io.StringIO(DEANS_WITH_LABELS), format="turtle", individuals=True,
        names="prefixed", prefixes={"ex": "http://ex.org/"},
    )
    nl.add_tuple_set([("Ana", "dean"), ("Luis", "professor")], name="staff")
    return nl


def _rows(nl, program):
    with nl.scope:
        return set(nl.execute_datalog_program(program))


def test_rdf_type_rules_make_instances_of_the_classes():
    """``rdf:type(x, C)`` makes ``x`` an instance of ``C``, as RDF says,
    for the types a rule states as for those of the file: the subclass
    and the existential axioms apply to both."""
    nl = _labelled_deans()
    juan = "http://ex.org/Juan"
    assert _rows(nl, "ans(x) :- ex:Dean(x)") == {(juan,)}
    assert _rows(nl, TYPED_BY_LABEL + "ans(x) :- ex:Dean(x)") == {
        (juan,), ("Ana",)}
    assert _rows(nl, TYPED_BY_LABEL + "ans(x) :- ex:Professor(x)") == {
        (juan,), ("Ana",), ("Luis",)}
    assert _rows(
        nl, TYPED_BY_LABEL + "ans(x) :- exists(y; ex:headOf(x, y))"
    ) == {(juan,), ("Ana",)}
    # the program's rules are those of its query: the file's types stay
    assert _rows(nl, "ans(x) :- ex:Dean(x)") == {(juan,)}


def test_rdf_type_in_a_question_names_the_class_by_its_label():
    """``rdf:type(y, k)``, ``k`` fixed by its label, applies to the unnamed
    unit a dean heads; a label of no class gives no answer."""
    nl = _labelled_deans()
    heads = "ans(x) :- exists(y, k; ex:headOf(x, y), rdf:type(y, k), "
    assert _rows(nl, TYPED_BY_LABEL + heads + "rdfs:label(k, 'unit'))") == {
        ("http://ex.org/Juan",), ("Ana",)}
    assert _rows(nl, heads + "rdfs:label(k, 'nothing'))") == set()


def test_rdf_type_from_a_predicate_the_ontology_constrains_is_refused():
    nl = _labelled_deans()
    with pytest.raises(NeuroLangException, match="ontology constrains"):
        _rows(nl, "rdf:type(x, <http://ex.org/Unit>) :- ex:Dean(x)\n"
                  "ans(x) :- ex:Unit(x)")


def test_a_type_stated_between_two_queries_is_seen():
    nl = _labelled_deans()
    assert _rows(nl, "ans(x) :- ex:Dean(x)") == {("http://ex.org/Juan",)}
    nl.execute_datalog_program(TYPED_BY_LABEL)
    assert _rows(nl, "ans(x) :- ex:Dean(x)") == {
        ("http://ex.org/Juan",), ("Ana",)}


def test_rdf_type_in_a_rule_of_the_program_names_the_class():
    """A rule of the program, not only the query's, finds the class of
    ``rdf:type(y, k)`` by its label, as a condition of the query does."""
    nl = _labelled_deans()
    assert _rows(nl, TYPED_BY_LABEL + (
        "heads_a_unit(x) :- exists(y, k; ex:headOf(x, y), rdf:type(y, k), "
        "rdfs:label(k, 'unit'))\nans(x) :- heads_a_unit(x)"
    )) == {("http://ex.org/Juan",), ("Ana",)}


def test_a_rule_reading_a_constant_relation_is_rewritten():
    """A rule whose body reads a constant relation of strings goes through
    the rewriting of the ontology's constraints, in a deterministic and in
    a probabilistic query."""
    owl = '''<?xml version="1.0"?>
    <rdf:RDF xmlns="http://www.w3.org/2002/07/owl#"
        xmlns:owl="http://www.w3.org/2002/07/owl#"
        xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
        xmlns:rdfs="http://www.w3.org/2000/01/rdf-schema#">
        <owl:Class rdf:ID="Dean">
            <rdfs:subClassOf rdf:resource="#Person" />
        </owl:Class>
    </rdf:RDF>'''
    nl = NeurolangPDL()
    nl.load_ontology(io.StringIO(owl))
    nl.add_tuple_set(
        [("ann", "xterm"), ("bob", "yterm"), ("cid", "zterm")], name="label"
    )
    nl.add_tuple_set([("ann",), ("cid",)], name="Person")
    nl.add_tuple_set([("xterm",), ("yterm",)], name="selected")
    terms = Constant[AbstractSet[Tuple[str]]](
        WrappedRelationalAlgebraSet(
            [("xterm",), ("zterm",)] + [(f"term{i}",) for i in range(20)]
        ),
        verify_type=False,
    )
    label, person, r = Symbol("label"), Symbol("Person"), Symbol("r")
    a, t = Symbol("a"), Symbol("t")
    nl.program_ir.walk(Implication(
        r(a, t), Conjunction((label(a, t), person(a), terms(t)))
    ))

    with nl.scope as e:
        e.ans[e.a] = e.r[e.a, e.t]
        res = nl.query((e.a,), e.ans[e.a])
    assert set(res) == {("ann",), ("cid",)}

    with nl.scope as e:
        e.p[e.a, e.PROB[e.a]] = e.r[e.a, e.t] // e.selected[e.t]
        res = nl.query((e.a, e.q), e.p[e.a, e.q])
    assert set(res) == {("ann", 1.0)}


def test_rules_reading_two_constant_relations_are_rewritten_apart():
    """Two rules differing in the middle rows of a constant relation, the
    same in its printed form, are two rewritings: the second query does
    not get the first one's."""
    owl = '''<?xml version="1.0"?>
    <rdf:RDF xmlns="http://www.w3.org/2002/07/owl#"
        xmlns:owl="http://www.w3.org/2002/07/owl#"
        xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
        xmlns:rdfs="http://www.w3.org/2000/01/rdf-schema#">
        <owl:Class rdf:ID="Dean">
            <rdfs:subClassOf rdf:resource="#Person" />
        </owl:Class>
    </rdf:RDF>'''
    nl = NeurolangPDL()
    nl.load_ontology(io.StringIO(owl))
    nl.add_tuple_set([(f"p{i}", f"t{i}") for i in range(100)], name="label")
    nl.add_tuple_set([(f"p{i}",) for i in range(100)], name="Person")
    label, person, r = Symbol("label"), Symbol("Person"), Symbol("r")
    a, t = Symbol("a"), Symbol("t")
    for left_out in ("p50", "p51"):
        terms = Constant[AbstractSet[Tuple[str]]](
            WrappedRelationalAlgebraSet([
                (f"t{i}",) for i in range(100) if f"p{i}" != left_out
            ]),
            verify_type=False,
        )
        with nl.scope:
            nl.program_ir.walk(Implication(
                r(a, t), Conjunction((label(a, t), person(a), terms(t)))
            ))
            with nl.scope as e:
                e.ans[e.a] = e.r[e.a, e.t]
                res = {x for x, in nl.query((e.a,), e.ans[e.a])}
        assert left_out not in res and len(res) == 99
