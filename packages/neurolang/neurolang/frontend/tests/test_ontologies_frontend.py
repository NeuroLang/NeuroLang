"""Ontologies loaded into a frontend: the parser's tests are in
``neurolang/datalog/tests/test_ontologies_parser.py``."""

import io
from typing import Callable
from nl_expr.logic import Implication

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
