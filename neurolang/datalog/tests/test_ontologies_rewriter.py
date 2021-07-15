import io

from ... import expression_walker as ew
from ...expression_walker import ReplaceExpressionWalker
from ...expressions import Constant, ExpressionBlock, Symbol
from ...logic import (Conjunction, ExistentialPredicate, FunctionApplication,
                      Implication)
from ...logic.transformations import CollapseConjunctions
from ..aggregation import DatalogWithAggregationMixin
from ..expressions import TranslateToLogic
from ..ontologies_parser import OntologyParser, RightImplication
from ..ontologies_rewriter import OntologyRewriter

S_ = Symbol
C_ = Constant
EP_ = ExistentialPredicate
EB_ = ExpressionBlock
FA_ = FunctionApplication
I_ = Implication
RI_ = RightImplication


class DatalogTranslator(
    TranslateToLogic, ew.IdentityWalker, DatalogWithAggregationMixin
):
    pass

def  _categorize_constraints(formulas):
    parsed_constraints = {}
    for sigma in formulas:
        sigma_functor = sigma.consequent.functor.name
        if sigma_functor in parsed_constraints:
            cons_set = parsed_constraints[sigma_functor]
            if sigma not in cons_set:
                cons_set.add(sigma)
                parsed_constraints[sigma_functor] = cons_set
        else:
            parsed_constraints[sigma_functor] = set([sigma])

    return parsed_constraints

def test_normal_rewriting_step():
    project = S_("project")
    inArea = S_("inArea")
    hasCollaborator = S_("hasCollaborator")
    p = S_("p")

    x = S_("x")
    y = S_("y")
    z = S_("z")
    a = S_("a")
    b = S_("b")
    db = C_("db")

    sigma = RI_(project(x) & inArea(x, y), hasCollaborator(z, y, x))
    q = I_(p(b), hasCollaborator(a, db, b))

    qB = EB_((q,))

    dt = DatalogTranslator()
    qB = dt.walk(qB)

    sigmaB = _categorize_constraints([sigma])

    orw = OntologyRewriter(qB, sigmaB)
    rewrite = orw.Xrewrite()

    assert len(rewrite) == 2
    imp1 = rewrite.pop()
    imp2 = rewrite.pop()
    assert imp1 == q or imp2 == q
    q2 = I_(p(b), project(b) & inArea(b, db))
    assert imp1 == q2 or imp2 == q2


def test_more_than_one_free_variable():
    project = S_("project")
    inArea = S_("inArea")
    hasCollaborator = S_("hasCollaborator")
    p = S_("p")

    w = S_("w")
    x = S_("x")
    y = S_("y")
    z = S_("z")
    a = S_("a")
    b = S_("b")
    c = S_("c")
    db = C_("db")

    sigma = RI_(project(x) & inArea(x, y), hasCollaborator(w, z, y, x))
    q = I_(p(b), hasCollaborator(c, a, db, b))

    qB = EB_((q,))

    dt = DatalogTranslator()
    qB = dt.walk(qB)

    sigmaB = _categorize_constraints([sigma])

    orw = OntologyRewriter(qB, sigmaB)
    rewrite = orw.Xrewrite()

    assert len(rewrite) == 2
    imp1 = rewrite.pop()
    imp2 = rewrite.pop()
    assert imp1 == q or imp2 == q
    q2 = I_(p(b), project(b) & inArea(b, db))
    assert imp1 == q2 or imp2 == q2


def test_unsound_rewriting_step_constant():
    project = S_("project")
    inArea = S_("inArea")
    hasCollaborator = S_("hasCollaborator")
    p = S_("p")

    x = S_("x")
    y = S_("y")
    z = S_("z")
    b = S_("b")
    db = C_("db")
    c = C_("c")

    sigma = RI_(project(x) & inArea(x, y), hasCollaborator(z, y, x))
    q = I_(p(b), hasCollaborator(c, db, b))

    qB = EB_((q,))

    dt = DatalogTranslator()
    qB = dt.walk(qB)

    sigmaB = _categorize_constraints([sigma])

    orw = OntologyRewriter(qB, sigmaB)
    rewrite = orw.Xrewrite()

    assert len(rewrite) == 1
    imp = rewrite.pop()
    assert imp == q


def test_unsound_rewriting_step_shared():
    project = S_("project")
    inArea = S_("inArea")
    hasCollaborator = S_("hasCollaborator")
    p = S_("p")

    x = S_("x")
    y = S_("y")
    z = S_("z")
    b = S_("b")
    db = C_("db")

    sigma = RI_(project(x) & inArea(x, y), hasCollaborator(z, y, x))
    q = I_(p(b), hasCollaborator(b, db, b))

    qB = EB_((q,))

    dt = DatalogTranslator()
    qB = dt.walk(qB)

    sigmaB = _categorize_constraints([sigma])

    orw = OntologyRewriter(qB, sigmaB)
    rewrite = orw.Xrewrite()

    assert len(rewrite) == 1
    imp = rewrite.pop()
    assert imp == q


def test_outside_variable():
    s = S_("s")
    r = S_("r")
    t = S_("t")
    p = S_("p")

    x = S_("x")
    y = S_("y")
    z = S_("z")

    a = S_("a")
    b = S_("b")
    c = S_("c")
    e = S_("e")

    sigma = RI_(s(x) & r(x, y), t(x, y, z))
    q2 = I_(p(a), s(c) & t(a, b, c) & t(a, e, c))

    qB = EB_((q2,))

    dt = DatalogTranslator()
    qB = dt.walk(qB)

    sigmaB = _categorize_constraints([sigma])

    orw = OntologyRewriter(qB, sigmaB)
    rewrite = orw.Xrewrite()

    assert len(rewrite) == 1


def test_example_4_3():
    project = S_("project")
    inArea = S_("inArea")
    hasCollaborator = S_("hasCollaborator")
    collaborator = S_("collaborator")
    p = S_("p")

    x = S_("x")
    y = S_("y")
    z = S_("z")
    a = S_("a")
    b = S_("b")
    c = S_("c")

    sigma1 = RI_(project(x) & inArea(x, y), hasCollaborator(z, y, x))
    sigma2 = RI_(hasCollaborator(x, y, z), collaborator(x))

    q = I_(p(b, c), hasCollaborator(a, b, c) & collaborator(a))

    qB = EB_((q,))

    dt = DatalogTranslator()
    qB = dt.walk(qB)

    sigmaB = _categorize_constraints([sigma1, sigma2])

    orw = OntologyRewriter(qB, sigmaB)
    rewrite = orw.Xrewrite()

    assert len(rewrite) == 4


def test_infinite_walker():

    subClassOf = Symbol(str("subClassOf"))
    rest = Symbol("rest")
    reg = Symbol("reg")

    x1 = Symbol("x1")
    y1 = Symbol("y1")
    x = Symbol("x")
    y = Symbol("y")

    sigma_ant = Conjunction((subClassOf(x1, y1), rest(x1, y1), rest(y1, y1)))
    S = subClassOf(x1, y1)
    q_ant = Conjunction((subClassOf(x1, y1), rest(y, x1), rest(x, y1), reg(x)))

    replace = dict({S: sigma_ant})
    rsw = ReplaceExpressionWalker(replace)
    sigma_rep = rsw.walk(q_ant)
    sigma_rep = CollapseConjunctions().walk(sigma_rep)

    expected = Conjunction((sigma_ant, rest(y, x1), rest(x, y1), reg(x)))
    expected = CollapseConjunctions().walk(expected)

    assert sigma_rep == expected


def test_distinguished_variables():
    project = S_("project")
    inArea = S_("inArea")
    hasCollaborator = S_("hasCollaborator")
    p = S_("p")

    x = S_("x")
    y = S_("y")
    a = S_("a")
    b = S_("b")

    sigma = RI_(project(x), hasCollaborator(y, x))
    q = I_(p(a), hasCollaborator(a, b) & inArea(b))

    qB = EB_((q,))

    dt = DatalogTranslator()
    qB = dt.walk(qB)

    sigmaB = _categorize_constraints([sigma])

    orw = OntologyRewriter(qB, sigmaB)
    rewrite = orw.Xrewrite()

    assert len(rewrite) == 1
    imp1 = rewrite.pop()
    assert imp1 == qB.formulas[0]

def test_empty_rewrite():
    owl = '''<?xml version="1.0"?>
    <rdf:RDF xmlns="http://www.w3.org/2002/07/owl#"
        xmlns:owl="http://www.w3.org/2002/07/owl#"
        xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
        xmlns:rdfs="http://www.w3.org/2000/01/rdf-schema#">
        <Ontology>
            <versionInfo>0.3.1</versionInfo>
        </Ontology>

        <owl:Class rdf:ID="AdministrativeStaff">
            <rdfs:label>administrative staff worker</rdfs:label>
            <rdfs:subClassOf rdf:resource="#Employee"/>
        </owl:Class>

        <owl:Class rdf:ID="Article">
            <rdfs:label>article</rdfs:label>
            <rdfs:subClassOf rdf:resource="#Publication"/>
        </owl:Class>

        <owl:Class rdf:ID="AssistantProfessor">
            <rdfs:label>assistant professor</rdfs:label>
            <rdfs:subClassOf rdf:resource="#Professor"/>
        </owl:Class>

        <owl:Class rdf:ID="AssociateProfessor">
            <rdfs:label>associate professor</rdfs:label>
            <rdfs:subClassOf rdf:resource="#Professor"/>
        </owl:Class>

        <owl:Class rdf:ID="Book">
            <rdfs:label>book</rdfs:label>
            <rdfs:subClassOf rdf:resource="#Publication"/>
        </owl:Class>
    </rdf:RDF>'''

    book = Symbol('book')
    x = Symbol('x')
    p = Symbol('p')

    onto = OntologyParser(io.StringIO(owl))
    constraints = onto.parse_ontology()

    q = I_(p(x), book(x))

    qB = EB_((q,))

    dt = DatalogTranslator()
    qB = dt.walk(qB)

    orw = OntologyRewriter(qB, constraints)
    rewrited = orw.Xrewrite()

    assert len(rewrited) == 1
    imp = rewrited.pop()
    assert imp == q


def test_ontology_parsed_rewrite():
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

    headof = Symbol('headOf')
    chair = Symbol('Chair')
    x = Symbol('x')
    y = Symbol('y')
    p = Symbol('p')

    onto = OntologyParser(io.StringIO(owl))
    constraints = onto.parse_ontology()

    q = I_(p(x), headof(x, y))

    qB = EB_((q,))

    dt = DatalogTranslator()
    qB = dt.walk(qB)

    orw = OntologyRewriter(qB, constraints)
    rewrited = orw.Xrewrite()

    rewrited = list(rewrited)

    assert len(rewrited) == 3
    assert q in rewrited
    index_no_q = [i for i, e in enumerate(rewrited) if e != q]
    assert len(index_no_q) == 2
    index_1 = index_no_q[0]
    index_2 = index_no_q[1]

    if rewrited[index_1].antecedent == chair(x):
        assert len(rewrited[index_2].antecedent.args) == 2
        assert rewrited[index_2].antecedent.args[0] == rewrited[index_2].consequent.args[0]
    elif rewrited[index_2].antecedent == chair(x):
        assert len(rewrited[index_1].antecedent.args) == 2
        assert rewrited[index_1].antecedent.args[0] == rewrited[index_1].consequent.args[0]
    else:
        assert False
