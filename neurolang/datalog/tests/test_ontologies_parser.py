
import io

from ...expressions import Constant, Symbol
from ...frontend import NeurolangPDL
from ..constraints_representation import RightImplication
from ..ontologies_parser import OntologyParser


def test_1():
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

    x = Symbol.fresh()
    Employee = Symbol('Employee')
    AdministrativeStaff = Symbol('AdministrativeStaff')
    Publication = Symbol('Publication')
    Article = Symbol('Article')
    Professor = Symbol('Professor')
    AssistantProfessor = Symbol('AssistantProfessor')
    AssociateProfessor = Symbol('AssociateProfessor')
    Book = Symbol('Book')

    label = Symbol('rdf-schema:label')

    imp1 = RightImplication(AdministrativeStaff(x), Employee(x))
    imp2 = RightImplication(Article(x), Publication(x))
    imp3 = RightImplication(AssistantProfessor(x), Professor(x))
    imp4 = RightImplication(AssociateProfessor(x), Professor(x))
    imp5 = RightImplication(Book(x), Publication(x))


    imp_label1 = RightImplication(AdministrativeStaff(x), label(x, Constant('administrative staff worker')))
    imp_label2 = RightImplication(AssistantProfessor(x), label(x, Constant('assistant professor')))
    imp_label3 = RightImplication(AssociateProfessor(x), label(x, Constant('associate professor')))
    imp_label4 = RightImplication(Book(x), label(x, Constant('book')))
    imp_label5 = RightImplication(Article(x), label(x, Constant('article')))

    onto = OntologyParser(io.StringIO(owl))
    constraints, _ = onto.parse_ontology()

    assert set(constraints.keys()) == set(['Employee', 'Publication', 'Professor', 'rdf-schema:label'])

    AdminConstraint = constraints['Employee']
    assert len(AdminConstraint) == 1 and isinstance(AdminConstraint, set)
    AdminConstraint = next(iter(AdminConstraint))
    assert isinstance(AdminConstraint, RightImplication)
    assert len(AdminConstraint.antecedent.args) == 1
    assert len(AdminConstraint.consequent.args) == 1
    assert AdminConstraint.antecedent.args[0] == AdminConstraint.consequent.args[0]
    assert AdminConstraint.antecedent.functor == imp1.antecedent.functor
    assert AdminConstraint.consequent.functor == imp1.consequent.functor

    PublicationConstraint = constraints['Publication']
    assert len(PublicationConstraint) == 2 and isinstance(PublicationConstraint, set)
    for pc in PublicationConstraint:
        assert isinstance(pc, RightImplication)
        assert len(pc.antecedent.args) == 1
        assert len(pc.consequent.args) == 1
        assert pc.antecedent.args[0] == pc.consequent.args[0]
        if pc.antecedent.functor == Article:
            assert pc.antecedent.functor == imp2.antecedent.functor
            assert pc.consequent.functor == imp2.consequent.functor
        elif pc.antecedent.functor == Book:
            assert pc.antecedent.functor == imp5.antecedent.functor
            assert pc.consequent.functor == imp5.consequent.functor
        else:
            assert False

    ProfessorConstraint = constraints['Professor']
    assert len(ProfessorConstraint) == 2 and isinstance(ProfessorConstraint, set)
    for pc in ProfessorConstraint:
        assert isinstance(pc, RightImplication)
        assert len(pc.antecedent.args) == 1
        assert len(pc.consequent.args) == 1
        assert pc.antecedent.args[0] == pc.consequent.args[0]
        if pc.antecedent.functor == AssistantProfessor:
            assert pc.antecedent.functor == imp3.antecedent.functor
            assert pc.consequent.functor == imp3.consequent.functor
        elif pc.antecedent.functor == AssociateProfessor:
            assert pc.antecedent.functor == imp4.antecedent.functor
            assert pc.consequent.functor == imp4.consequent.functor
        else:
            assert False

    Labels = constraints['rdf-schema:label']
    assert len(Labels) == 5 and isinstance(Labels, set)
    for l in Labels:
        assert isinstance(l, RightImplication)
        assert len(l.antecedent.args) == 1
        assert len(l.consequent.args) == 2
        assert l.antecedent.args[0] == l.consequent.args[0]
        assert isinstance(l.consequent.args[1], Constant)
        assert l.consequent.functor == label
        if l.antecedent.functor == AdministrativeStaff:
            assert l.consequent.args[1] == imp_label1.consequent.args[1]
        elif l.antecedent.functor == AssistantProfessor:
            assert l.consequent.args[1] == imp_label2.consequent.args[1]
        elif l.antecedent.functor == AssociateProfessor:
            assert l.consequent.args[1] == imp_label3.consequent.args[1]
        elif l.antecedent.functor == Book:
            assert l.consequent.args[1] == imp_label4.consequent.args[1]
        elif l.antecedent.functor == Article:
            assert l.consequent.args[1] == imp_label5.consequent.args[1]
        else:
            assert False

def test_2():
    owl = '''<?xml version="1.0"?>
    <rdf:RDF xmlns="http://www.w3.org/2002/07/owl#"
        xmlns:owl="http://www.w3.org/2002/07/owl#"
        xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
        xmlns:rdfs="http://www.w3.org/2000/01/rdf-schema#">
        <Ontology>
            <versionInfo>0.3.1</versionInfo>
        </Ontology>

        <owl:Class rdf:ID="ClericalStaff">
            <rdfs:label>clerical staff worker</rdfs:label>
            <rdfs:subClassOf rdf:resource="#AdministrativeStaff"/>
        </owl:Class>

        <owl:Class rdf:ID="College">
            <rdfs:label>school</rdfs:label>
            <rdfs:subClassOf rdf:resource="#Organization"/>
        </owl:Class>

        <owl:Class rdf:ID="ConferencePaper">
            <rdfs:label>conference paper</rdfs:label>
            <rdfs:subClassOf rdf:resource="#Article"/>
        </owl:Class>

        <owl:Class rdf:ID="Course">
            <rdfs:label>teaching course</rdfs:label>
            <rdfs:subClassOf rdf:resource="#Work"/>
        </owl:Class>
    </rdf:RDF>'''

    x = Symbol('x')
    ClericalStaff = Symbol('ClericalStaff')
    AdministrativeStaff = Symbol('AdministrativeStaff')
    College = Symbol('College')
    Organization = Symbol('Organization')
    ConferencePaper = Symbol('ConferencePaper')
    Article = Symbol('Article')
    Course = Symbol('Course')
    Work = Symbol('Work')
    label = Symbol('rdf-schema:label')

    imp1 = RightImplication(ClericalStaff(x), AdministrativeStaff(x))
    imp2 = RightImplication(College(x), Organization(x))
    imp3 = RightImplication(ConferencePaper(x), Article(x))
    imp4 = RightImplication(Course(x), Work(x))

    imp_label1 = RightImplication(ClericalStaff(x), label(x, Constant('clerical staff worker')))
    imp_label2 = RightImplication(College(x), label(x, Constant('school')),)
    imp_label3 = RightImplication(ConferencePaper(x), label(x, Constant('conference paper')))
    imp_label4 = RightImplication(Course(x), label(x, Constant('teaching course')))

    onto = OntologyParser(io.StringIO(owl))
    constraints, _ = onto.parse_ontology()

    assert set(constraints.keys()) == set(['AdministrativeStaff', 'Organization', 'Article', 'Work', 'rdf-schema:label'])

    AdminConstraint = constraints['AdministrativeStaff']
    assert len(AdminConstraint) == 1 and isinstance(AdminConstraint, set)
    AdminConstraint = next(iter(AdminConstraint))
    assert isinstance(AdminConstraint, RightImplication)
    assert len(AdminConstraint.antecedent.args) == 1
    assert len(AdminConstraint.consequent.args) == 1
    assert AdminConstraint.antecedent.args[0] == AdminConstraint.consequent.args[0]
    assert AdminConstraint.antecedent.functor == imp1.antecedent.functor
    assert AdminConstraint.consequent.functor == imp1.consequent.functor

    OrganizationConstraint = constraints['Organization']
    assert len(OrganizationConstraint) == 1 and isinstance(OrganizationConstraint, set)
    OrganizationConstraint = next(iter(OrganizationConstraint))
    assert isinstance(OrganizationConstraint, RightImplication)
    assert len(OrganizationConstraint.antecedent.args) == 1
    assert len(OrganizationConstraint.consequent.args) == 1
    assert OrganizationConstraint.antecedent.args[0] == OrganizationConstraint.consequent.args[0]
    assert OrganizationConstraint.antecedent.functor == imp2.antecedent.functor
    assert OrganizationConstraint.consequent.functor == imp2.consequent.functor

    ArticleConstraint = constraints['Article']
    assert len(ArticleConstraint) == 1 and isinstance(ArticleConstraint, set)
    ArticleConstraint = next(iter(ArticleConstraint))
    assert isinstance(ArticleConstraint, RightImplication)
    assert len(ArticleConstraint.antecedent.args) == 1
    assert len(ArticleConstraint.consequent.args) == 1
    assert ArticleConstraint.antecedent.args[0] == ArticleConstraint.consequent.args[0]
    assert ArticleConstraint.antecedent.functor == imp3.antecedent.functor
    assert ArticleConstraint.consequent.functor == imp3.consequent.functor

    WorkConstraint = constraints['Work']
    assert len(WorkConstraint) == 1 and isinstance(WorkConstraint, set)
    WorkConstraint = next(iter(WorkConstraint))
    assert isinstance(WorkConstraint, RightImplication)
    assert len(WorkConstraint.antecedent.args) == 1
    assert len(WorkConstraint.consequent.args) == 1
    assert WorkConstraint.antecedent.args[0] == WorkConstraint.consequent.args[0]
    assert WorkConstraint.antecedent.functor == imp4.antecedent.functor
    assert WorkConstraint.consequent.functor == imp4.consequent.functor


    Labels = constraints['rdf-schema:label']
    assert len(Labels) == 4 and isinstance(Labels, set)
    for l in Labels:
        assert isinstance(l, RightImplication)
        assert len(l.antecedent.args) == 1
        assert len(l.consequent.args) == 2
        assert l.antecedent.args[0] == l.consequent.args[0]
        assert isinstance(l.consequent.args[1], Constant)
        assert l.consequent.functor == label
        if l.antecedent.functor == ClericalStaff:
            assert l.consequent.args[1] == imp_label1.consequent.args[1]
        elif l.antecedent.functor == College:
            assert l.consequent.args[1] == imp_label2.consequent.args[1]
        elif l.antecedent.functor == ConferencePaper:
            assert l.consequent.args[1] == imp_label3.consequent.args[1]
        elif l.antecedent.functor == Course:
            assert l.consequent.args[1] == imp_label4.consequent.args[1]
        else:
            assert False


def test_3():
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

    Professor = Symbol('Professor')
    Chair = Symbol('Chair')
    headOf = Symbol('headOf')
    Department = Symbol('Department')
    Person = Symbol('Person')

    label = Symbol('rdf-schema:label')

    onto = OntologyParser(io.StringIO(owl))
    constraints, _ = onto.parse_ontology()

    for c in constraints:
        if c.startswith('fresh'):
            support_rule = next(iter(constraints[c]))
            supportChair = support_rule.consequent.functor
            x = support_rule.consequent.args[0]
            y = support_rule.consequent.args[1]
            break

    assert set(constraints.keys()) == set(['Person', 'headOf', 'Department', 'Professor', 'rdf-schema:label', c])

    imp1 = RightImplication(supportChair(x, y), Person(x))
    imp2 = RightImplication(Chair(x), supportChair(x, y))
    imp3 = RightImplication(supportChair(x, y), headOf(x, y))
    imp4 = RightImplication(supportChair(x, y), Department(y))
    imp5 = RightImplication(Chair(x), Professor(x),)

    imp_label = RightImplication(Chair(x), label(x, Constant('chair')))

    PersonConstraint = constraints['Person']
    assert len(PersonConstraint) == 1 and isinstance(PersonConstraint, set)
    PersonConstraint = next(iter(PersonConstraint))
    assert isinstance(PersonConstraint, RightImplication)
    assert len(PersonConstraint.antecedent.args) == 2
    assert len(PersonConstraint.consequent.args) == 1
    assert PersonConstraint.antecedent.args[0] == PersonConstraint.consequent.args[0]
    assert PersonConstraint.antecedent.functor == imp1.antecedent.functor
    assert PersonConstraint.consequent.functor == imp1.consequent.functor

    SupportConstraint = constraints[c]
    assert len(SupportConstraint) == 1 and isinstance(SupportConstraint, set)
    SupportConstraint = next(iter(SupportConstraint))
    assert isinstance(SupportConstraint, RightImplication)
    assert len(SupportConstraint.antecedent.args) == 1
    assert len(SupportConstraint.consequent.args) == 2
    assert SupportConstraint.antecedent.args[0] == SupportConstraint.consequent.args[0]
    assert SupportConstraint.antecedent.functor == imp2.antecedent.functor
    assert SupportConstraint.consequent.functor == imp2.consequent.functor

    headOfConstraint = constraints['headOf']
    assert len(headOfConstraint) == 1 and isinstance(headOfConstraint, set)
    headOfConstraint = next(iter(headOfConstraint))
    assert isinstance(headOfConstraint, RightImplication)
    assert len(headOfConstraint.antecedent.args) == 2
    assert len(headOfConstraint.consequent.args) == 2
    assert headOfConstraint.antecedent.args == headOfConstraint.consequent.args
    assert headOfConstraint.antecedent.functor == imp3.antecedent.functor
    assert headOfConstraint.consequent.functor == imp3.consequent.functor

    DepartmentConstraint = constraints['Department']
    assert len(DepartmentConstraint) == 1 and isinstance(DepartmentConstraint, set)
    DepartmentConstraint = next(iter(DepartmentConstraint))
    assert isinstance(DepartmentConstraint, RightImplication)
    assert len(DepartmentConstraint.antecedent.args) == 2
    assert len(DepartmentConstraint.consequent.args) == 1
    assert DepartmentConstraint.antecedent.args[1] == DepartmentConstraint.consequent.args[0]
    assert DepartmentConstraint.antecedent.functor == imp4.antecedent.functor
    assert DepartmentConstraint.consequent.functor == imp4.consequent.functor

    ProfessorConstraint = constraints['Professor']
    assert len(ProfessorConstraint) == 1 and isinstance(ProfessorConstraint, set)
    ProfessorConstraint = next(iter(ProfessorConstraint))
    assert isinstance(ProfessorConstraint, RightImplication)
    assert len(ProfessorConstraint.antecedent.args) == 1
    assert len(ProfessorConstraint.consequent.args) == 1
    assert ProfessorConstraint.antecedent.args[0] == ProfessorConstraint.consequent.args[0]
    assert ProfessorConstraint.antecedent.functor == imp5.antecedent.functor
    assert ProfessorConstraint.consequent.functor == imp5.consequent.functor

    Labels = constraints['rdf-schema:label']
    assert len(Labels) == 1 and isinstance(Labels, set)
    l = Labels.pop()
    assert isinstance(l, RightImplication)
    assert len(l.antecedent.args) == 1
    assert len(l.consequent.args) == 2
    assert l.antecedent.args[0] == l.consequent.args[0]
    assert isinstance(l.consequent.args[1], Constant)
    assert l.consequent.functor == label
    if l.antecedent.functor == Chair:
        assert l.consequent.args[1] == imp_label.consequent.args[1]
    else:
        assert False



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
        # TODO Not working with solve_all()
        #f_term = nl.solve_all()

    #res = f_term['answer'].as_pandas_dataframe().values
    res = f_term.as_pandas_dataframe().values
    assert (res == [['Chair']]).all()

