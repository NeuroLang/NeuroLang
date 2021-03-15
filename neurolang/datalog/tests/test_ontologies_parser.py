
import io

from ...expressions import Symbol
from ..constraints_representation import RightImplication
from ..expressions import Implication
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
    Employee = Symbol('employee')
    AdministrativeStaff = Symbol('administrativestaff')
    Publication = Symbol('publication')
    Article = Symbol('article')
    Professor = Symbol('professor')
    AssistantProfessor = Symbol('assistantprofessor')
    AssociateProfessor = Symbol('associateprofessor')
    Book = Symbol('book')

    imp1 = Implication(Employee(x), AdministrativeStaff(x))
    imp2 = Implication(Publication(x), Article(x))
    imp3 = Implication(Professor(x), AssistantProfessor(x))
    imp4 = Implication(Professor(x), AssociateProfessor(x))
    imp5 = Implication(Publication(x), Book(x))


    AdministrativeStaff_label = Symbol('administrative_staff_worker')
    AssistantProfessor_label = Symbol('assistant_professor')
    AssociateProfessor_label = Symbol('associate_professor')
    
    imp_label1 = Implication(AdministrativeStaff(x), AdministrativeStaff_label(x))
    imp_label2 = Implication(AssistantProfessor(x), AssistantProfessor_label(x))
    imp_label3 = Implication(AssociateProfessor(x), AssociateProfessor_label(x))

    onto = OntologyParser(io.StringIO(owl))
    rules, constraints = onto.parse_ontology()

    
    assert len(constraints) == 0
    assert len(rules) == 8

    assert isinstance(rules[0], Implication)
    assert len(rules[0].antecedent.args) == 1
    assert len(rules[0].consequent.args) == 1
    assert rules[0].antecedent.args[0] == rules[0].consequent.args[0]
    assert rules[0].antecedent.functor == imp_label1.antecedent.functor
    assert rules[0].consequent.functor == imp_label1.consequent.functor

    assert isinstance(rules[1], Implication)
    assert len(rules[1].antecedent.args) == 1
    assert len(rules[1].consequent.args) == 1
    assert rules[1].antecedent.args[0] == rules[1].consequent.args[0]
    assert rules[1].antecedent.functor == imp1.antecedent.functor
    assert rules[1].consequent.functor == imp1.consequent.functor

    assert isinstance(rules[2], Implication)
    assert len(rules[2].antecedent.args) == 1
    assert len(rules[2].consequent.args) == 1
    assert rules[2].antecedent.args[0] == rules[2].consequent.args[0]
    assert rules[2].antecedent.functor == imp2.antecedent.functor
    assert rules[2].consequent.functor == imp2.consequent.functor

    assert isinstance(rules[3], Implication)
    assert len(rules[3].antecedent.args) == 1
    assert len(rules[3].consequent.args) == 1
    assert rules[3].antecedent.args[0] == rules[3].consequent.args[0]
    assert rules[3].antecedent.functor == imp_label2.antecedent.functor
    assert rules[3].consequent.functor == imp_label2.consequent.functor

    assert isinstance(rules[4], Implication)
    assert len(rules[4].antecedent.args) == 1
    assert len(rules[4].consequent.args) == 1
    assert rules[4].antecedent.args[0] == rules[4].consequent.args[0]
    assert rules[4].antecedent.functor == imp3.antecedent.functor
    assert rules[4].consequent.functor == imp3.consequent.functor
    
    assert isinstance(rules[5], Implication)
    assert len(rules[5].antecedent.args) == 1
    assert len(rules[5].consequent.args) == 1
    assert rules[5].antecedent.args[0] == rules[5].consequent.args[0]
    assert rules[5].antecedent.functor == imp_label3.antecedent.functor
    assert rules[5].consequent.functor == imp_label3.consequent.functor

    assert isinstance(rules[6], Implication)
    assert len(rules[6].antecedent.args) == 1
    assert len(rules[6].consequent.args) == 1
    assert rules[6].antecedent.args[0] == rules[6].consequent.args[0]
    assert rules[6].antecedent.functor == imp4.antecedent.functor
    assert rules[6].consequent.functor == imp4.consequent.functor

    assert isinstance(rules[7], Implication)
    assert len(rules[7].antecedent.args) == 1
    assert len(rules[7].consequent.args) == 1
    assert rules[7].antecedent.args[0] == rules[7].consequent.args[0]
    assert rules[7].antecedent.functor == imp5.antecedent.functor
    assert rules[7].consequent.functor == imp5.consequent.functor


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
    ClericalStaff = Symbol('clericalstaff')
    AdministrativeStaff = Symbol('administrativestaff')
    College = Symbol('college')
    Organization = Symbol('organization')
    ConferencePaper = Symbol('conferencepaper')
    Article = Symbol('article')
    Course = Symbol('course')
    Work = Symbol('work')

    imp1 = Implication(AdministrativeStaff(x), ClericalStaff(x))
    imp2 = Implication(Organization(x), College(x))
    imp3 = Implication(Article(x), ConferencePaper(x))
    imp4 = Implication(Work(x), Course(x))


    ClericalStaff_label = Symbol('clerical_staff_worker')
    College_label = Symbol('school')
    ConferencePaper_label = Symbol('conference_paper')
    Course_label = Symbol('teaching_course')

    
    imp_label1 = Implication(ClericalStaff(x), ClericalStaff_label(x))
    imp_label2 = Implication(College(x), College_label(x))
    imp_label3 = Implication(ConferencePaper(x), ConferencePaper_label(x))
    imp_label4 = Implication(Course(x), Course_label(x))

    onto = OntologyParser(io.StringIO(owl))
    rules, constraints = onto.parse_ontology()

    
    assert len(constraints) == 0
    assert len(rules) == 8

    assert isinstance(rules[0], Implication)
    assert len(rules[0].antecedent.args) == 1
    assert len(rules[0].consequent.args) == 1
    assert rules[0].antecedent.args[0] == rules[0].consequent.args[0]
    assert rules[0].antecedent.functor == imp_label1.antecedent.functor
    assert rules[0].consequent.functor == imp_label1.consequent.functor

    assert isinstance(rules[1], Implication)
    assert len(rules[1].antecedent.args) == 1
    assert len(rules[1].consequent.args) == 1
    assert rules[1].antecedent.args[0] == rules[1].consequent.args[0]
    assert rules[1].antecedent.functor == imp1.antecedent.functor
    assert rules[1].consequent.functor == imp1.consequent.functor

    assert isinstance(rules[2], Implication)
    assert len(rules[2].antecedent.args) == 1
    assert len(rules[2].consequent.args) == 1
    assert rules[2].antecedent.args[0] == rules[2].consequent.args[0]
    assert rules[2].antecedent.functor == imp_label2.antecedent.functor
    assert rules[2].consequent.functor == imp_label2.consequent.functor

    assert isinstance(rules[3], Implication)
    assert len(rules[3].antecedent.args) == 1
    assert len(rules[3].consequent.args) == 1
    assert rules[3].antecedent.args[0] == rules[3].consequent.args[0]
    assert rules[3].antecedent.functor == imp2.antecedent.functor
    assert rules[3].consequent.functor == imp2.consequent.functor

    assert isinstance(rules[4], Implication)
    assert len(rules[4].antecedent.args) == 1
    assert len(rules[4].consequent.args) == 1
    assert rules[4].antecedent.args[0] == rules[4].consequent.args[0]
    assert rules[4].antecedent.functor == imp_label3.antecedent.functor
    assert rules[4].consequent.functor == imp_label3.consequent.functor
    
    assert isinstance(rules[5], Implication)
    assert len(rules[5].antecedent.args) == 1
    assert len(rules[5].consequent.args) == 1
    assert rules[5].antecedent.args[0] == rules[5].consequent.args[0]
    assert rules[5].antecedent.functor == imp3.antecedent.functor
    assert rules[5].consequent.functor == imp3.consequent.functor

    assert isinstance(rules[6], Implication)
    assert len(rules[6].antecedent.args) == 1
    assert len(rules[6].consequent.args) == 1
    assert rules[6].antecedent.args[0] == rules[6].consequent.args[0]
    assert rules[6].antecedent.functor == imp_label4.antecedent.functor
    assert rules[6].consequent.functor == imp_label4.consequent.functor

    assert isinstance(rules[7], Implication)
    assert len(rules[7].antecedent.args) == 1
    assert len(rules[7].consequent.args) == 1
    assert rules[7].antecedent.args[0] == rules[7].consequent.args[0]
    assert rules[7].antecedent.functor == imp4.antecedent.functor
    assert rules[7].consequent.functor == imp4.consequent.functor



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

    '''
    Professor(X) :- Chair(X).
    #exists{Y}supportChair(X,Y) :- Chair(X).
    headOf(X,Y) :- supportChair(X,Y).
    Department(Y) :- supportChair(X,Y).
    Person(X) :- supportChair(X,Y). 
    '''


    x = Symbol('x')
    y = Symbol('y')
    Professor = Symbol('professor')
    Chair = Symbol('chair')
    supportChair = Symbol.fresh()
    headOf = Symbol('headof')
    Department = Symbol('department')
    Person = Symbol('person')

    imp1 = Implication(Person(x), supportChair(x, y))
    imp2 = RightImplication(Chair(x), supportChair(x, y))
    imp3 = Implication(headOf(x, y), supportChair(x, y))
    imp4 = Implication(Department(y), supportChair(x, y))
    imp5 = Implication(Professor(x), Chair(x))

    onto = OntologyParser(io.StringIO(owl))
    rules, constraints = onto.parse_ontology()

    
    assert len(constraints) == 1
    assert len(rules) == 4

    assert isinstance(rules[0], Implication)
    assert len(rules[0].antecedent.args) == 2
    assert len(rules[0].consequent.args) == 1
    assert rules[0].antecedent.args[0] == rules[0].consequent.args[0]
    assert rules[0].consequent.functor == imp1.consequent.functor

    assert isinstance(constraints[0], RightImplication)
    assert len(constraints[0].antecedent.args) == 1
    assert len(constraints[0].consequent.args) == 2
    assert constraints[0].antecedent.args[0] == constraints[0].consequent.args[0]
    assert constraints[0].antecedent.functor == imp2.antecedent.functor

    assert isinstance(rules[1], Implication)
    assert len(rules[1].antecedent.args) == 2
    assert len(rules[1].consequent.args) == 1
    assert rules[1].antecedent.args[1] == rules[1].consequent.args[0]
    assert rules[1].consequent.functor == imp4.consequent.functor

    assert isinstance(rules[2], Implication)
    assert len(rules[2].antecedent.args) == 2
    assert len(rules[2].consequent.args) == 2
    assert rules[2].antecedent.args == rules[2].consequent.args
    assert rules[2].consequent.functor == imp3.consequent.functor
    
    assert isinstance(rules[3], Implication)
    assert len(rules[3].antecedent.args) == 1
    assert len(rules[3].consequent.args) == 1
    assert rules[3].antecedent.args[0] == rules[3].consequent.args[0]
    assert rules[3].antecedent.functor == imp5.antecedent.functor
    assert rules[3].consequent.functor == imp5.consequent.functor


def test_4():
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
    </rdf:RDF>'''

    '''
    Professor(X) :- Dean(X).
    #exists{Y}supportDean(X,Y) :- Dean(X).
    headOf(X,Y) :- supportDean(X,Y).
    College(Y) :- supportDean(X,Y).
    Person(X) :- supportDean(X,Y).
    '''

    x = Symbol('x')
    y = Symbol('y')
    Professor = Symbol('professor')
    Dean = Symbol('dean')
    supportDean = Symbol.fresh()
    headOf = Symbol('headof')
    College = Symbol('college')
    Person = Symbol('person')

    imp1 = Implication(Person(x), supportDean(x, y))
    imp2 = RightImplication(Dean(x), supportDean(x, y))
    imp3 = Implication(headOf(x, y), supportDean(x, y))
    imp4 = Implication(College(y), supportDean(x, y))
    imp5 = Implication(Professor(x), Dean(x))

    onto = OntologyParser(io.StringIO(owl))
    rules, constraints = onto.parse_ontology()

    
    assert len(constraints) == 1
    assert len(rules) == 4

    assert isinstance(rules[0], Implication)
    assert len(rules[0].antecedent.args) == 2
    assert len(rules[0].consequent.args) == 1
    assert rules[0].antecedent.args[0] == rules[0].consequent.args[0]
    assert rules[0].consequent.functor == imp1.consequent.functor

    assert isinstance(constraints[0], RightImplication)
    assert len(constraints[0].antecedent.args) == 1
    assert len(constraints[0].consequent.args) == 2
    assert constraints[0].antecedent.args[0] == constraints[0].consequent.args[0]
    assert constraints[0].antecedent.functor == imp2.antecedent.functor

    assert isinstance(rules[1], Implication)
    assert len(rules[1].antecedent.args) == 2
    assert len(rules[1].consequent.args) == 1
    assert rules[1].antecedent.args[1] == rules[1].consequent.args[0]
    assert rules[1].consequent.functor == imp4.consequent.functor

    assert isinstance(rules[2], Implication)
    assert len(rules[2].antecedent.args) == 2
    assert len(rules[2].consequent.args) == 2
    assert rules[2].antecedent.args == rules[2].consequent.args
    assert rules[2].consequent.functor == imp3.consequent.functor
    
    assert isinstance(rules[3], Implication)
    assert len(rules[3].antecedent.args) == 1
    assert len(rules[3].consequent.args) == 1
    assert rules[3].antecedent.args[0] == rules[3].consequent.args[0]
    assert rules[3].antecedent.functor == imp5.antecedent.functor
    assert rules[3].consequent.functor == imp5.consequent.functor

def test_all_values():
    owl = '''<?xml version="1.0"?>
    <rdf:RDF xmlns="http://www.w3.org/2002/07/owl#"
     xml:base="http://www.w3.org/2002/07/owl"
     xmlns:dc="http://purl.org/dc/elements/1.1/"
     xmlns:ro="http://www.obofoundry.org/ro/ro.owl#"
     xmlns:obo="http://data.bioontology.org/metadata/obo/"
     xmlns:owl="http://www.w3.org/2002/07/owl#"
     xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
     xmlns:xml="http://www.w3.org/XML/1998/namespace"
     xmlns:xsd="http://www.w3.org/2001/XMLSchema#"
     xmlns:obo1="http://purl.obolibrary.org/obo/"
     xmlns:rdfs="http://www.w3.org/2000/01/rdf-schema#"
     xmlns:skos="http://www.w3.org/2004/02/skos/core#"
     xmlns:terms="http://purl.org/dc/terms/"
     xmlns:metadata="http://data.bioontology.org/metadata/">
        <Ontology>
            <versionInfo>0.3.1</versionInfo>
        </Ontology>

        <Class rdf:about="http://www.cognitiveatlas.org/ontology/cogat.owl#CAO_00539">
            <rdfs:subClassOf rdf:resource="http://www.cognitiveatlas.org/ontology/cogat.owl#CAO_00380"/>
            <rdfs:subClassOf>
                <Restriction>
                    <onProperty rdf:resource="http://www.cognitiveatlas.org/ontology/cogat.owl#measured_by"/>
                    <someValuesFrom rdf:resource="http://www.cognitiveatlas.org/ontology/cogat.owl#CAO_02986"/>
                </Restriction>
            </rdfs:subClassOf>
            <rdfs:subClassOf>
                <Restriction>
                    <onProperty rdf:resource="http://www.cognitiveatlas.org/ontology/cogat.owl#measured_by"/>
                    <someValuesFrom rdf:resource="http://www.cognitiveatlas.org/ontology/cogat.owl#CAO_02987"/>
                </Restriction>
            </rdfs:subClassOf>
            <rdfs:subClassOf>
                <Restriction>
                    <onProperty rdf:resource="http://www.cognitiveatlas.org/ontology/cogat.owl#measured_by"/>
                    <allValuesFrom>
                        <Class>
                            <unionOf rdf:parseType="Collection">
                                <rdf:Description rdf:about="http://www.cognitiveatlas.org/ontology/cogat.owl#CAO_02986"/>
                                <rdf:Description rdf:about="http://www.cognitiveatlas.org/ontology/cogat.owl#CAO_02987"/>
                            </unionOf>
                        </Class>
                    </allValuesFrom>
                </Restriction>
            </rdfs:subClassOf>
            <metadata:prefixIRI rdf:datatype="http://www.w3.org/2001/XMLSchema#string">cogat:CAO_00539</metadata:prefixIRI>
            <terms:Contributor>Brenda Gregory</terms:Contributor>
            <terms:Date>2011-05-18 21:24:27</terms:Date>
            <terms:Title>Verbal memory</terms:Title>
            <terms:identifier>trm_4a3fd79d0b457</terms:identifier>
            <rdfs:label>Verbal memory</rdfs:label>
            <skos:altLabel></skos:altLabel>
            <skos:definition>Recall based on spoken words.</skos:definition>
            <skos:hasTopConcept>Learning and Memory</skos:hasTopConcept>
            <skos:prefLabel>Verbal memory</skos:prefLabel>
        </Class>
    </rdf:RDF>
    '''

    onto = OntologyParser(io.StringIO(owl))
    res = onto.parse_ontology()

    a = 1