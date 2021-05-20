
import io

from ...frontend import NeurolangPDL
from ...expressions import Constant, Symbol
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
    Employee = Symbol('Employee')
    AdministrativeStaff = Symbol('AdministrativeStaff')
    Publication = Symbol('Publication')
    Article = Symbol('Article')
    Professor = Symbol('Professor')
    AssistantProfessor = Symbol('AssistantProfessor')
    AssociateProfessor = Symbol('AssociateProfessor')
    Book = Symbol('Book')
    
    label = Symbol('label')

    imp1 = Implication(Employee(x), AdministrativeStaff(x))
    imp2 = Implication(Publication(x), Article(x))
    imp3 = Implication(Professor(x), AssistantProfessor(x))
    imp4 = Implication(Professor(x), AssociateProfessor(x))
    imp5 = Implication(Publication(x), Book(x))

    
    imp_label1 = Implication(AdministrativeStaff(x), label(x, Constant('administrative staff worker')))
    imp_label2 = Implication(AssistantProfessor(x), label(x, Constant('assistant professor')))
    imp_label3 = Implication(AssociateProfessor(x), label(x, Constant('associate professor')))
    imp_label4 = Implication(Book(x), label(x, Constant('book')))
    imp_label5 = Implication(Article(x), label(x, Constant('article')))

    onto = OntologyParser(io.StringIO(owl))
    rules = onto.parse_ontology()

    assert len(rules) == 10

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
    ClericalStaff = Symbol('ClericalStaff')
    AdministrativeStaff = Symbol('AdministrativeStaff')
    College = Symbol('College')
    Organization = Symbol('Organization')
    ConferencePaper = Symbol('ConferencePaper')
    Article = Symbol('Article')
    Course = Symbol('Course')
    Work = Symbol('Work')
    label = Symbol('ĺabel')

    imp1 = Implication(AdministrativeStaff(x), ClericalStaff(x))
    imp2 = Implication(Organization(x), College(x))
    imp3 = Implication(Article(x), ConferencePaper(x))
    imp4 = Implication(Work(x), Course(x))


    ClericalStaff_label = Constant('clerical staff worker')
    College_label = Constant('school')
    ConferencePaper_label = Constant('conference paper')
    Course_label = Constant('teaching course')

    
    imp_label1 = Implication(ClericalStaff(x), label(x, ClericalStaff_label))
    imp_label2 = Implication(College(x), label(x, College_label))
    imp_label3 = Implication(ConferencePaper(x), label(x, ConferencePaper_label))
    imp_label4 = Implication(Course(x), label(x, Course_label))

    onto = OntologyParser(io.StringIO(owl))
    rules = onto.parse_ontology()

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



    Professor = Symbol('Professor')
    Chair = Symbol('Chair')
    headOf = Symbol('headOf')
    Department = Symbol('Department')
    Person = Symbol('Person')

    onto = OntologyParser(io.StringIO(owl))
    constraints = onto.parse_ontology()

    
    assert len(constraints) == 5

    for c in constraints:
        if c.antecedent.functor.is_fresh:
            supportChair = c.antecedent.functor
            x = c.antecedent.args[0]
            y = c.antecedent.args[1]
            break

    imp1 = RightImplication(supportChair(x, y), Person(x))
    imp2 = RightImplication(Chair(x), supportChair(x, y))
    imp3 = RightImplication(supportChair(x, y), headOf(x, y))
    imp4 = RightImplication(supportChair(x, y), Department(y))
    imp5 = RightImplication(Professor(x), Chair(x))

    index = constraints.index(imp4)
    # RightImplication(supportChair(x, y), Department(y))
    assert isinstance(constraints[index], RightImplication)
    assert len(constraints[index].antecedent.args) == 2
    assert len(constraints[index].consequent.args) == 1
    assert constraints[index].antecedent.args[1] == constraints[index].consequent.args[0]
    assert constraints[index].consequent.functor == imp4.consequent.functor

    index = constraints.index(imp3)
    # RightImplication(supportChair(x, y), headOf(x, y))
    assert isinstance(constraints[index], RightImplication)
    assert len(constraints[index].antecedent.args) == 2
    assert len(constraints[index].consequent.args) == 2
    assert constraints[index].antecedent.args == constraints[index].consequent.args
    assert constraints[index].consequent.functor == imp3.consequent.functor

    for n, c in enumerate(constraints):
        if (
            c.antecedent.functor == Professor and 
            c.consequent.functor == Chair
        ):
            index = n
            imp = c
            break

    # RightImplication(Professor(x), Chair(x))
    assert isinstance(constraints[index], RightImplication)
    assert len(constraints[index].antecedent.args) == 1
    assert len(constraints[index].consequent.args) == 1
    assert constraints[index].antecedent.args == constraints[index].consequent.args
    assert constraints[index].consequent.functor == imp.consequent.functor
    
    index = constraints.index(imp2)
    # RightImplication(Chair(x), supportChair(x, y))
    assert isinstance(constraints[index], RightImplication)
    assert len(constraints[index].antecedent.args) == 1
    assert len(constraints[index].consequent.args) == 2
    assert constraints[index].antecedent.args[0] == constraints[index].consequent.args[0]
    assert constraints[index].antecedent.functor == imp2.antecedent.functor

    for n, c in enumerate(constraints):
        if (
            c.antecedent.functor.is_fresh and 
            c.consequent.functor == Person
        ):
            index = n
            imp = c
            break
    # RightImplication(supportChair(x, y), Person(x))
    assert isinstance(constraints[index], RightImplication)
    assert len(constraints[index].antecedent.args) == 2
    assert len(constraints[index].consequent.args) == 1
    assert constraints[index].antecedent.args[0] == constraints[index].consequent.args[0]
    assert constraints[index].consequent.functor == imp.consequent.functor


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

    Professor = Symbol('Professor')
    Dean = Symbol('Dean')
    headOf = Symbol('headOf')
    College = Symbol('College')
    Person = Symbol('Person')

    onto = OntologyParser(io.StringIO(owl))
    constraints = onto.parse_ontology()

    assert len(constraints) == 5
    
    for c in constraints:
        if c.antecedent.functor.is_fresh:
            supportDean = c.antecedent.functor
            x = c.antecedent.args[0]
            y = c.antecedent.args[1]
            break

    imp1 = RightImplication(supportDean(x, y), Person(x))
    imp2 = RightImplication(Dean(x), supportDean(x, y))
    imp3 = RightImplication(supportDean(x, y), headOf(x, y))
    imp4 = RightImplication(supportDean(x, y), College(y))
    imp5 = Implication(Professor(x), Dean(x))

    index = constraints.index(imp4)
    assert isinstance(constraints[index], RightImplication)
    assert len(constraints[index].antecedent.args) == 2
    assert len(constraints[index].consequent.args) == 1
    assert constraints[index].antecedent.args[1] == constraints[index].consequent.args[0]
    assert constraints[index].consequent.functor == imp4.consequent.functor

    index = constraints.index(imp3)
    assert isinstance(constraints[index], RightImplication)
    assert len(constraints[index].antecedent.args) == 2
    assert len(constraints[index].consequent.args) == 2
    assert constraints[index].antecedent.args == constraints[index].consequent.args
    assert constraints[index].consequent.functor == imp3.consequent.functor

    for n, c in enumerate(constraints):
        if (
            c.antecedent.functor == Professor and 
            c.consequent.functor == Dean
        ):
            index = n
            imp = c
            break

    assert isinstance(constraints[index], RightImplication)
    assert len(constraints[index].antecedent.args) == 1
    assert len(constraints[index].consequent.args) == 1
    assert constraints[index].antecedent.args == constraints[index].consequent.args
    assert constraints[index].consequent.functor == imp.consequent.functor
    
    index = constraints.index(imp2)
    assert isinstance(constraints[index], RightImplication)
    assert len(constraints[index].antecedent.args) == 1
    assert len(constraints[index].consequent.args) == 2
    assert constraints[index].antecedent.args[0] == constraints[index].consequent.args[0]
    assert constraints[index].antecedent.functor == imp2.antecedent.functor

    for n, c in enumerate(constraints):
        if (
            c.antecedent.functor.is_fresh and 
            c.consequent.functor == Person
        ):
            index = n
            imp = c
            break
    assert isinstance(constraints[index], RightImplication)
    assert len(constraints[index].antecedent.args) == 2
    assert len(constraints[index].consequent.args) == 1
    assert constraints[index].antecedent.args[0] == constraints[index].consequent.args[0]
    assert constraints[index].consequent.functor == imp.consequent.functor



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


def test_iobc():
    from nilearn import datasets, image
    import nibabel
    import numpy as np
    import pandas as pd
    from neurolang.frontend import NeurolangPDL, ExplicitVBR, ExplicitVBROverlay
    from typing import Callable, Iterable

    iobc = datasets.utils._fetch_files(
        datasets.utils._get_dataset_dir('ontology'),
        [
            (
                'iobc.xrdf',
                'http://data.bioontology.org/ontologies/IOBC/download?'
                'apikey=8b5b7825-538d-40e0-9e9e-5ab9274a9aeb&download_format=rdf',
                {'move': 'iobc.xrdf'}
            )
        ]
    )[0]

    nl = NeurolangPDL()
    nl.load_ontology(iobc)

    mni_mask = image.resample_img(
        nibabel.load(datasets.fetch_icbm152_2009()["gm"]),
        np.eye(3) * 2
    )

    ns_database_fn, ns_features_fn = datasets.utils._fetch_files(
        "neurolang",
        [
            (
                "database.txt",
                "https://github.com/neurosynth/neurosynth-data"
                "/raw/master/current_data.tar.gz",
                {"uncompress": True},
            ),
            (
                "features.txt",
                "https://github.com/neurosynth/neurosynth-data"
                "/raw/master/current_data.tar.gz",
                {"uncompress": True},
            ),
        ],
    )

    ns_database = pd.read_csv(ns_database_fn, sep="\t")
    ns_database = ns_database[["x", "y", "z", "id"]]

    ns_features = pd.read_csv(ns_features_fn, sep="\t")
    ns_docs = ns_features[["pmid"]].drop_duplicates()
    ns_terms = pd.melt(
        ns_features, var_name="term", id_vars="pmid", value_name="TfIdf"
    ).query("TfIdf > 1e-3")[["term", "pmid"]]


    terms_det = nl.add_tuple_set(
            ns_terms.term.unique(), name='terms_det'
    )

    label = nl.new_symbol(name='rdf-schema:label')
    related = nl.new_symbol(name='core:related')
    altLabel = nl.new_symbol(name='core:altLabel')

    @nl.add_symbol
    def word_lower(name: str) -> str:
        return name.lower()

    @nl.add_symbol
    def agg_create_region_overlay_MNI(
        x: Iterable, y: Iterable, z: Iterable, p: Iterable
    ) -> ExplicitVBR:
        voxels = nibabel.affines.apply_affine(
            np.linalg.inv(mni_mask.affine),
            np.c_[x, y, z]
        )
        return ExplicitVBROverlay(
            voxels, mni_mask.affine, p, image_dim=mni_mask.shape
        )

    @nl.add_symbol
    def mean(iterable: Iterable) -> float:
        return np.mean(iterable)


    @nl.add_symbol
    def std(iterable: Iterable) -> float:
        return np.std(iterable)

    with nl.scope as e:
        e.ontology_related[e.ne, e.l] = (
            label(e.e, e.ne) &
            related(e.e, e.r) &
            label(e.r, e.nr) &
            (e.l == word_lower[e.nr])
        )
        
        #e.ontology_synonym[e.ne, e.l] = (
        #    label(e.e, e.ne) &
        #    altLabel(e.e, e.r) &
        #    (e.l == word_lower[e.r])
        #)
        
        e.res[e.entity, e.relation, e.term] = (
            e.ontology_related[e.entity, e.term] &
            e.terms_det[e.entity] &
            e.terms_det[e.term] &
            (e.relation == 'related')
        )
        
        #e.res[e.entity, e.relation, e.term] = (
        #    e.ontology_synonym[e.entity, e.term] &
        #    e.terms_det[e.entity] &
        #    e.terms_det[e.term] &
        #    (e.relation == 'synonym')
        #)
        
        #res = nl.solve_all()
        r = nl.query((e.entity, e.relation, e.term), e.res[e.entity, e.relation, e.term])
    a = 1

def test_cogat():
    from nilearn import datasets, image
    import pandas as pd
    import numpy as np
    import nibabel as nib

    cogAt = datasets.utils._fetch_files(
        datasets.utils._get_dataset_dir('ontology'),
        [
            (
                'cogat_old.xml',
                'https://data.bioontology.org/ontologies/COGAT/submissions/7/download?'
                'apikey=8b5b7825-538d-40e0-9e9e-5ab9274a9aeb',
                {'move': 'cogat_old.xml'}
            )
        ]
    )[0]

    #mni_mask = image.resample_img(
    #    nib.load(datasets.fetch_icbm152_2009()["gm"]),
    #    np.eye(3) * 2
    #)

    #ns_database_fn, ns_features_fn = datasets.utils._fetch_files(
    #    datasets.utils._get_dataset_dir('neurosynth'),
    #    [
    #        (
    #            'database.txt',
    #            'https://github.com/neurosynth/neurosynth-data/raw/master/current_data.tar.gz',
    #            {'uncompress': True}
    #        ),
    #        (
    #            'features.txt',
    #            'https://github.com/neurosynth/neurosynth-data/raw/master/current_data.tar.gz',
    #            {'uncompress': True}
    #        ),
    #    ]
    #)

    #ns_database = pd.read_csv(ns_database_fn, sep="\t")
    #ns_database = ns_database[["x", "y", "z", "id"]]

    #ns_features = pd.read_csv(ns_features_fn, sep="\t")
    #ns_docs = ns_features[["pmid"]].drop_duplicates()
    #ns_terms = pd.melt(
    #    ns_features, var_name="term", id_vars="pmid", value_name="TfIdf"
    #).query("TfIdf > 1e-3")[["term", "pmid"]]

    import rdflib
    from rdflib import RDFS
    #g = rdflib.Graph()
    #g.load(cogAt)

    from rdflib import BNode

    #onto_dic = {}
    #for obj in g.subjects():
    #    if isinstance(obj, BNode):
    #        continue
    #    for b in g.triples((obj, RDFS.label, None)):
    #        label = b[2].lower().replace(' ', '_')
    #        obj_split = obj.split('#')
    #        if len(obj_split) == 2:
    #            name = obj_split[1]
    #            namespace = obj_split[0].split('/')[-1]
    #            if name[0] != '' and namespace != '':
    #                res = namespace + ':' + name
    #            else:
    #                res = name
    #        else:
    #            obj_split = obj.split('/')
    #            res = obj_split[-1]

    #        onto_dic[label] = res

    #group_terms = ns_terms.groupby('term')
    #dic_term_pmid = {}

    #for term, ids in group_terms:
    #    term = term.lower().replace(' ', '_')
    #    dic_term_pmid[term] = ids

    #merge_dic = {}
    #for k, v in onto_dic.items():
    #    if k in dic_term_pmid.keys():
    #        vl = v.lower()
    #        merge_dic[vl] = dic_term_pmid[k]

    nl = NeurolangPDL()
    nl.load_ontology(cogAt)

    #for k, v in dic_term_pmid.items():
    #    if k in onto_dic.keys():
    #        cogat_key = onto_dic[k]
    #        nl.add_tuple_set(tuple(v.pmid.values), name=cogat_key)

    #SelectedStudy = nl.add_uniform_probabilistic_choice_over_set(
    #    ns_docs, name="SelectedStudy"
    #)

    #TermInStudy = nl.add_tuple_set(ns_terms, name="TermInStudy")
    #FocusReported = nl.add_tuple_set(ns_database, name="FocusReported")
    #Voxel = nl.add_tuple_set(
    #    nib.affines.apply_affine(
    #        mni_mask.affine,
    #        np.transpose(mni_mask.get_fdata().nonzero())
    #    ),
    #    name='Voxel'
    #)

    @nl.add_symbol
    def word_lower(name: str) -> str:
        return name.lower()

    part_of = nl.new_symbol(name='ro.owl:part_of')
    label = nl.new_symbol(name='rdf-schema:label')
    perception = nl.new_symbol(name='cogat.owl:CAO_00418')
    attention = nl.new_symbol(name='cogat.owl:CAO_00141')
    listening = nl.new_symbol(name='cogat.owl:CAO_00370')
    auditory_attention = nl.new_symbol(name='cogat.owl:CAO_00149')
    spatial_attention = nl.new_symbol(name='cogat.owl:CAO_00507')
    visual_attention = nl.new_symbol(name='cogat.owl:CAO_00541')
    consciousness = nl.new_symbol(name='cogat.owl:CAO_00216')
    attention_capacity = nl.new_symbol(name='cogat.owl:CAO_00142')
    autonoesis = nl.new_symbol(name='cogat.owl:CAO_00693')
    episodic_memory = nl.new_symbol(name='cogat.owl:CAO_00277') 

    altLabel = nl.new_symbol(name='core:altLabel')

    with nl.scope as e:
        #e.answer[e.a] = (
        #    perception[e.a]   
        #)

        e.ontology_synonym[e.ne, e.l] = (
            label(e.e, e.ne) &
            altLabel(e.e, e.r) &
            (e.l == word_lower[e.r])
        )

        f_term = nl.query((e.ne, e.l), e.ontology_synonym(e.ne, e.l))

    a = 1 