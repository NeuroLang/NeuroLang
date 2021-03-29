
import io

from ...frontend import NeurolangPDL
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


    x = Symbol('x')
    y = Symbol('y')
    Professor = Symbol('professor')
    Chair = Symbol('chair')
    supportChair = Symbol.fresh()
    headOf = Symbol('headof')
    Department = Symbol('department')
    Person = Symbol('person')

    imp1 = RightImplication(supportChair(x, y), Person(x))
    imp2 = RightImplication(Chair(x), supportChair(x, y))
    imp3 = RightImplication(supportChair(x, y), headOf(x, y))
    imp4 = RightImplication(supportChair(x, y), Department(y))
    imp5 = Implication(Professor(x), Chair(x))

    onto = OntologyParser(io.StringIO(owl))
    rules, constraints = onto.parse_ontology()

    
    assert len(constraints) == 4
    assert len(rules) == 1

    assert isinstance(constraints[0], RightImplication)
    assert len(constraints[0].antecedent.args) == 2
    assert len(constraints[0].consequent.args) == 1
    assert constraints[0].antecedent.args[0] == constraints[0].consequent.args[0]
    assert constraints[0].consequent.functor == imp1.consequent.functor

    assert isinstance(constraints[1], RightImplication)
    assert len(constraints[1].antecedent.args) == 2
    assert len(constraints[1].consequent.args) == 1
    assert constraints[1].antecedent.args[1] == constraints[1].consequent.args[0]
    assert constraints[1].consequent.functor == imp4.consequent.functor

    assert isinstance(constraints[2], RightImplication)
    assert len(constraints[2].antecedent.args) == 1
    assert len(constraints[2].consequent.args) == 2
    assert constraints[2].antecedent.args[0] == constraints[2].consequent.args[0]
    assert constraints[2].antecedent.functor == imp2.antecedent.functor

    assert isinstance(constraints[3], RightImplication)
    assert len(constraints[3].antecedent.args) == 2
    assert len(constraints[3].consequent.args) == 2
    assert constraints[3].antecedent.args[0] == constraints[3].consequent.args[0]
    assert constraints[3].consequent.functor == imp3.consequent.functor
    
    assert isinstance(rules[0], Implication)
    assert len(rules[0].antecedent.args) == 1
    assert len(rules[0].consequent.args) == 1
    assert rules[0].antecedent.args[0] == rules[0].consequent.args[0]
    assert rules[0].antecedent.functor == imp5.antecedent.functor
    assert rules[0].consequent.functor == imp5.consequent.functor


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

    imp1 = RightImplication(supportDean(x, y), Person(x))
    imp2 = RightImplication(Dean(x), supportDean(x, y))
    imp3 = RightImplication(supportDean(x, y), headOf(x, y))
    imp4 = RightImplication(supportDean(x, y), College(y))
    imp5 = Implication(Professor(x), Dean(x))

    onto = OntologyParser(io.StringIO(owl))
    rules, constraints = onto.parse_ontology()

    
    assert len(constraints) == 4
    assert len(rules) == 1

    assert isinstance(constraints[0], RightImplication)
    assert len(constraints[0].antecedent.args) == 2
    assert len(constraints[0].consequent.args) == 1
    assert constraints[0].antecedent.args[0] == constraints[0].consequent.args[0]
    assert constraints[0].consequent.functor == imp1.consequent.functor

    assert isinstance(constraints[1], RightImplication)
    assert len(constraints[1].antecedent.args) == 2
    assert len(constraints[1].consequent.args) == 1
    assert constraints[1].antecedent.args[1] == constraints[1].consequent.args[0]
    assert constraints[1].consequent.functor == imp4.consequent.functor

    assert isinstance(constraints[2], RightImplication)
    assert len(constraints[2].antecedent.args) == 1
    assert len(constraints[2].consequent.args) == 2
    assert constraints[2].antecedent.args[0] == constraints[2].consequent.args[0]
    assert constraints[2].antecedent.functor == imp2.antecedent.functor

    assert isinstance(constraints[3], RightImplication)
    assert len(constraints[3].antecedent.args) == 2
    assert len(constraints[3].consequent.args) == 2
    assert constraints[3].antecedent.args[0] == constraints[3].consequent.args[0]
    assert constraints[3].consequent.functor == imp3.consequent.functor
    
    assert isinstance(rules[0], Implication)
    assert len(rules[0].antecedent.args) == 1
    assert len(rules[0].consequent.args) == 1
    assert rules[0].antecedent.args[0] == rules[0].consequent.args[0]
    assert rules[0].antecedent.functor == imp5.antecedent.functor
    assert rules[0].consequent.functor == imp5.consequent.functor


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

    nl.add_tuple_set([('Juan',), ('Manuel',)], name='dean')
    nl.add_tuple_set([('Miguel',), ('Alberto',)], name='chair')
    nl.add_tuple_set([('College A',), ('College B',)], name='college')
    nl.add_tuple_set([('Department A',), ('Department B',)], name='department')

    with nl.scope as e:
        e.answer[e.a] = (
            e.headof[e.a, e.b] &
            e.college[e.b]
        )

        f_term = nl.solve_all()
    
    res = f_term['answer'].as_pandas_dataframe().values
    assert (res == [['Juan'], ['Manuel']]).all()


def test_cogat():
    from nilearn import datasets, image
    import pandas as pd
    import numpy as np
    import nibabel as nib

    cogAt = datasets.utils._fetch_files(
        datasets.utils._get_dataset_dir('CogAt'),
        [
            (
                'cogat.xml',
                'http://data.bioontology.org/ontologies/COGAT/download?'
                'apikey=8b5b7825-538d-40e0-9e9e-5ab9274a9aeb&download_format=rdf',
                {'move': 'cogat.xml'}
            )
        ]
    )[0]

    mni_t1 = nib.load(datasets.fetch_icbm152_2009()['t1'])
    mni_t1_4mm = image.resample_img(mni_t1, np.eye(3) * 4)

    ns_database_fn, ns_features_fn = datasets.utils._fetch_files(
        datasets.utils._get_dataset_dir('neurosynth'),
        [
            (
                'database.txt',
                'https://github.com/neurosynth/neurosynth-data/raw/master/current_data.tar.gz',
                {'uncompress': True}
            ),
            (
                'features.txt',
                'https://github.com/neurosynth/neurosynth-data/raw/master/current_data.tar.gz',
                {'uncompress': True}
            ),
        ]
    )

    ns_database = pd.read_csv(ns_database_fn, sep=f'\t')
    ijk_positions = (
        nib.affines.apply_affine(
            np.linalg.inv(mni_t1_4mm.affine),
            ns_database[['x', 'y', 'z']]
        ).astype(int)
    )
    ns_database['i'] = ijk_positions[:, 0]
    ns_database['j'] = ijk_positions[:, 1]
    ns_database['k'] = ijk_positions[:, 2]

    ns_features = pd.read_csv(ns_features_fn, sep=f'\t')
    ns_terms = (
        pd.melt(
                ns_features,
                var_name='term', id_vars='pmid', value_name='TfIdf'
        )
        .query('TfIdf > 1e-3')[['pmid', 'term']]
    )
    ns_docs = ns_features[['pmid']].drop_duplicates()

    import rdflib
    from rdflib import RDFS
    g = rdflib.Graph()
    g.load(cogAt)

    onto_dic = {}
    for a in g.subjects():
        for b in g.triples((a, RDFS.label, None)):
            ent = a.split('#')[1]
            label = b[2].lower().replace(' ', '_')
            onto_dic[label] = ent

    group_terms = ns_terms.groupby('term')
    dic_term_pmid = {}

    for term, ids in group_terms:
        term = term.lower().replace(' ', '_')
        dic_term_pmid[term] = ids
        
    merge_dic = {}
    for k, v in onto_dic.items():
        if k in dic_term_pmid.keys():
            vl = v.lower()
            merge_dic[vl] = dic_term_pmid[k]

    nl = NeurolangPDL()
    nl.load_ontology(cogAt)

    for k, v in dic_term_pmid.items():
        nl.add_tuple_set(v.pmid.values, name=k)  

    with nl.scope as e:
        e.answer[e.a] = (
            e.part_of[e.a, e.b] &
            e.cao_00418[e.b] #perception 
            
        )

        f_term = nl.solve_all()

    a = 1 