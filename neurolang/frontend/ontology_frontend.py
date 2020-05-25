from .query_resolution_datalog import QueryBuilderDatalog
from . import RegionFrontendDatalogSolver

import pandas as pd
import nibabel as nib
import numpy as np
import pickle

from rdflib import OWL, RDFS
from nilearn import datasets, plotting
from matplotlib import pyplot as plt
from scipy import special
from scipy.stats import norm

from neurolang.datalog import DatalogProgram
from neurolang.expressions import Symbol, Constant, ExpressionBlock
from neurolang.logic import Implication, Union
from neurolang.datalog.constraints_representation import (
    DatalogConstraintsProgram,
)
from neurolang.expression_walker import (
    ExpressionBasicEvaluator,
    IdentityWalker,
)
from neurolang.datalog.expressions import TranslateToLogic
from neurolang.datalog.aggregation import DatalogWithAggregationMixin
from neurolang.frontend.query_resolution import RegionMixin
from neurolang.datalog.aggregation import AggregationApplication, Chase
from neurolang.regions import (
    Region,
    region_union as region_union_,
    region_intersection as region_intersection_,
    ExplicitVBR,
)
from neurolang import frontend as fe
from neurolang.datalog.chase import (
    ChaseSemiNaive,
    ChaseNaive,
    ChaseNamedRelationalAlgebraMixin,
    ChaseGeneral,
)
from neurolang.datalog.ontologies_parser import OntologyParser
from neurolang.datalog.ontologies_rewriter import OntologyRewriter
from neurolang.region_solver import RegionSolver
from .neurosynth_utils import NeuroSynthHandler
from neurolang.datalog.expression_processing import (
    reachable_code,
    extract_logic_predicates,
)


class Chase(Chase, ChaseNaive, ChaseNamedRelationalAlgebraMixin, ChaseGeneral):
    pass


class DatalogTranslator(
    TranslateToLogic, IdentityWalker, DatalogWithAggregationMixin
):
    pass


class Datalog(
    TranslateToLogic,
    DatalogWithAggregationMixin,
    DatalogProgram,
    ExpressionBasicEvaluator,
):
    pass


class DatalogRegions(
    TranslateToLogic,
    RegionSolver,
    DatalogWithAggregationMixin,
    DatalogProgram,
    ExpressionBasicEvaluator,
):
    pass


class NeurolangOntologyDL(QueryBuilderDatalog):
    def __init__(self, paths, load_format="xml", solver=None):
        if solver is None:
            solver = RegionFrontendDatalogSolver()

        self.onto = OntologyParser(paths, load_format)
        d_pred, u_constraints = self.onto.parse_ontology()
        self.d_pred = d_pred
        self.u_constraints = u_constraints

        super().__init__(solver, chase_class=Chase)

    def separate_deterministic_probabilistic_code(
        self, det_symbols=None, prob_symbols=None
    ):
        if det_symbols is None:
            det_symbols = set()
        if prob_symbols is None:
            prob_symbols = set()

        assert len(self.current_program.expression.formulas) == 1
        query_pred = self.current_program.expression.formulas[0]
        query_reachable_code = reachable_code(query_pred, self.solver)
        deterministic_symbols = set(
            self.solver.extensional_database().keys()
        ) | set(det_symbols)
        deterministic_program = list()
        probabilistic_symbols = set() | set(prob_symbols)
        probabilistic_program = list()
        unclassified_code = list(query_reachable_code.formulas)
        unclassified = 0
        initial_unclassified_length = len(unclassified_code) + 1
        while (
            len(unclassified_code) > 0
            and unclassified <= initial_unclassified_length
        ):
            pred = unclassified_code.pop(0)
            preds_antecedent = set(
                p.functor
                for p in extract_logic_predicates(pred.antecedent)
                if p.functor != pred.consequent.functor
            )
            if not probabilistic_symbols.isdisjoint(preds_antecedent):
                probabilistic_symbols.add(pred.consequent.functor)
                probabilistic_program.append(pred)
                unclassified = 0
                initial_unclassified_length = len(unclassified_code) + 1
            elif deterministic_symbols.issuperset(preds_antecedent):
                deterministic_symbols.add(pred.consequent.functor)
                deterministic_program.append(pred)
                unclassified = 0
                initial_unclassified_length = len(unclassified_code) + 1
            else:
                unclassified_code.append(pred)
                unclassified += 1
        assert probabilistic_symbols.isdisjoint(deterministic_symbols)
        assert len(unclassified_code) == 0

        return deterministic_program, probabilistic_program

    def solve_query(self, symbol_prob, ns_terms=None):
        self.ns_terms = ns_terms
        if self.ns_terms is not None:
            prob_terms, prob_terms_voxels = self.load_neurosynth_database(
                self.ns_terms
            )
            self.prob_terms = prob_terms
            self.prob_terms_voxels = prob_terms_voxels

        eB2 = self.rewrite_database_with_ontology()
        dl = self.load_facts(eB2)
        sol = self.build_chase_solution(dl)
        # dlProb = self.load_probabilistic_facts(sol)
        # result = self.solve_probabilistic_query(dlProb, symbol_prob)

        return sol

    def load_neurosynth_database(self, terms):
        nsh = NeuroSynthHandler()
        data = nsh.ns_study_tfidf_feature_for_terms(terms)
        ns_data_term_voxel = pd.DataFrame(
            data, columns=["study", "term", "prob"]
        )
        ns_data_term_voxel = ns_data_term_voxel[["prob", "term", "study"]]
        ns_data_term_voxel = ns_data_term_voxel[ns_data_term_voxel.prob > 0]

        data = nsh.ns_prob_terms(terms)
        ns_data_term = pd.DataFrame(data, columns=["term", "prob"])
        ns_data_term = ns_data_term[["prob", "term"]]

        return ns_data_term, ns_data_term_voxel

    def rewrite_database_with_ontology(self):
        orw = OntologyRewriter(self.get_expressions(), self.u_constraints)
        rewrite = orw.Xrewrite()

        eB2 = ()
        for imp in rewrite:
            eB2 += (imp[0],)

        return Union(eB2)

    def load_facts(self, eB2):
        destrieux_to_voxels = Symbol("destrieux_to_voxels")
        relation_name = Symbol("relation_name")

        with open("./data/regiones.pickle", "rb") as fp:
            inter_regions = pickle.load(fp)

        relations_list = self.destrieux_name_to_fma_relations()
        r_name = tuple(
            [
                relation_name(Constant(destrieux), Constant(fma))
                for destrieux, fma in relations_list
            ]
        )

        dl = self.solver
        dl.add_extensional_predicate_from_tuples(
            self.onto.get_triples_symbol(),
            self.d_pred[self.onto.get_triples_symbol()],
        )
        dl.add_extensional_predicate_from_tuples(
            self.onto.get_pointers_symbol(),
            self.d_pred[self.onto.get_pointers_symbol()],
        )
        dl.add_extensional_predicate_from_tuples(
            destrieux_to_voxels,
            [(ns, ds, region) for ns, ds, region in inter_regions],
        )
        dl.add_extensional_predicate_from_tuples(
            relation_name, [(a.args[0].value, a.args[1].value) for a in r_name]
        )

        dl.walk(eB2)

        return dl

    def build_chase_solution(self, dl):

        dc = Chase(dl)
        solution_instance = dc.build_chase_solution()
        return solution_instance

    def load_probabilistic_facts(self, solution_instance):
        dlProb = ProbDatalogProgram()

        file = open("./data/xyz_from_neurosynth.pkl", "rb")
        ret = pickle.load(file)
        file.close()

        term = Symbol("term")
        neurosynth_data = Symbol("neurosynth_data")
        neurosynth_region = Symbol("neurosynth_region")

        for instance in solution_instance.items():
            dlProb.add_extensional_predicate_from_tuples(
                instance[0], instance[1].value
            )

        if self.ns_terms is not None:
            dlProb.add_probfacts_from_tuples(
                term, set(self.prob_terms.itertuples(index=False, name=None))
            )
            dlProb.add_probfacts_from_tuples(
                neurosynth_data,
                set(self.prob_terms_voxels.itertuples(index=False, name=None)),
            )

        dlProb.add_extensional_predicate_from_tuples(
            neurosynth_region, [(k, v) for k, v in ret.items()]
        )

        return dlProb

    def solve_probabilistic_query(self, dlProb, symbol):
        dt2 = DatalogTranslator()
        eb = dt2.walk(self.get_prob_expressions())
        dlProb.walk(eb)

        z = Symbol.fresh()

        dl_program = probdatalog_to_datalog(dlProb, datalog=DatalogRegions)
        dc = Chase(dl_program)
        solution_instance = dc.build_chase_solution()
        grounded = build_grounding(dlProb, solution_instance)

        gm = TranslateGroundedProbDatalogToGraphicalModel().walk(grounded)
        sym = symbol.expression.formulas[0].consequent.function
        query = SuccQuery(sym(z))
        solver = QueryGraphicalModelSolver(gm)
        result = solver.walk(query)

        return result

    def get_expressions(self):
        exp = ()
        for e in self.current_program:
            if not str.startswith(
                e.expression.consequent.functor.name, "probability"
            ):
                exp = exp + (e.expression,)

        return Union(exp)

    def get_prob_expressions(self):
        exp = ()
        for e in self.current_program:
            if str.startswith(
                e.expression.consequent.functor.name, "probability"
            ):
                exp = exp + (e.expression,)

        return Union(exp)

    def destrieux_name_to_fma_relations(self):
        return [
            ("l_g_and_s_frontomargin", "Left frontomarginal gyrus"),
            ("l_g_and_s_occipital_inf", "Left inferior occipital gyrus"),
            ("l_g_and_s_paracentral", "Left paracentral lobule"),
            ("l_g_and_s_subcentral", "Left subcentral gyrus"),
            (
                "l_g_and_s_transv_frontopol",
                "Left superior transverse frontopolar gyrus",
            ),
            ("l_g_and_s_cingul_ant", "Left anterior cingulate gyrus"),
            (
                "l_g_and_s_cingul_mid_ant",
                "Left anterior middle cingulate gyrus",
            ),
            (
                "l_g_and_s_cingul_mid_post",
                "Left posterior middle cingulate gyrus",
            ),
            (
                "l_g_cingul_post_dorsal",
                "Dorsal segment of left posterior middle cingulate gyrus",
            ),
            (
                "l_g_cingul_post_ventral",
                "Ventral segment of left posterior middle cingulate gyrus",
            ),
            ("l_g_cuneus", "Left cuneus"),
            (
                "l_g_front_inf_opercular",
                "Opercular part of left inferior frontal gyrus",
            ),
            (
                "l_g_front_inf_orbital",
                "Orbital part of left inferior frontal gyrus",
            ),
            (
                "l_g_front_inf_triangul",
                "Triangular part of left inferior frontal gyrus",
            ),
            ("l_g_front_middle", "Left middle frontal gyrus"),
            ("l_g_front_sup", "Left superior frontal gyrus"),
            ("l_g_ins_lg_and_s_cent_ins", "Left central insular sulcus"),
            ("l_g_ins_lg_and_s_cent_ins", "Left long insular gyrus"),
            ("l_g_insular_short", "Short insular gyrus"),
            ("l_g_occipital_middleLeft", " 	Left lateral occipital gyrus"),
            ("l_g_occipital_sup", "Left superior occipital gyrus"),
            ("l_g_oc_temp_lat_fusifor", "Left fusiform gyrus"),
            ("l_g_oc_temp_med_lingual", "Left lingual gyrus"),
            ("l_g_oc_temp_med_parahip", "Left parahippocampal gyrus"),
            ("l_g_orbital", "Left orbital gyrus"),
            ("l_g_pariet_inf_angular", "Left angular gyrus"),
            ("l_g_pariet_inf_supramar", "Left supramarginal gyrus"),
            ("l_g_parietal_sup", "Left superior parietal lobule"),
            ("l_g_postcentral", "Left postcentral gyrus"),
            ("l_g_precentral", "Left precentral gyrus"),
            ("l_g_precuneus", "Left precuneus"),
            ("l_g_rectus", "Left straight gyrus"),
            ("l_g_subcallosal", "Left paraterminal gyrus"),
            ("l_g_temp_sup_g_t_transv", "Left transverse temporal gyrus"),
            ("l_g_temp_sup_lateral", "Left superior temporal gyrus"),
            ("l_g_temp_sup_plan_polar", "Left superior temporal gyrus"),
            ("l_g_temp_sup_plan_tempo", "Left superior temporal gyrus"),
            ("l_g_temporal_inf", "Left inferior temporal gyrus"),
            ("l_g_temporal_middle", "Left middle temporal gyrus"),
            (
                "l_lat_fis_ant_horizont",
                "Anterior horizontal limb of left lateral sulcus",
            ),
            (
                "l_lat_fis_ant_vertical",
                "Anterior ascending limb of left lateral sulcus",
            ),
            (
                "l_lat_fis_post",
                "Posterior ascending limb of left lateral sulcus",
            ),
            ("l_lat_fis_post", "Left lateral sulcus"),
            ("l_pole_occipital", "Left occipital pole"),
            ("l_pole_temporal", "Left temporal pole"),
            ("l_s_calcarine", "Left Calcarine sulcus"),
            ("l_s_central", "Left central sulcus"),
            ("l_s_cingul_marginalis", "Left marginal sulcus"),
            ("l_s_circular_insula_ant", "Circular sulcus of left insula"),
            ("l_s_circular_insula_inf", "Circular sulcus of left insula"),
            ("l_s_circular_insula_sup", "Circular sulcus of left insula"),
            ("l_s_collat_transv_ant", "Left collateral sulcus"),
            ("l_s_collat_transv_post", "Left collateral sulcus"),
            ("l_s_front_inf", "Left inferior frontal sulcus"),
            ("l_s_front_sup", "Left superior frontal sulcus"),
            ("l_s_intrapariet_and_p_trans", "Left intraparietal sulcus"),
            ("l_s_oc_middle_and_lunatus", "Left lunate sulcus"),
            ("l_s_oc_sup_and_transversal", "Left transverse occipital sulcus"),
            ("l_s_occipital_ant", "Left anterior occipital sulcus"),
            ("l_s_oc_temp_lat", "Left occipitotemporal sulcus"),
            ("l_s_oc_temp_med_and_lingual", "Left intralingual sulcus"),
            ("l_s_orbital_lateral", "Left orbital sulcus"),
            ("l_s_orbital_med_olfact", "Left olfactory sulcus"),
            ("l_s_orbital_h_shaped", "Left transverse orbital sulcus"),
            ("l_s_orbital_h_shaped", "Left orbital sulcus"),
            ("l_s_parieto_occipital", "Left parieto-occipital sulcus"),
            ("l_s_pericallosal", "Left callosal sulcus"),
            ("l_s_postcentral", "Left postcentral sulcus"),
            ("l_s_precentral_inf_part", "Left precentral sulcus"),
            ("l_s_precentral_sup_part", "Left precentral sulcus"),
            ("l_s_suborbital", "Left fronto-orbital sulcus"),
            ("l_s_subparietal", "Left subparietal sulcus"),
            ("l_s_temporal_inf", "Left inferior temporal sulcus"),
            ("l_s_temporal_sup", "Left superior temporal sulcus"),
            ("l_s_temporal_transverse", "Left transverse temporal sulcus"),
            ("r_g_and_s_frontomargin", "Right frontomarginal gyrus"),
            ("r_g_and_s_occipital_inf", "Right inferior occipital gyrus"),
            ("r_g_and_s_paracentral", "Right paracentral lobule"),
            ("r_g_and_s_subcentral", "Right subcentral gyrus"),
            (
                "r_g_and_s_transv_frontopol",
                "Right superior transverse frontopolar gyrus",
            ),
            ("r_g_and_s_cingul_ant", "Right anterior cingulate gyrus"),
            (
                "r_g_and_s_cingul_mid_ant",
                "Right anterior middle cingulate gyrus",
            ),
            (
                "r_g_and_s_cingul_mid_post",
                "Right posterior middle cingulate gyrus",
            ),
            (
                "r_g_cingul_post_dorsal",
                "Dorsal segment of right posterior middle cingulate gyrus",
            ),
            (
                "r_g_cingul_post_ventral",
                "Ventral segment of right posterior middle cingulate gyrus",
            ),
            ("r_g_cuneus", "Right cuneus"),
            (
                "r_g_front_inf_opercular",
                "Opercular part of right inferior frontal gyrus",
            ),
            (
                "r_g_front_inf_orbital",
                "Orbital part of right inferior frontal gyrus",
            ),
            (
                "r_g_front_inf_triangul",
                "Triangular part of right inferior frontal gyrus",
            ),
            ("r_g_front_middle", "Right middle frontal gyrus"),
            ("r_g_front_sup", "Right superior frontal gyrus"),
            ("r_g_ins_lg_and_s_cent_ins", "Right central insular sulcus"),
            ("r_g_ins_lg_and_s_cent_ins", "Right long insular gyrus"),
            ("r_g_insular_short", "Right short insular gyrus"),
            ("r_g_occipital_middle", "Right lateral occipital gyrus"),
            ("r_g_occipital_sup", "Right superior occipital gyrus"),
            ("r_g_oc_temp_lat_fusifor", "Right fusiform gyrus"),
            ("r_g_oc_temp_med_lingual", "Right lingual gyrus"),
            ("r_g_oc_temp_med_parahip", "Right parahippocampal gyrus"),
            ("r_g_orbital", "Right orbital gyrus"),
            ("r_g_pariet_inf_angular", "Right angular gyrus"),
            ("r_g_pariet_inf_supramar", "Right supramarginal gyrus"),
            ("r_g_parietal_sup", "Right superior parietal lobule"),
            ("r_g_postcentral", "Right postcentral gyrus"),
            ("r_g_precentral", "Right precentral gyrus"),
            ("r_g_precuneus", "Right precuneus"),
            ("r_g_rectus", "Right straight gyrus"),
            ("r_g_subcallosal", "Right paraterminal gyrus"),
            ("r_g_temp_sup_g_t_transv", "Right transverse temporal gyrus"),
            ("r_g_temp_sup_lateral", "Right superior temporal gyrus"),
            ("r_g_temp_sup_plan_polar", "Right superior temporal gyrus"),
            ("r_g_temp_sup_plan_tempo", "Right superior temporal gyrus"),
            ("r_g_temporal_inf", "Right inferior temporal gyrus"),
            ("r_g_temporal_middle", "Right middle temporal gyrus"),
            (
                "r_lat_fis_ant_horizont",
                "Anterior horizontal limb of right lateral sulcus",
            ),
            (
                "r_lat_fis_ant_vertical",
                "Anterior ascending limb of right lateral sulcus",
            ),
            ("r_lat_fis_post", "Right lateral sulcus"),
            (
                "r_lat_fis_post",
                "Posterior ascending limb of right lateral sulcus",
            ),
            ("r_pole_occipital", "Right occipital pole"),
            ("r_pole_temporal", "Right temporal pole"),
            ("r_s_calcarine", "Right Calcarine sulcus"),
            ("r_s_central", "Right central sulcus"),
            ("r_s_cingul_marginalis", "Right marginal sulcus"),
            ("r_s_circular_insula_ant", "Circular sulcus of Right insula"),
            ("r_s_circular_insula_inf", "Circular sulcus of Right insula"),
            ("r_s_circular_insula_sup", "Circular sulcus of Right insula"),
            ("r_s_collat_transv_ant", "Right collateral sulcus"),
            ("r_s_collat_transv_post", "Right collateral sulcus"),
            ("r_s_front_inf", "Right inferior frontal sulcus"),
            ("r_s_front_sup", "Right superior frontal sulcus"),
            ("r_s_intrapariet_and_p_trans", "Right intraparietal sulcus"),
            ("r_s_oc_middle_and_lunatus", "Right lunate sulcus"),
            (
                "r_s_oc_sup_and_transversal",
                "Right transverse occipital sulcus",
            ),
            ("r_s_occipital_ant", "Right anterior occipital sulcus"),
            ("r_s_oc_temp_lat", "Right occipitotemporal sulcus"),
            ("r_s_oc_temp_med_and_lingual", "Right intralingual sulcus"),
            ("r_s_orbital_lateral", "Right orbital sulcus"),
            ("r_s_orbital_med_olfact", "Right olfactory sulcus"),
            ("r_s_orbital_h_shaped", "Right orbital sulcus"),
            ("r_s_orbital_h_shaped", "Right transverse orbital sulcus"),
            ("r_s_parieto_occipital", "Right parieto-occipital sulcus"),
            ("r_s_pericallosal", "Right callosal sulcus"),
            ("r_s_postcentral", "Right postcentral sulcus"),
            ("r_s_precentral_inf_part", "Right precentral sulcus"),
            ("r_s_precentral_sup_part", "Right precentral sulcus"),
            ("r_s_suborbital", "Right fronto-orbital sulcus"),
            ("r_s_subparietal", "Right subparietal sulcus"),
            ("r_s_temporal_inf", "Right inferior temporal sulcus"),
            ("r_s_temporal_sup", "Right superior temporal sulcus"),
            ("r_s_temporal_transverse", "Right transverse temporal sulcus"),
        ]
