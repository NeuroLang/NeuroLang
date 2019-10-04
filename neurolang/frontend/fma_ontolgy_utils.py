from owlready2 import get_ontology
from nilearn import datasets

import nibabel as nib
import os.path

from urllib.request import urlretrieve

class FMAOntology():
    def __init__(self, path=None):
        if path is None:
            if not os.path.exists('fma.owl'):
                print("Downloading FMA Ontology")
                urlretrieve("http://data.bioontology.org/ontologies/FMA/submissions/29/download?apikey=8b5b7825-538d-40e0-9e9e-5ab9274a9aeb", "fma.owl")

            path = 'fma.owl'
        self._ontology = get_ontology(path)
        self.loaded = False

    def load_ontology(self):
        if not self.loaded:
            self._ontology.load()
            self.loaded = True

    def get_nodes(self, main_label):
        if self.loaded:
            return self.process_nodes(main_label)

    def get_subclasses(self, main_label):
        if self.loaded:
            return self.process_subclasses(main_label, apply=self.parse_isSub)

    def get_synonyms(self, main_label):
        if self.loaded:
            return self.process_synonyms(main_label, apply=self.parse_isSyn)

    def process_nodes(self, main_label):
        labels = []
        for main in self._ontology.search(label=main_label):
            subs = list(main.subclasses())
            while subs:
                subc = subs.pop(0)
                labels.append(subc.label[0])
                subs = subs + list(subc.subclasses())
        labels = list(dict.fromkeys(labels))
        return labels

    def process_subclasses(self, main_label, apply=None):
        subclasses = dict()
        for main in self._ontology.search(label=main_label):
            subs = [main]
            while subs:
                subc = subs.pop(0)
                temp = list(subc.subclasses())
                if len(temp) > 0:
                    subclasses[subc.label[0]] = [e.label[0] for e in temp]
                    subs = subs + temp
        if apply is not None:
            subclasses = apply(subclasses)
        return subclasses

    def parse_isSub(self, subc):
        subclass = []
        for k, v in subc.items():
            for elem in v:
                subclass.append((k, elem))
        return subclass

    def process_synonyms(self, main_label, apply=None):
        synonyms = dict()
        for main in self._ontology.search(label=main_label):
            subs = [main]
            while subs:
                subc = subs.pop(0)
                temp = list(subc.subclasses())
                syn = subc.synonym
                if len(syn) > 0:
                    synonyms[subc.label[0]] = syn
                subs = subs + temp
        if apply is not None:
            synonyms = apply(synonyms)
        return synonyms

    def parse_isSyn(self, syn):
        syns = []
        for k, v in syn.items():
            for elem in v:
                syns.append((k, elem))
                syns.append((elem, k))
        return syns

    def get_destrieux_relations(self):
        return [
            ('l_g_and_s_frontomargin', 'Left frontomarginal gyrus'),
            ('l_g_and_s_occipital_inf', 'Left inferior occipital gyrus'),
            ('l_g_and_s_paracentral', 'Left paracentral lobule'),
            ('l_g_and_s_subcentral', 'Left subcentral gyrus'),
            (
                'l_g_and_s_transv_frontopol',
                'Left superior transverse frontopolar gyrus'
            ),
            ('l_g_and_s_cingul_ant', 'Left anterior cingulate gyrus'),
            (
                'l_g_and_s_cingul_mid_ant',
                'Left anterior middle cingulate gyrus'
            ),
            (
                'l_g_and_s_cingul_mid_post',
                'Left posterior middle cingulate gyrus'
            ),
            (
                'l_g_cingul_post_dorsal',
                'Dorsal segment of left posterior middle cingulate gyrus'
            ),
            (
                'l_g_cingul_post_ventral',
                'Ventral segment of left posterior middle cingulate gyrus'
            ),
            ('l_g_cuneus', 'Left cuneus'),
            (
                'l_g_front_inf_opercular',
                'Opercular part of left inferior frontal gyrus'
            ),
            (
                'l_g_front_inf_orbital',
                'Orbital part of left inferior frontal gyrus'
            ),
            (
                'l_g_front_inf_triangul',
                'Triangular part of left inferior frontal gyrus'
            ),
            ('l_g_front_middle', 'Left middle frontal gyrus'),
            ('l_g_front_sup', 'Left superior frontal gyrus'),
            ('l_g_ins_lg_and_s_cent_ins', 'Left central insular sulcus'),
            ('l_g_ins_lg_and_s_cent_ins', 'Left long insular gyrus'),
            ('l_g_insular_short', 'Short insular gyrus'),
            ('l_g_occipital_middleLeft', 'lateral occipital gyru'),
            ('l_g_occipital_sup', 'Left superior occipital gyrus'),
            ('l_g_oc_temp_lat_fusifor', 'Left fusiform gyrus'),
            ('l_g_oc_temp_med_lingual', 'Left lingual gyrus'),
            ('l_g_oc_temp_med_parahip', 'Left parahippocampal gyrus'),
            ('l_g_orbital', 'Left orbital gyrus'),
            ('l_g_pariet_inf_angular', 'Left angular gyrus'),
            ('l_g_pariet_inf_supramar', 'Left supramarginal gyrus'),
            ('l_g_parietal_sup', 'Left superior parietal lobule'),
            ('l_g_postcentral', 'Left postcentral gyrus'),
            ('l_g_precentral', 'Left precentral gyrus'),
            ('l_g_precuneus', 'Left precuneus'),
            ('l_g_rectus', 'Left straight gyrus'),
            ('l_g_subcallosal', 'Left paraterminal gyrus'),
            ('l_g_temp_sup_g_t_transv', 'Left transverse temporal gyrus'),
            ('l_g_temp_sup_lateral', 'Left superior temporal gyrus'),
            ('l_g_temp_sup_plan_polar', 'Left superior temporal gyrus'),
            ('l_g_temp_sup_plan_tempo', 'Left superior temporal gyrus'),
            ('l_g_temporal_inf', 'Left inferior temporal gyrus'),
            ('l_g_temporal_middle', 'Left middle temporal gyrus'),
            (
                'l_lat_fis_ant_horizont',
                'Anterior horizontal limb of left lateral sulcus'
            ),
            (
                'l_lat_fis_ant_vertical',
                'Anterior ascending limb of left lateral sulcus'
            ),
            (
                'l_lat_fis_post',
                'Posterior ascending limb of left lateral sulcus'
            ),
            ('l_lat_fis_post', 'Left lateral sulcus'),
            ('l_pole_occipital', 'Left occipital pole'),
            ('l_pole_temporal', 'Left temporal pole'),
            ('l_s_calcarine', 'Left Calcarine sulcus'),
            ('l_s_central', 'Left central sulcus'),
            ('l_s_cingul_marginalis', 'Left marginal sulcus'),
            ('l_s_circular_insula_ant', 'Circular sulcus of left insula'),
            ('l_s_circular_insula_inf', 'Circular sulcus of left insula'),
            ('l_s_circular_insula_sup', 'Circular sulcus of left insula'),
            ('l_s_collat_transv_ant', 'Left collateral sulcus'),
            ('l_s_collat_transv_post', 'Left collateral sulcus'),
            ('l_s_front_inf', 'Left inferior frontal sulcus'),
            ('l_s_front_sup', 'Left superior frontal sulcus'),
            ('l_s_intrapariet_and_p_trans', 'Left intraparietal sulcus'),
            ('l_s_oc_middle_and_lunatus', 'Left lunate sulcus'),
            ('l_s_oc_sup_and_transversal', 'Left transverse occipital sulcus'),
            ('l_s_occipital_ant', 'Left anterior occipital sulcus'),
            ('l_s_oc_temp_lat', 'Left occipitotemporal sulcus'),
            ('l_s_oc_temp_med_and_lingual', 'Left intralingual sulcus'),
            ('l_s_orbital_lateral', 'Left orbital sulcus'),
            ('l_s_orbital_med_olfact', 'Left olfactory sulcus'),
            ('l_s_orbital_h_shaped', 'Left transverse orbital sulcus'),
            ('l_s_orbital_h_shaped', 'Left orbital sulcus'),
            ('l_s_parieto_occipital', 'Left parieto-occipital sulcus'),
            ('l_s_pericallosal', 'Left callosal sulcus'),
            ('l_s_postcentral', 'Left postcentral sulcus'),
            ('l_s_precentral_inf_part', 'Left precentral sulcus'),
            ('l_s_precentral_sup_part', 'Left precentral sulcus'),
            ('l_s_suborbital', 'Left fronto-orbital sulcus'),
            ('l_s_subparietal', 'Left subparietal sulcus'),
            ('l_s_temporal_inf', 'Left inferior temporal sulcus'),
            ('l_s_temporal_sup', 'Left superior temporal sulcus'),
            ('l_s_temporal_transverse', 'Left transverse temporal sulcus'),
            ('r_g_and_s_frontomargin', 'Right frontomarginal gyrus'),
            ('r_g_and_s_occipital_inf', 'Right inferior occipital gyrus'),
            ('r_g_and_s_paracentral', 'Right paracentral lobule'),
            ('r_g_and_s_subcentral', 'Right subcentral gyrus'),
            (
                'r_g_and_s_transv_frontopol',
                'Right superior transverse frontopolar gyrus'
            ),
            ('r_g_and_s_cingul_ant', 'Right anterior cingulate gyrus'),
            (
                'r_g_and_s_cingul_mid_ant',
                'Right anterior middle cingulate gyrus'
            ),
            (
                'r_g_and_s_cingul_mid_post',
                'Right posterior middle cingulate gyrus'
            ),
            (
                'r_g_cingul_post_dorsal',
                'Dorsal segment of right posterior middle cingulate gyrus'
            ),
            (
                'r_g_cingul_post_ventral',
                'Ventral segment of right posterior middle cingulate gyrus'
            ),
            ('r_g_cuneus', 'Right cuneus'),
            (
                'r_g_front_inf_opercular',
                'Opercular part of right inferior frontal gyrus'
            ),
            (
                'r_g_front_inf_orbital',
                'Orbital part of right inferior frontal gyrus'
            ),
            (
                'r_g_front_inf_triangul',
                'Triangular part of right inferior frontal gyrus'
            ),
            ('r_g_front_middle', 'Right middle frontal gyrus'),
            ('r_g_front_sup', 'Right superior frontal gyrus'),
            ('r_g_ins_lg_and_s_cent_ins', 'Right central insular sulcus'),
            ('r_g_ins_lg_and_s_cent_ins', 'Right long insular gyrus'),
            ('r_g_insular_short', 'Right short insular gyrus'),
            ('r_g_occipital_middle', 'Right lateral occipital gyrus'),
            ('r_g_occipital_sup', 'Right superior occipital gyrus'),
            ('r_g_oc_temp_lat_fusifor', 'Right fusiform gyrus'),
            ('r_g_oc_temp_med_lingual', 'Right lingual gyrus'),
            ('r_g_oc_temp_med_parahip', 'Right parahippocampal gyrus'),
            ('r_g_orbital', 'Right orbital gyrus'),
            ('r_g_pariet_inf_angular', 'Right angular gyrus'),
            ('r_g_pariet_inf_supramar', 'Right supramarginal gyrus'),
            ('r_g_parietal_sup', 'Right superior parietal lobule'),
            ('r_g_postcentral', 'Right postcentral gyrus'),
            ('r_g_precentral', 'Right precentral gyrus'),
            ('r_g_precuneus', 'Right precuneus'),
            ('r_g_rectus', 'Right straight gyrus'),
            ('r_g_subcallosal', 'Right paraterminal gyrus'),
            ('r_g_temp_sup_g_t_transv', 'Right transverse temporal gyrus'),
            ('r_g_temp_sup_lateral', 'Right superior temporal gyrus'),
            ('r_g_temp_sup_plan_polar', 'Right superior temporal gyrus'),
            ('r_g_temp_sup_plan_tempo', 'Right superior temporal gyrus'),
            ('r_g_temporal_inf', 'Right inferior temporal gyrus'),
            ('r_g_temporal_middle', 'Right middle temporal gyrus'),
            (
                'r_lat_fis_ant_horizont',
                'Anterior horizontal limb of right lateral sulcus'
            ),
            (
                'r_lat_fis_ant_vertical',
                'Anterior ascending limb of right lateral sulcus'
            ),
            ('r_lat_fis_post', 'Right lateral sulcus'),
            (
                'r_lat_fis_post',
                'Posterior ascending limb of right lateral sulcus'
            ),
            ('r_pole_occipital', 'Right occipital pole'),
            ('r_pole_temporal', 'Right temporal pole'),
            ('r_s_calcarine', 'Right Calcarine sulcus'),
            ('r_s_central', 'Right central sulcus'),
            ('r_s_cingul_marginalis', 'Right marginal sulcus'),
            ('r_s_circular_insula_ant', 'Circular sulcus of Right insula'),
            ('r_s_circular_insula_inf', 'Circular sulcus of Right insula'),
            ('r_s_circular_insula_sup', 'Circular sulcus of Right insula'),
            ('r_s_collat_transv_ant', 'Right collateral sulcus'),
            ('r_s_collat_transv_post', 'Right collateral sulcus'),
            ('r_s_front_inf', 'Right inferior frontal sulcus'),
            ('r_s_front_sup', 'Right superior frontal sulcus'),
            ('r_s_intrapariet_and_p_trans', 'Right intraparietal sulcus'),
            ('r_s_oc_middle_and_lunatus', 'Right lunate sulcus'),
            (
                'r_s_oc_sup_and_transversal',
                'Right transverse occipital sulcus'
            ),
            ('r_s_occipital_ant', 'Right anterior occipital sulcus'),
            ('r_s_oc_temp_lat', 'Right occipitotemporal sulcus'),
            ('r_s_oc_temp_med_and_lingual', 'Right intralingual sulcus'),
            ('r_s_orbital_lateral', 'Right orbital sulcus'),
            ('r_s_orbital_med_olfact', 'Right olfactory sulcus'),
            ('r_s_orbital_h_shaped', 'Right orbital sulcus'),
            ('r_s_orbital_h_shaped', 'Right transverse orbital sulcus'),
            ('r_s_parieto_occipital', 'Right parieto-occipital sulcus'),
            ('r_s_pericallosal', 'Right callosal sulcus'),
            ('r_s_postcentral', 'Right postcentral sulcus'),
            ('r_s_precentral_inf_part', 'Right precentral sulcus'),
            ('r_s_precentral_sup_part', 'Right precentral sulcus'),
            ('r_s_suborbital', 'Right fronto-orbital sulcus'),
            ('r_s_subparietal', 'Right subparietal sulcus'),
            ('r_s_temporal_inf', 'Right inferior temporal sulcus'),
            ('r_s_temporal_sup', 'Right superior temporal sulcus'),
            ('r_s_temporal_transverse', 'Right transverse temporal sulcus'),
        ]

    def init_ontology(
        self,
        neurolangDL,
        root_node='Segment of brain',
        destriuex_relations=False
    ):

        nodes = self.get_nodes(root_node)
        subc = self.get_subclasses(root_node)
        syn = self.get_synonyms(root_node)

        neurolangDL.add_tuple_set(((str(e), ) for e in nodes), name='nodes')

        neurolangDL.add_tuple_set(((
            str(e1),
            str(e2),
        ) for e1, e2 in subc),
                                  name='subclasses')

        neurolangDL.add_tuple_set(((
            str(e1),
            str(e2),
        ) for e1, e2 in syn),
                                  name='synonyms')

        x = neurolangDL.new_symbol(name='x')
        y = neurolangDL.new_symbol(name='y')
        z = neurolangDL.new_symbol(name='z')
        val = neurolangDL.new_symbol(name='val')
        isSub = neurolangDL.new_symbol(name='isSub')
        isSyn = neurolangDL.new_symbol(name='isSyn')

        if destriuex_relations:
            destrieux_dataset = datasets.fetch_atlas_destrieux_2009()
            destrieux_map = nib.load(destrieux_dataset['maps'])

            d = []
            for label_number, name in destrieux_dataset['labels']:
                if label_number == 0:
                    continue
                name = name.decode()
                region = neurolangDL.create_region(
                    destrieux_map, label=label_number
                )
                if region is None:
                    continue
                name = name.replace('-', '_').replace(' ', '_')
                d.append((name.lower(), region))

            neurolangDL.add_tuple_set(((
                e1,
                e2,
            ) for e1, e2 in d),
                                      name='destrieux_regions')

            destrieux = self.get_destrieux_relations()
            relation = neurolangDL.new_symbol(name='relation')

            neurolangDL.add_tuple_set(((
                e1,
                e2,
            ) for e1, e2 in destrieux),
                                      name='relations')

            neurolangDL.query(
                relation(x, y), neurolangDL.symbols.relations(x, y)
            )

        neurolangDL.query(val(x), neurolangDL.symbols.nodes(x))
        neurolangDL.query(isSyn(x, y), neurolangDL.symbols.synonyms(x, y))
        neurolangDL.query(isSub(x, y), neurolangDL.symbols.subclasses(x, y))

        neurolangDL.query(isSub(x, z), isSub(x, y) & isSub(y, z))

        return neurolangDL