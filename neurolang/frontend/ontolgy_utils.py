import rdflib
import pandas as pd
import nibabel as nib

from nilearn import datasets


class OntologyHandler():
    def __init__(self, paths, namespaces):
        self.namespaces_dic = None
        self.owl_dic = None
        if isinstance(paths, list):
            self.df = self._parse_ontology(paths, namespaces)
        else:
            self.df = self._parse_ontology([paths], [namespaces])

    def _parse_ontology(self, paths, namespaces):
        df = pd.DataFrame()
        temp = []
        for path in paths:
            g = rdflib.Graph()
            g.load(path)
            gdf = pd.DataFrame(iter(g))
            gdf = gdf.astype(str)
            gdf.columns = ['Entity', 'Property', 'Value']
            temp.append(gdf)

        df = df.append(temp)

        namespaces_properties = df[~df.Property.str.
                                   contains('#')].Property.unique()
        namespaces_properties = list(
            filter(
                lambda x: (x in n for n in namespaces), namespaces_properties
            )
        )
        namespaces_prop = list(
            map(
                lambda x: (x[0] + ':' + x[1]).lstrip('0123456789.-_ :'),
                list(map(lambda y: y.split('/')[-2:], namespaces_properties))
            )
        )
        self.namespaces_dic = dict(zip(namespaces_properties, namespaces_prop))

        owl_properties = df[df.Property.str.contains('#')].Property.unique()
        owl_rdf = list(
            map(
                lambda a: list(
                    map(
                        lambda s: s.replace('-', '_').lstrip('0123456789.-_ '),
                        a.split('/')[-1].split('#')
                    )
                ), owl_properties
            )
        )
        owl_rdf = list(map(lambda x: x[0] + ':' + x[1], owl_rdf))
        self.owl_dic = dict(zip(owl_properties, owl_rdf))

        return df

    def replace_property(self, prop):
        if prop in self.owl_dic:
            new_prop = self.owl_dic[prop]
        elif prop in self.namespaces_dic:
            new_prop = self.namespaces_dic[prop]
        else:
            new_prop = prop
        return new_prop

    def load_ontology(self, neurolangDL, destriuex_relations=False):
        neurolangDL.add_tuple_set(((e1, ) for e1, e2, e3 in self.df.values),
                                  name='dom')
        neurolangDL.add_tuple_set(((self.replace_property(e2), )
                                   for e1, e2, e3 in self.df.values),
                                  name='dom')
        neurolangDL.add_tuple_set(((e3, ) for e1, e2, e3 in self.df.values),
                                  name='dom')
        neurolangDL.add_tuple_set(((
            e1,
            self.replace_property(e2),
            e3,
        ) for e1, e2, e3 in self.df.values),
                                  name='triple')

        all_props = list(self.owl_dic.keys()
                         ) + list(self.namespaces_dic.keys())
        for prop in all_props:
            name = self.replace_property(prop)
            temp = self.df.loc[self.df.Property == prop]
            symbol_name = name.replace(':', '_')
            neurolangDL.add_tuple_set(((
                x,
                z,
            ) for x, y, z in temp.values),
                                      name=symbol_name)

        x1 = neurolangDL.new_symbol(name='x1')
        y1 = neurolangDL.new_symbol(name='y1')
        x2 = neurolangDL.new_symbol(name='x2')
        y2 = neurolangDL.new_symbol(name='y2')

        neurolangDL.symbols.rdf_schema_subPropertyOf[
            y1, y2] = neurolangDL.symbols.rdf_schema_subPropertyOf(
                x1, x2
            ) & neurolangDL.symbols.owl_inverseOf(
                y1, x1
            ) & neurolangDL.symbols.owl_inverseOf(y2, x2)
        neurolangDL.symbols.rdf_schema_subPropertyOf[
            x1, x1] = neurolangDL.symbols.rdf_syntax_ns_type(
                x1, 'http://www.w3.org/2002/07/owl#ObjectProperty'
            )
        neurolangDL.symbols.rdf_schema_subPropertyOf[
            y1, y2] = neurolangDL.symbols.rdf_schema_subPropertyOf(
                y1, x1
            ) & neurolangDL.symbols.rdf_schema_subPropertyOf(x1, y2)

        neurolangDL.symbols.rdf_schema_subClassOf[
            y1, y2] = neurolangDL.symbols.rdf_schema_subClassOf(
                x1, x2
            ) & neurolangDL.symbols.rdf_syntax_ns_rest(
                y1, x1
            ) & neurolangDL.symbols.rdf_syntax_ns_rest(y2, x2)
        neurolangDL.symbols.rdf_schema_subClassOf[
            x1, x1] = neurolangDL.symbols.rdf_syntax_ns_type(
                x1, 'http://www.w3.org/2002/07/owl#Class'
            )
        neurolangDL.symbols.rdf_schema_subClassOf[
            y1, y2] = neurolangDL.symbols.rdf_schema_subClassOf(
                y1, x1
            ) & neurolangDL.symbols.rdf_schema_subClassOf(x1, y2)

        neurolangDL.symbols.owl_disjointWith[
            y1, y2] = neurolangDL.symbols.owl_disjointWith(
                x1, x2
            ) & neurolangDL.symbols.rdf_schema_subClassOf(
                y1, x1
            ) & neurolangDL.symbols.rdf_schema_subClassOf(y2, x2)

        if destriuex_relations:
            relations_list = self.get_destrieux_relations()
            neurolangDL.add_tuple_set(((
                e1,
                e2,
            ) for e1, e2 in relations_list),
                                      name='relations')

            destrieux_dataset = datasets.fetch_atlas_destrieux_2009()
            destrieux_map = nib.load(destrieux_dataset['maps'])

            destrieux = []
            for label_number, name in destrieux_dataset['labels']:
                if label_number == 0:
                    continue
                name = name.decode()
                region = neurolangDL.create_region(
                    destrieux_map, label=label_number
                )
                if region is None:
                    continue
                name = name.replace('-', '_').replace(' ', '_').lower()
                destrieux.append((name, region))

            neurolangDL.add_tuple_set(((
                name,
                region,
            ) for name, region in destrieux),
                                      name='destrieux_regions')

        return neurolangDL

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
