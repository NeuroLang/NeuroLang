"""
Per subject processing of the general notebook
==============================================

TODO: This needs a description
"""

import numpy as np
import pandas as pd
import nibabel as nib

from neurolang import frontend as fe
from neurolang.logic.horn_clauses import Fol2DatalogMixin
from neurolang import regions
from nilearn import datasets

# TODO: Fix this in a way that it works well in both 3.7 and 3.8
try:
    from statistics import multimode
except ImportError:
    from statistics import mode

    def multimode(data):
        return [mode(data)]


from neurolang.datalog.chase.negation import DatalogChaseNegation
from neurolang.datalog.negation import DatalogProgramNegationMixin
from neurolang.frontend.drs.translate_to_dl import CnlFrontendMixin


import general_notebook_queries as queries


###############################################################################
# Helper to create a frontend with the desired configuration
#


def create_frontend():
    class Program(
        Fol2DatalogMixin,
        DatalogProgramNegationMixin,
        fe.RegionFrontendDatalogSolver,
    ):
        pass

    class Chase(DatalogChaseNegation):
        pass

    class NeurolangFrontend(CnlFrontendMixin, fe.QueryBuilderDatalog):
        def __init__(self):
            super().__init__(
                Program(), chase_class=Chase,
            )

    return NeurolangFrontend()


###############################################################################
# Constant definitions
#

SUBJ_FILES_PATH = "./general_notebook_subjects/"

X_LABELS = ["medial", "during_x", "lateral"]
Y_LABELS = ["anterior", "during_y", "posterior"]
Z_LABELS = ["superior", "during_z", "inferior"]


###############################################################################
# Function to compare the relations between the bounding boxes
# of two sulcus
#


def voxel_relations_using_interval_algebra(
    ys_origin_sulcus, ys_target_sulcus, length
):
    before = set()
    during = set()
    after = set()
    J_minus = min(ys_target_sulcus)
    J_plus = max(ys_target_sulcus) + length

    I_boxes = set(ys_origin_sulcus)
    for x in I_boxes:
        I_minus = x
        I_plus = I_minus + length

        if I_minus < I_plus < J_minus < J_plus:
            before.add(x)
        if J_minus < I_minus < I_plus < J_plus:
            during.add(x)
        if J_minus < J_plus < I_minus < I_plus:
            after.add(x)
    before_pc = len(before) / len(I_boxes) * 100
    during_pc = len(during) / len(I_boxes) * 100
    after_pc = len(after) / len(I_boxes) * 100

    values = [before_pc, during_pc, after_pc]

    return values


###############################################################################
# Function to create a predicate for the dominant sets
#


def making_dominant_sets_relative_to_primary(
    info_dict, primary_sulcus, s, labels, nl, axis
):

    ps = info_dict[s]["primaries"][primary_sulcus]

    if ps is np.nan:
        # TODO: What implies that this is nan?
        raise Exception(f"{s} primaries {primary_sulcus} is np.nan")

    x = nl.new_region_symbol("x")
    res = nl.query(
        (x,),
        (
            nl.symbols.region(x)
            & (
                nl.symbols.anterior_of(x, ps)
                | nl.symbols.posterior_of(x, ps)
                | nl.symbols.superior_of(x, ps)
                | nl.symbols.inferior_of(x, ps)
            )
        ),
    )

    relation_dicts = {k: set() for k in (X_LABELS + Y_LABELS + Z_LABELS)}

    for (r,) in res:
        sulcus_relativity = voxel_relations_using_interval_algebra(
            ps.value.to_xyz().T[axis], r.to_xyz().T[axis], length=0.1,
        )
        relations = []
        relations.append(labels[np.argmax(np.array(sulcus_relativity))])
        modes = multimode(relations)
        for m in modes:
            relation_dicts[m].add(r)

    if axis == 1:
        for rel in Y_LABELS:
            nl.add_tuple_set(
                relation_dicts[rel],
                name=f"{primary_sulcus}_{rel}_dominant_contains",
            )
    elif axis == 2:
        for rel in Z_LABELS:
            nl.add_tuple_set(
                relation_dicts[rel],
                name=f"{primary_sulcus}_{rel}_dominant_contains",
            )
    elif axis == 0:
        for rel in X_LABELS:
            nl.add_tuple_set(
                relation_dicts[rel],
                name=f"{primary_sulcus}_{rel}_dominant_contains",
            )


###############################################################################
# Load and process the sulci of a subject
#


def process_sulci(s, nl):
    d1 = {}
    d1[s] = {}
    d1[s]["destrieux_sulci"] = {}
    d1[s]["destrieux_affines"] = {}
    d1[s]["destrieux_spatial_images"] = {}
    d1[s]["primaries"] = {}

    spatial_images_destrieux = {}

    destrieux_dataset = datasets.fetch_atlas_destrieux_2009()
    destrieux_map = nib.load(destrieux_dataset["maps"])

    surface = nib.load(SUBJ_FILES_PATH + f"{s}.L.pial.32k_fs_LR.surf.gii")
    destrieux_dataset = datasets.fetch_atlas_destrieux_2009(surface)
    destrieux_map = nib.load(destrieux_dataset["maps"])
    subject_destrieux_overlay = nib.load(
        SUBJ_FILES_PATH + f"{s}.L.aparc.a2009s.32k_fs_LR.label.gii"
    )

    renamed_destrieux = {}
    destrieux_affines = {}

    for label_number, name in destrieux_dataset["labels"]:
        name = name.decode()
        if (
            not name.startswith("L ")
            or ("S_" not in name and "Lat_Fis" not in name)
            or ("G_" in name)
        ):
            continue
        labels_on_surface = subject_destrieux_overlay.darrays[0].data
        points_in_region = surface.darrays[0].data[
            labels_on_surface == label_number
        ]
        ijk_points = nib.affines.apply_affine(
            np.linalg.inv(destrieux_map.affine), points_in_region
        ).astype(int)
        region = fe.ExplicitVBR(
            ijk_points, destrieux_map.affine, image_dim=destrieux_map.shape
        )

        name_fixed = "L_" + name[2:].replace("-", "_")

        destrieux_affines[name_fixed] = region
        nl.add_region(region, name=name_fixed)
        renamed_destrieux[name_fixed] = nl.symbols[name_fixed].value
        spatial_images_destrieux[name_fixed] = nl.symbols[
            name_fixed
        ].value.spatial_image()

    nl.add_tuple_set(
        [
            (v.value,)
            for k, v in nl.symbol_table.symbols_by_type(regions.Region).items()
        ],
        name="region",
    )

    nl.add_tuple_set(
        [(v, k) for k, v in renamed_destrieux.items()], name="is_named",
    )

    d1[s]["destrieux_sulci"] = renamed_destrieux
    d1[s]["destrieux_affines"] = destrieux_affines
    d1[s]["destrieux_spatial_images"] = spatial_images_destrieux
    d1[s]["primaries"]["Central_sulcus"] = nl.symbols.L_S_central
    d1[s]["primaries"]["Lateral_fissure"] = nl.symbols.L_Lat_Fis_post
    d1[s]["primaries"]["Callosal_sulcus"] = nl.symbols.L_S_pericallosal
    d1[s]["primaries"][
        "Parieto_occipital_sulcus"
    ] = nl.symbols.L_S_parieto_occipital
    d1[s]["primaries"]["Calcarine_sulcus"] = nl.symbols.L_S_calcarine
    d1[s]["primaries"][
        "Anterior_horizontal_ramus_LF"
    ] = nl.symbols.L_Lat_Fis_ant_Horizont
    d1[s]["primaries"][
        "Anterior_vertical_ramus_LF"
    ] = nl.symbols.L_Lat_Fis_ant_Vertical

    def is_more_lateral_than_(x: regions.Region, y: regions.Region) -> bool:
        return bool(
            np.abs(np.average(x.to_xyz().T[0]))
            > np.abs(np.average(y.to_xyz().T[0]))
        )

    nl.add_symbol(is_more_lateral_than_, name="is_more_lateral_than")

    def is_more_medial_than_(x: regions.Region, y: regions.Region) -> bool:
        return bool(
            np.abs(np.average(x.to_xyz().T[0]))
            < np.abs(np.average(y.to_xyz().T[0]))
        )

    nl.add_symbol(is_more_medial_than_, name="is_more_medial_than")

    def is_more_anterior_than_(x: regions.Region, y: regions.Region) -> bool:
        return bool(
            np.abs(np.average(x.to_xyz().T[1]))
            < np.abs(np.average(y.to_xyz().T[1]))
        )

    nl.add_symbol(is_more_anterior_than_, name="is_more_anterior_than")

    def is_more_posterior_than_(x: regions.Region, y: regions.Region) -> bool:
        return bool(
            np.abs(np.average(x.to_xyz().T[1]))
            > np.abs(np.average(x.to_xyz().T[1]))
        )

    nl.add_symbol(is_more_posterior_than_, name="is_more_posterior_than")

    def is_more_inferior_than_(x: regions.Region, y: regions.Region) -> bool:
        return bool(
            np.abs(np.average(x.to_xyz().T[2]))
            < np.abs(np.average(y.to_xyz().T[2]))
        )

    nl.add_symbol(is_more_inferior_than_, name="is_more_inferior_than")

    def is_more_superior_than_(x: regions.Region, y: regions.Region) -> bool:
        return bool(
            np.abs(np.average(x.to_xyz().T[2]))
            > np.abs(np.average(y.to_xyz().T[2]))
        )

    nl.add_symbol(is_more_superior_than_, name="is_more_superior_than")

    subject_info = d1
    subject_folds = (renamed_destrieux, s)
    return subject_info, subject_folds


###############################################################################
# Process the given sulci with the defined queries
#


def process_NL(subject_folds, subject_info, nl):
    d_queries = []
    renamed_destrieux, s = subject_folds
    Primary_Sulci = set()
    names_of_primary_sulci = list(subject_info[s]["primaries"].keys())
    for prim in names_of_primary_sulci:
        Primary_Sulci.add(subject_info[s]["primaries"][prim])
    nl.add_tuple_set([(v.value,) for v in Primary_Sulci], name="primary_sulci")

    for sulcus in (
        "Central_sulcus",
        "Lateral_fissure",
        "Parieto_occipital_sulcus",
        "Callosal_sulcus",
        "Calcarine_sulcus",
        "Anterior_horizontal_ramus_LF",
        "Anterior_vertical_ramus_LF",
    ):
        for labels, axis in ((X_LABELS, 0), (Y_LABELS, 1), (Z_LABELS, 2)):
            making_dominant_sets_relative_to_primary(
                subject_info, sulcus, s, labels, nl, axis
            )

    Found_sulci = set()
    nl.add_tuple_set(Found_sulci, name="found_sulci")

    Queries = [
        queries.Q_cingulate_cnl,
    ]

    for q in Queries:
        res = q(nl)
        if len(res) == 0:
            d_queries.append(
                {
                    "subject": s,
                    "query": q.__name__,
                    "sulcus": "No sulcus found",
                }
            )
        else:
            for r in res:
                Found_sulci.add(r)
                name = None
                for k, v in renamed_destrieux.items():
                    if v == r[0]:
                        name = k
                        break
                else:
                    raise Exception("No name for region")
                nl.add_tuple_set(Found_sulci, name="found_sulci")
                d_queries.append(
                    {
                        "subject": s,
                        "query": q.__name__,
                        "sulcus": name,
                        # "region": r[0],
                    }
                )

    df_d_queries_s = pd.DataFrame(d_queries)
    df_d_queries_s.to_csv(f"NeuroLang_queries_LH")
    return d_queries


###############################################################################
# Process a subject loading its sulci and then running the queries
#


def process_subject(s):
    nl = create_frontend()
    d1, dNL = process_sulci(s, nl)
    d2 = process_NL(dNL, d1, nl)
    return d1, d2
