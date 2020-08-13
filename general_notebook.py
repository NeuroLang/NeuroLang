#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import nibabel as nib

from neurolang import frontend as fe
from neurolang.logic.horn_clauses import Fol2DatalogMixin
from neurolang import regions
from nilearn import datasets
from statistics import mode

import general_notebook_queries as queries


# In[2]:


def create_frontend():
    class Solver(Fol2DatalogMixin, fe.RegionFrontendDatalogSolver):
        pass

    return fe.NeurolangDL(Solver())


# In[4]:


subject_ids = ["105923", "111514"]


# In[6]:


subj_files_path = "./"


# In[8]:


x_labels = ["medial", "during_x", "lateral"]
y_labels = ["anterior", "during_y", "posterior"]
z_labels = ["superior", "during_z", "inferior"]


# In[10]:


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


# In[12]:


def making_dominant_sets_relative_to_primary(
    info_dict, primary_sulcus, s, labels, nl, axis
):
    if info_dict[s]["primaries"][primary_sulcus] is not np.nan:
        x = nl.new_region_symbol("x")
        ps = info_dict[s]["primaries"][primary_sulcus]

        __import__("pdb").set_trace()

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

        anterior = set()
        posterior = set()
        during_y = set()
        superior = set()
        inferior = set()
        during_z = set()
        medial = set()
        lateral = set()
        during_x = set()

        for r in res:
            if r.symbol_name.startswith("L_"):
                if r.symbol_name in info_dict[s]["destrieux_sulci"].keys():
                    sulcus_relativity = voxel_relations_using_interval_algebra(
                        info_dict[s]["primaries"][primary_sulcus]
                        .value.to_xyz()
                        .T[axis],
                        info_dict[s]["destrieux_sulci"][r.symbol_name].T[axis],
                        length=0.1,
                    )
                    relations = []
                    relations.append(
                        labels[np.argmax(np.array(sulcus_relativity))]
                    )
                    if mode(relations) == "anterior":
                        anterior.add(nl.symbols[r.symbol_name])
                    elif mode(relations) == "posterior":
                        posterior.add(nl.symbols[r.symbol_name])
                    elif mode(relations) == "during_y":
                        during_y.add(nl.symbols[r.symbol_name])
                    elif mode(relations) == "superior":
                        superior.add(nl.symbols[r.symbol_name])
                    elif mode(relations) == "inferior":
                        inferior.add(nl.symbols[r.symbol_name])
                    elif mode(relations) == "during_z":
                        during_z.add(nl.symbols[r.symbol_name])
                    elif mode(relations) == "medial":
                        medial.add(nl.symbols[r.symbol_name])
                    elif mode(relations) == "lateral":
                        lateral.add(nl.symbols[r.symbol_name])
                    elif mode(relations) == "during_x":
                        during_x.add(nl.symbols[r.symbol_name])
                    else:
                        continue
                else:
                    pass

            if axis == 1:
                anterior_dominant = nl.add_region_set(
                    anterior, name=f"{primary_sulcus}_anterior_dominant"
                )
                posterior_dominant = nl.add_region_set(
                    posterior, name=f"{primary_sulcus}_posterior_dominant"
                )
                during_y_dominant = nl.add_region_set(
                    during_y, name=f"{primary_sulcus}_during_y_dominant"
                )
            elif axis == 2:
                superior_dominant = nl.add_region_set(
                    superior, name=f"{primary_sulcus}_superior_dominant"
                )
                inferior_dominant = nl.add_region_set(
                    inferior, name=f"{primary_sulcus}_inferior_dominant"
                )
                during_z_dominant = nl.add_region_set(
                    during_z, name=f"{primary_sulcus}_during_z_dominant"
                )
            elif axis == 0:
                medial_dominant = nl.add_region_set(
                    medial, name=f"{primary_sulcus}_medial_dominant"
                )
                lateral_dominant = nl.add_region_set(
                    lateral, name=f"{primary_sulcus}_lateral_dominant"
                )
                during_x_dominant = nl.add_region_set(
                    during_x, name=f"{primary_sulcus}_during_x_dominant"
                )
    else:
        pass


# In[15]:


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

    surface = nib.load(subj_files_path + f"{s}.L.pial.32k_fs_LR.surf.gii")
    destrieux_dataset = datasets.fetch_atlas_destrieux_2009(surface)
    destrieux_map = nib.load(destrieux_dataset["maps"])
    subject_destrieux_overlay = nib.load(
        subj_files_path + f"{s}.L.aparc.a2009s.32k_fs_LR.label.gii"
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

    is_more_lateral_than = nl.add_symbol(
        is_more_lateral_than_, name="is_more_lateral_than"
    )

    def is_more_medial_than_(x: regions.Region, y: regions.Region) -> bool:
        return bool(
            np.abs(np.average(x.to_xyz().T[0]))
            < np.abs(np.average(y.to_xyz().T[0]))
        )

    is_more_medial_than = nl.add_symbol(
        is_more_medial_than_, name="is_more_medial_than"
    )

    def is_more_anterior_than_(x: regions.Region, y: regions.Region) -> bool:
        return bool(
            np.abs(np.average(x.to_xyz().T[1]))
            < np.abs(np.average(y.to_xyz().T[1]))
        )

    is_more_anterior_than = nl.add_symbol(
        is_more_anterior_than_, name="is_more_anterior_than"
    )

    def is_more_posterior_than_(x: regions.Region, y: regions.Region) -> bool:
        return bool(
            np.abs(np.average(x.to_xyz().T[1]))
            > np.abs(np.average(x.to_xyz().T[1]))
        )

    is_more_posterior_than = nl.add_symbol(
        is_more_posterior_than_, name="is_more_posterior_than"
    )

    def is_more_inferior_than_(x: regions.Region, y: regions.Region) -> bool:
        return bool(
            np.abs(np.average(x.to_xyz().T[2]))
            < np.abs(np.average(y.to_xyz().T[2]))
        )

    is_more_inferior_than = nl.add_symbol(
        is_more_inferior_than_, name="is_more_inferior_than"
    )

    def is_more_superior_than_(x: regions.Region, y: regions.Region) -> bool:
        return bool(
            np.abs(np.average(x.to_xyz().T[2]))
            > np.abs(np.average(y.to_xyz().T[2]))
        )

    is_more_superior_than = nl.add_symbol(
        is_more_superior_than_, name="is_more_superior_than"
    )

    subject_info = d1
    subject_folds = (renamed_destrieux, s)
    return subject_info, subject_folds


# In[16]:


def process_NL(subject_folds, subject_info, nl):
    d_queries = []
    renamed_destrieux, s = subject_folds
    Primary_Sulci = set()
    names_of_primary_sulci = list(subject_info[s]["primaries"].keys())
    for prim in names_of_primary_sulci:
        Primary_Sulci.add(subject_info[s]["primaries"][prim])
    primary_sulci = nl.add_tuple_set(
        [(v.value,) for v in Primary_Sulci], name="primary_sulci"
    )

    for sulcus in (
        "Central_sulcus",
        "Lateral_fissure",
        "Parieto_occipital_sulcus",
        "Callosal_sulcus",
        "Calcarine_sulcus",
        "Anterior_horizontal_ramus_LF",
        "Anterior_vertical_ramus_LF",
    ):
        for labels, axis in ((x_labels, 0), (y_labels, 1), (z_labels, 2)):
            making_dominant_sets_relative_to_primary(
                subject_info, sulcus, s, labels, nl, axis
            )

    Found_sulci = set()
    found_sulci = nl.add_region_set(Found_sulci, name="found_sulci")

    Queries = [
        queries.Q_inferior_temporal,
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
                nl.add_region_set(Found_sulci, name="found_sulci")
                d_queries.append(
                    {
                        "subject": s,
                        "query": q.__name__,
                        "sulcus": r.symbol_name,
                    }
                )

    df_d_queries_s = pd.DataFrame(d_queries)
    df_d_queries_s.to_csv(f"NeuroLang_queries_LH")
    return d_queries


# In[17]:


def process_subjects(s):
    nl = create_frontend()
    d1, dNL = process_sulci(s, nl)
    d2 = process_NL(dNL, d1, nl)
    return d1, d2


# In[18]:


results = [process_subjects(i) for i in subject_ids]


# In[21]:


infos = []
for sid in subject_ids:
    i = subject_ids.index(sid)
    infos.append(results[i][0][sid])
df_infos = pd.DataFrame(data=infos, index=subject_ids)
df_infos.to_pickle("NeuroLang_queries_info_LH")


# In[ ]:
