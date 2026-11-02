"""Write the shipped term mappings ``nl_tracts/mappings/freesurfer_*.tsv``.

The vocabulary is tract_querier's, ``data/freesurfer_queries.qry``: the
Desikan names in lower case (``precentral``, ``temporalpole``), and the
groups that file defines as unions of them (``frontal``, ``temporal``,
``parietal``, ``occipital``, ``cingular``, ``orbitofrontalgyrus``,
``middle_frontal_gyrus``, ``inferior_frontal_gyrus``,
``superior_parietal_lobule``, ``inferior_parietal_lobule``, ``subcortical``,
``cerebellum``, ``ventricles``, ``corpus_callosum``). The groups are copied
as the file has them, including what looks odd (``subcortical`` has no
caudate, hippocampus or accumbens), so that rules translated from
tract_querier's queries select the same labels.

The Destrieux assignments are a draft by the author of the package, to be
reviewed by an anatomist: the Destrieux parcellation has no one-to-one
Desikan equivalent, so a label gets the Desikan name only where it is the
obvious one and a lobe otherwise. Labels on the boundary of two lobes
(central sulcus, lateral fissure, parieto-occipital sulcus), the medial wall
and ``Unknown`` get no term, on purpose: ``Atlas.unmapped_labels`` reports
them.

Usage: ``python tools/make_freesurfer_mappings.py [output-directory]``
(default ``nl_tracts/mappings``, next to this directory).
"""

import sys
from pathlib import Path

HEADER = "atlas_label_name\tterm\themi\n"

# --- Desikan (aparc): name -> groups, from freesurfer_queries.qry ---------
GROUPS = {
    "middle_frontal_gyrus": "rostralmiddlefrontal caudalmiddlefrontal",
    "inferior_frontal_gyrus": "parsopercularis parstriangularis parsorbitalis",
    "orbitofrontalgyrus": "lateralorbitofrontal medialorbitofrontal",
    "cingular": (
        "caudalanteriorcingulate isthmuscingulate posteriorcingulate "
        "rostralanteriorcingulate"
    ),
    "temporal": (
        "entorhinal parahippocampal temporalpole fusiform superiortemporal "
        "middletemporal inferiortemporal transversetemporal bankssts"
    ),
    "parietal": "postcentral precuneus supramarginal superiorparietal "
                "inferiorparietal",
    "superior_parietal_lobule": "postcentral precuneus superiorparietal",
    "inferior_parietal_lobule": "supramarginal inferiorparietal",
    "occipital": "lingual pericalcarine cuneus lateraloccipital",
}
# frontal is the union of other groups and of these
GROUPS["frontal"] = (
    "superiorfrontal frontalpole precentral paracentral "
    + GROUPS["middle_frontal_gyrus"] + " "
    + GROUPS["inferior_frontal_gyrus"] + " "
    + GROUPS["orbitofrontalgyrus"]
)
DESIKAN = (
    "bankssts caudalanteriorcingulate caudalmiddlefrontal cuneus entorhinal "
    "fusiform inferiorparietal inferiortemporal isthmuscingulate "
    "lateraloccipital lateralorbitofrontal lingual medialorbitofrontal "
    "middletemporal parahippocampal paracentral parsopercularis "
    "parsorbitalis parstriangularis pericalcarine postcentral "
    "posteriorcingulate precentral precuneus rostralanteriorcingulate "
    "rostralmiddlefrontal superiorfrontal superiorparietal superiortemporal "
    "supramarginal frontalpole temporalpole transversetemporal insula"
).split()


def desikan_terms(name):
    return [name] + [g for g, ms in GROUPS.items() if name in ms.split()]


# --- aseg: (raw name stem with {H} for Left/Right, terms) ------------------
SUBCORTICAL = "subcortical"
ASEG_SIDED = {  # stem -> terms; Left-<stem> / Right-<stem>
    "Cerebral-White-Matter": ["cerebral_white_matter"],
    "Cerebral-Cortex": ["cerebral_cortex"],
    "Lateral-Ventricle": ["lateral_ventricle", "ventricles"],
    "Inf-Lat-Vent": ["inf_lat_vent"],
    "Cerebellum-White-Matter": ["cerebellum_white_matter", "cerebellum"],
    "Cerebellum-Cortex": ["cerebellum_cortex", "cerebellum"],
    "Thalamus": ["thalamus_proper", SUBCORTICAL],
    "Thalamus-Proper": ["thalamus_proper", SUBCORTICAL],
    "Caudate": ["caudate"],
    "Putamen": ["putamen", SUBCORTICAL],
    "Pallidum": ["pallidum", SUBCORTICAL],
    "Hippocampus": ["hippocampus"],
    "Amygdala": ["amygdala", SUBCORTICAL],
    "Accumbens-area": ["accumbens_area"],
    "VentralDC": ["ventraldc", SUBCORTICAL],
    "vessel": ["vessel"],
    "choroid-plexus": ["choroid_plexus", SUBCORTICAL, "ventricles"],
    "WM-hypointensities": ["wm_hypointensities"],
    "non-WM-hypointensities": ["non_wm_hypointensities"],
}
ASEG_MIDLINE = {
    "3rd-Ventricle": ["third_ventricle", "ventricles"],
    "4th-Ventricle": ["fourth_ventricle", "ventricles"],
    "5th-Ventricle": ["fifth_ventricle", "ventricles"],
    "Brain-Stem": ["brain_stem", SUBCORTICAL],
    "CSF": ["csf"],
    "Optic-Chiasm": ["optic_chiasm"],
    "WM-hypointensities": ["wm_hypointensities"],
    "non-WM-hypointensities": ["non_wm_hypointensities"],
    **{
        f"CC_{p}": ["corpus_callosum"]
        for p in ("Posterior", "Mid_Posterior", "Central", "Mid_Anterior",
                  "Anterior")
    },
}


def aseg_aparc_rows():
    rows = []
    for stem, terms in ASEG_SIDED.items():
        for prefix, hemi in (("Left", "left"), ("Right", "right")):
            rows += [(f"{prefix}-{stem}", t, hemi) for t in terms]
    for name, terms in ASEG_MIDLINE.items():
        rows += [(name, t, "none") for t in terms]
    for tag, hemi in (("lh", "left"), ("rh", "right")):
        for name in DESIKAN:
            rows += [
                (f"ctx-{tag}-{name}", t, hemi)
                for t in desikan_terms(name)
            ]
        rows += [(f"ctx-{tag}-corpuscallosum", t, hemi)
                 for t in ("corpuscallosum", "cingular")]
    return rows


# --- Destrieux (a2009s): stem -> terms (draft, see the docstring) ----------
F, T, P, O, C, INS = (
    "frontal", "temporal", "parietal", "occipital", "cingular", "insula",
)
IFG, MFG, OFG = (
    "inferior_frontal_gyrus", "middle_frontal_gyrus", "orbitofrontalgyrus",
)
SPL, IPL = "superior_parietal_lobule", "inferior_parietal_lobule"
DESTRIEUX = {
    "G_and_S_frontomargin": [F, "frontalpole"],
    "G_and_S_occipital_inf": [O],
    "G_and_S_paracentral": [F, "paracentral"],
    "G_and_S_subcentral": [F],
    "G_and_S_transv_frontopol": [F, "frontalpole"],
    "G_and_S_cingul-Ant": [C, "rostralanteriorcingulate"],
    "G_and_S_cingul-Mid-Ant": [C, "caudalanteriorcingulate"],
    "G_and_S_cingul-Mid-Post": [C, "posteriorcingulate"],
    "G_cingul-Post-dorsal": [C, "posteriorcingulate"],
    "G_cingul-Post-ventral": [C, "isthmuscingulate"],
    "G_cuneus": [O, "cuneus"],
    "G_front_inf-Opercular": [F, IFG, "parsopercularis"],
    "G_front_inf-Orbital": [F, IFG, "parsorbitalis"],
    "G_front_inf-Triangul": [F, IFG, "parstriangularis"],
    "G_front_middle": [F, MFG],
    "G_front_sup": [F, "superiorfrontal"],
    "G_Ins_lg_and_S_cent_ins": [INS],
    "G_insular_short": [INS],
    "G_occipital_middle": [O, "lateraloccipital"],
    "G_occipital_sup": [O, "lateraloccipital"],
    "G_oc-temp_lat-fusifor": [T, "fusiform"],
    "G_oc-temp_med-Lingual": [O, "lingual"],
    "G_oc-temp_med-Parahip": [T, "parahippocampal"],
    "G_orbital": [F, OFG],
    "G_pariet_inf-Angular": [P, IPL, "inferiorparietal"],
    "G_pariet_inf-Supramar": [P, IPL, "supramarginal"],
    "G_parietal_sup": [P, SPL, "superiorparietal"],
    "G_postcentral": [P, SPL, "postcentral"],
    "G_precentral": [F, "precentral"],
    "G_precuneus": [P, SPL, "precuneus"],
    "G_rectus": [F, OFG, "medialorbitofrontal"],
    "G_subcallosal": [F, OFG, "medialorbitofrontal"],
    "G_temp_sup-G_T_transv": [T, "transversetemporal", "superiortemporal"],
    "G_temp_sup-Lateral": [T, "superiortemporal"],
    "G_temp_sup-Plan_polar": [T, "superiortemporal"],
    "G_temp_sup-Plan_tempo": [T, "superiortemporal"],
    "G_temporal_inf": [T, "inferiortemporal"],
    "G_temporal_middle": [T, "middletemporal"],
    "Pole_occipital": [O],
    "Pole_temporal": [T, "temporalpole"],
    "S_calcarine": [O, "pericalcarine"],
    "S_cingul-Marginalis": [C],
    "S_circular_insula_ant": [INS],
    "S_circular_insula_inf": [INS],
    "S_circular_insula_sup": [INS],
    "S_collat_transv_ant": [T],
    "S_collat_transv_post": [O],
    "S_front_inf": [F],
    "S_front_middle": [F],
    "S_front_sup": [F],
    "S_interm_prim-Jensen": [P],
    "S_intrapariet_and_P_trans": [P],
    "S_oc_middle_and_Lunatus": [O],
    "S_oc_sup_and_transversal": [O],
    "S_occipital_ant": [O],
    "S_oc-temp_lat": [T],
    "S_oc-temp_med_and_Lingual": [O],
    "S_orbital_lateral": [F, OFG],
    "S_orbital_med-olfact": [F, OFG],
    "S_orbital-H_Shaped": [F, OFG],
    "S_pericallosal": [C],
    "S_postcentral": [P],
    "S_precentral-inf-part": [F],
    "S_precentral-sup-part": [F],
    "S_suborbital": [F],
    "S_subparietal": [P],
    "S_temporal_inf": [T],
    "S_temporal_sup": [T],
    "S_temporal_transverse": [T],
}
# no term, on purpose (boundaries, medial wall): see the docstring
DESTRIEUX_UNMAPPED = (
    "Lat_Fis-ant-Horizont", "Lat_Fis-ant-Vertical", "Lat_Fis-post",
    "S_central", "S_parieto_occipital", "Medial_wall", "Unknown",
)


def destrieux_rows():
    """FreeSurfer's colour LUT (``ctx_lh_<stem>``) and nilearn's
    lateralised atlas (``L <stem>``), both spellings."""
    rows = []
    for stem, terms in DESTRIEUX.items():
        for fs, nilearn, hemi in (
            ("ctx_lh_", "L ", "left"), ("ctx_rh_", "R ", "right")
        ):
            for prefix in (fs, nilearn):
                rows += [(prefix + stem, t, hemi) for t in terms]
    return rows


def write(path, rows, comment):
    lines = [f"# {c}\n" for c in comment.strip().splitlines()]
    with open(path, "w", encoding="utf-8") as f:
        f.writelines(lines)
        f.write(HEADER)
        f.writelines(f"{n}\t{t}\t{h}\n" for n, t, h in rows)


def main(out):
    out = Path(out)
    write(
        out / "freesurfer_aseg_aparc.tsv",
        aseg_aparc_rows(),
        "FreeSurfer aseg and aparc (Desikan-Killiany), raw LUT names.\n"
        "Vocabulary and groups: tract_querier data/freesurfer_queries.qry.\n"
        "Generated by tools/make_freesurfer_mappings.py; edit that.",
    )
    write(
        out / "freesurfer_destrieux.tsv",
        destrieux_rows(),
        "FreeSurfer a2009s (Destrieux): ctx_lh_/ctx_rh_ names of the colour\n"
        "LUT and L /R  names of nilearn's lateralised atlas. A draft to be\n"
        "reviewed: no term for the lateral fissure, central sulcus,\n"
        "parieto-occipital sulcus, medial wall and Unknown.\n"
        "Generated by tools/make_freesurfer_mappings.py; edit that.",
    )


if __name__ == "__main__":
    main(
        sys.argv[1] if len(sys.argv) > 1
        else Path(__file__).resolve().parent.parent / "nl_tracts" / "mappings"
    )
