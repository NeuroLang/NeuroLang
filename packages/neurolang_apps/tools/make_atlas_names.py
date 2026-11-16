"""
Make the tables of the names and the abbreviations of the regions of the
Destrieux, Schaefer and Tian atlases, which the engines join with their
labels (``engines/mni_atlases``, ``engines/connectivity_atlases``):

* ``destrieux_names.tsv``: the full name of each of the 148 Destrieux
  regions, from Table 1 of Destrieux C, Fischl B, Dale A, Halgren E,
  NeuroImage 53:1-15 (2010), doi:10.1016/j.neuroimage.2010.06.010
  ("ligual" read "lingual");
* ``tian_names.tsv``: of the 32 regions of Tian's subcortex, scale 2,
  from Table 2 of the atlas's nomenclature
  (Nomenclature-subcortex-parcellation-hierarchy-3T.pdf of
  https://github.com/yetianmed/subcortex);
* ``schaefer_names.tsv`` (100 parcels) and ``schaefer400_names.tsv`` (400),
  17 networks: the network and the component of each parcel, from the
  glossary of the CBIG repository (MIT): the networks of
  Yeo2011_17networks_N1000.split_components.glossary.csv
  (Yeo_JNeurophysiol11_SplitLabels), the components of the table "Parcel
  Name Abbreviations" of Schaefer2018_LocalGlobal/Parcellations/README.md;
  "parcel names are obtained by concatenating the number of networks,
  hemisphere, network name, component name, and parcel number";
* ``abbreviations.tsv``: ``atlas  abbreviation  meaning``, the
  abbreviations of the labels of the three atlases: Schaefer's of those
  sources, Tian's of its Table 2, Destrieux's read in the full names of its
  Table 1.

Each name ends with the hemisphere. The labels are those of the engines
(neurolang.atlases.label_name: '-' and '_' as spaces).

    python tools/make_atlas_names.py      # needs the internet (nilearn)
"""

from pathlib import Path

ENGINES = (Path(__file__).resolve().parents[1] / "neurolang_apps" / "utils"
           / "engines")
SIDES = (("L", "LH", "lh", "left"), ("R", "RH", "rh", "right"))

DESTRIEUX = [
    ('G_and_S_frontomargin', 'Fronto-marginal gyrus (of Wernicke) and sulcus'),
    ('G_and_S_occipital_inf', 'Inferior occipital gyrus (O3) and sulcus'),
    ('G_and_S_paracentral', 'Paracentral lobule and sulcus'),
    ('G_and_S_subcentral', 'Subcentral gyrus (central operculum) and sulci'),
    ('G_and_S_transv_frontopol', 'Transverse frontopolar gyri and sulci'),
    ('G_and_S_cingul-Ant', 'Anterior part of the cingulate gyrus and sulcus (ACC)'),
    ('G_and_S_cingul-Mid-Ant', 'Middle-anterior part of the cingulate gyrus and sulcus (aMCC)'),
    ('G_and_S_cingul-Mid-Post', 'Middle-posterior part of the cingulate gyrus and sulcus (pMCC)'),
    ('G_cingul-Post-dorsal', 'Posterior-dorsal part of the cingulate gyrus (dPCC)'),
    ('G_cingul-Post-ventral', 'Posterior-ventral part of the cingulate gyrus (vPCC, isthmus of the cingulate gyrus)'),
    ('G_cuneus', 'Cuneus (O6)'),
    ('G_front_inf-Opercular', 'Opercular part of the inferior frontal gyrus'),
    ('G_front_inf-Orbital', 'Orbital part of the inferior frontal gyrus'),
    ('G_front_inf-Triangul', 'Triangular part of the inferior frontal gyrus'),
    ('G_front_middle', 'Middle frontal gyrus (F2)'),
    ('G_front_sup', 'Superior frontal gyrus (F1)'),
    ('G_Ins_lg_and_S_cent_ins', 'Long insular gyrus and central sulcus of the insula'),
    ('G_insular_short', 'Short insular gyri'),
    ('G_occipital_middle', 'Middle occipital gyrus (O2, lateral occipital gyrus)'),
    ('G_occipital_sup', 'Superior occipital gyrus (O1)'),
    ('G_oc-temp_lat-fusifor', 'Lateral occipito-temporal gyrus (fusiform gyrus, O4-T4)'),
    ('G_oc-temp_med-Lingual', 'Lingual gyrus, lingual part of the medial occipito-temporal gyrus (O5)'),
    ('G_oc-temp_med-Parahip', 'Parahippocampal gyrus, parahippocampal part of the medial occipito-temporal gyrus (T5)'),
    ('G_orbital', 'Orbital gyri'),
    ('G_pariet_inf-Angular', 'Angular gyrus'),
    ('G_pariet_inf-Supramar', 'Supramarginal gyrus'),
    ('G_parietal_sup', 'Superior parietal lobule (lateral part of P1)'),
    ('G_postcentral', 'Postcentral gyrus'),
    ('G_precentral', 'Precentral gyrus'),
    ('G_precuneus', 'Precuneus (medial part of P1)'),
    ('G_rectus', 'Straight gyrus, Gyrus rectus'),
    ('G_subcallosal', 'Subcallosal area, subcallosal gyrus'),
    ('G_temp_sup-G_T_transv', 'Anterior transverse temporal gyrus (of Heschl)'),
    ('G_temp_sup-Lateral', 'Lateral aspect of the superior temporal gyrus'),
    ('G_temp_sup-Plan_polar', 'Planum polare of the superior temporal gyrus'),
    ('G_temp_sup-Plan_tempo', 'Planum temporale or temporal plane of the superior temporal gyrus'),
    ('G_temporal_inf', 'Inferior temporal gyrus (T3)'),
    ('G_temporal_middle', 'Middle temporal gyrus (T2)'),
    ('Lat_Fis-ant-Horizont', 'Horizontal ramus of the anterior segment of the lateral sulcus (or fissure)'),
    ('Lat_Fis-ant-Vertical', 'Vertical ramus of the anterior segment of the lateral sulcus (or fissure)'),
    ('Lat_Fis-post', 'Posterior ramus (or segment) of the lateral sulcus (or fissure)'),
    ('Pole_occipital', 'Occipital pole'),
    ('Pole_temporal', 'Temporal pole'),
    ('S_calcarine', 'Calcarine sulcus'),
    ('S_central', "Central sulcus (Rolando's fissure)"),
    ('S_cingul-Marginalis', 'Marginal branch (or part) of the cingulate sulcus'),
    ('S_circular_insula_ant', 'Anterior segment of the circular sulcus of the insula'),
    ('S_circular_insula_inf', 'Inferior segment of the circular sulcus of the insula'),
    ('S_circular_insula_sup', 'Superior segment of the circular sulcus of the insula'),
    ('S_collat_transv_ant', 'Anterior transverse collateral sulcus'),
    ('S_collat_transv_post', 'Posterior transverse collateral sulcus'),
    ('S_front_inf', 'Inferior frontal sulcus'),
    ('S_front_middle', 'Middle frontal sulcus'),
    ('S_front_sup', 'Superior frontal sulcus'),
    ('S_interm_prim-Jensen', 'Sulcus intermedius primus (of Jensen)'),
    ('S_intrapariet_and_P_trans', 'Intraparietal sulcus (interparietal sulcus) and transverse parietal sulci'),
    ('S_oc_middle_and_Lunatus', 'Middle occipital sulcus and lunatus sulcus'),
    ('S_oc_sup_and_transversal', 'Superior occipital sulcus and transverse occipital sulcus'),
    ('S_occipital_ant', 'Anterior occipital sulcus and preoccipital notch (temporo-occipital incisure)'),
    ('S_oc-temp_lat', 'Lateral occipito-temporal sulcus'),
    ('S_oc-temp_med_and_Lingual', 'Medial occipito-temporal sulcus (collateral sulcus) and lingual sulcus'),
    ('S_orbital_lateral', 'Lateral orbital sulcus'),
    ('S_orbital_med-olfact', 'Medial orbital sulcus (olfactory sulcus)'),
    ('S_orbital-H_Shaped', 'Orbital sulci (H-shaped sulci)'),
    ('S_parieto_occipital', 'Parieto-occipital sulcus (or fissure)'),
    ('S_pericallosal', 'Pericallosal sulcus (S of corpus callosum)'),
    ('S_postcentral', 'Postcentral sulcus'),
    ('S_precentral-inf-part', 'Inferior part of the precentral sulcus'),
    ('S_precentral-sup-part', 'Superior part of the precentral sulcus'),
    ('S_suborbital', 'Suborbital sulcus (sulcus rostrales, supraorbital sulcus)'),
    ('S_subparietal', 'Subparietal sulcus'),
    ('S_temporal_inf', 'Inferior temporal sulcus'),
    ('S_temporal_sup', 'Superior temporal sulcus (parallel sulcus)'),
    ('S_temporal_transverse', 'Transverse temporal sulcus'),
]

TIAN = [
    ('aHIP', 'Anterior hippocampus'),
    ('pHIP', 'Posterior hippocampus'),
    ('lAMY', 'Lateral amygdala'),
    ('mAMY', 'Medial amygdala'),
    ('THA-DP', 'Dorsoposterior thalamus'),
    ('THA-VP', 'Ventroposterior thalamus'),
    ('THA-VA', 'Ventroanterior thalamus'),
    ('THA-DA', 'Dorsoanterior thalamus'),
    ('NAc-shell', 'Nucleus accumbens, shell'),
    ('NAc-core', 'Nucleus accumbens, core'),
    ('pGP', 'Posterior globus pallidus'),
    ('aGP', 'Anterior globus pallidus'),
    ('aPUT', 'Anterior putamen'),
    ('pPUT', 'Posterior putamen'),
    ('aCAU', 'Anterior caudate'),
    ('pCAU', 'Posterior caudate'),
]

SCHAEFER_NETWORKS = {
    "VisCent": "Central visual", "VisPeri": "Peripheral visual",
    "SomMotA": "Somatomotor A", "SomMotB": "Somatomotor B",
    "DorsAttnA": "Dorsal attention A", "DorsAttnB": "Dorsal attention B",
    "SalVentAttnA": "Salience / ventral attention A",
    "SalVentAttnB": "Salience / ventral attention B",
    "LimbicA": "Limbic A", "LimbicB": "Limbic B",
    "ContA": "Control A", "ContB": "Control B", "ContC": "Control C",
    "DefaultA": "Default A", "DefaultB": "Default B", "DefaultC": "Default C",
    "TempPar": "Temporal parietal",
}

SCHAEFER_COMPONENTS = {
    "AntTemp": "anterior temporal", "Aud": "auditory", "Cent": "central",
    "Cinga": "cingulate anterior", "Cingm": "mid-cingulate",
    "Cingp": "cingulate posterior", "ExStr": "extrastriate cortex",
    "ExStrInf": "extra-striate inferior", "ExStrSup": "extra-striate superior",
    "FEF": "frontal eye fields", "FPole": "frontal pole",
    "FrMed": "frontal medial", "FrOper": "frontal operculum",
    "IFG": "inferior frontal gyrus", "Ins": "insula",
    "IPL": "inferior parietal lobule", "IPS": "intraparietal sulcus",
    "OFC": "orbital frontal cortex", "ParMed": "parietal medial",
    "ParOcc": "parietal occipital", "ParOper": "parietal operculum",
    "pCun": "precuneus", "pCunPCC": "precuneus posterior cingulate cortex",
    "PFCd": "dorsal prefrontal cortex", "PFCl": "lateral prefrontal cortex",
    "PFCld": "lateral dorsal prefrontal cortex",
    "PFClv": "lateral ventral prefrontal cortex",
    "PFCm": "medial prefrontal cortex",
    "PFCmp": "medial posterior prefrontal cortex",
    "PFCv": "ventral prefrontal cortex", "PHC": "parahippocampal cortex",
    "PostC": "post central", "PrC": "precentral", "PrCd": "precentral dorsal",
    "PrCv": "precentral ventral", "RSC": "retrosplenial cortex",
    "Rsp": "retrosplenial", "S2": "S2", "SPL": "superior parietal lobule",
    "ST": "superior temporal", "Striate": "striate cortex",
    "StriCal": "striate calcarine", "Temp": "temporal",
    "TempOcc": "temporal occipital", "TempPole": "temporal pole",
}

ABBREVIATIONS = [
    ("schaefer", "17Networks",
     "the 17-network parcellation of Yeo et al. (2011) the parcels are "
     "matched to"),
    ("schaefer", "LH", "left hemisphere"),
    ("schaefer", "RH", "right hemisphere"),
    *(("schaefer", a, f"{m} network") for a, m in SCHAEFER_NETWORKS.items()),
    *(("schaefer", a, m) for a, m in SCHAEFER_COMPONENTS.items()),
    ("destrieux", "L", "left hemisphere"),
    ("destrieux", "R", "right hemisphere"),
    *(("destrieux", a, m) for a, m in (
        ("G", "gyrus"), ("S", "sulcus"), ("ant", "anterior"),
        ("Ant", "anterior"), ("post", "posterior"), ("Post", "posterior"),
        ("inf", "inferior"), ("sup", "superior"), ("med", "medial"),
        ("lat", "lateral"), ("Lat", "lateral"), ("Mid", "middle"),
        ("cent", "central"), ("cingul", "cingulate"),
        ("collat", "collateral"), ("Fis", "fissure (Lat Fis: the lateral "
                                         "sulcus or fissure)"),
        ("front", "frontal"), ("frontomargin", "fronto-marginal"),
        ("frontopol", "frontopolar"), ("fusifor", "fusiform"),
        ("Horizont", "horizontal"), ("Ins", "insular"), ("ins", "insula"),
        ("interm", "intermedius (interm prim Jensen: sulcus intermedius "
                   "primus of Jensen)"),
        ("prim", "primus"), ("intrapariet", "intraparietal"),
        ("lg", "long"), ("oc", "occipital (oc temp: occipito-temporal)"),
        ("olfact", "olfactory"), ("P", "parietal (P trans: transverse "
                                      "parietal sulci)"),
        ("Parahip", "parahippocampal"), ("pariet", "parietal"),
        ("Plan", "planum"), ("tempo", "temporale (Plan tempo: planum "
                                      "temporale)"),
        ("Supramar", "supramarginal"), ("T", "temporal (G T transv: "
                                            "transverse temporal gyrus)"),
        ("temp", "temporal"), ("trans", "transverse"),
        ("transv", "transverse"), ("Triangul", "triangular"),
    )),
    ("tian", "lh", "left hemisphere"),
    ("tian", "rh", "right hemisphere"),
    *(("tian", a, m) for a, m in (
        ("HIP", "hippocampus"), ("aHIP", "anterior hippocampus"),
        ("pHIP", "posterior hippocampus"), ("AMY", "amygdala"),
        ("lAMY", "lateral amygdala"), ("mAMY", "medial amygdala"),
        ("THA", "thalamus"), ("DA", "dorsoanterior"), ("VA", "ventroanterior"),
        ("DP", "dorsoposterior"), ("VP", "ventroposterior"),
        ("NAc", "nucleus accumbens"), ("GP", "globus pallidus"),
        ("aGP", "anterior globus pallidus"),
        ("pGP", "posterior globus pallidus"), ("PUT", "putamen"),
        ("aPUT", "anterior putamen"), ("pPUT", "posterior putamen"),
        ("CAU", "caudate nucleus"), ("aCAU", "anterior caudate"),
        ("pCAU", "posterior caudate"),
    )),
]


def label(name: str) -> str:
    """The label of a region as the engines name it."""
    return name.replace("-", " ").replace("_", " ")


def write(path: Path, header: str, rows) -> None:
    path.write_text("\n".join([header, *("\t".join(r) for r in rows)])
                    + "\n")


def schaefer_names(n_rois: int) -> list:
    """The label and full name of each parcel of Schaefer's atlas of
    *n_rois* parcels, 17 networks: '17Networks LH DefaultA pCunPCC 1' is
    'Default A network, precuneus posterior cingulate cortex 1, left'."""
    from nilearn import datasets

    atlas = datasets.fetch_atlas_schaefer_2018(
        n_rois=n_rois, yeo_networks=17, resolution_mm=2, verbose=0)
    rows = []
    for raw in atlas["labels"]:
        raw = raw.decode() if isinstance(raw, bytes) else str(raw)
        if "Networks" not in raw:
            continue  # the background
        _, hemi, network, *component, number = raw.split("_")
        side = dict((h, s) for _, h, _, s in SIDES)[hemi]
        part = " ".join(SCHAEFER_COMPONENTS[c] for c in component)
        name = (f"{SCHAEFER_NETWORKS[network]} network, {part} {number}"
                if part else f"{SCHAEFER_NETWORKS[network]} network {number}")
        rows.append((label(raw), f"{name}, {side}"))
    return rows


def main() -> None:
    mni = ENGINES / "mni_atlases"
    write(mni / "destrieux_names.tsv", "label\tname",
          [(f"{d} {label(short)}", f"{name}, {side}")
           for d, _, _, side in SIDES for short, name in DESTRIEUX])
    write(mni / "tian_names.tsv", "label\tname",
          [(f"{label(short)} {t}", f"{name}, {side}")
           for _, _, t, side in SIDES for short, name in TIAN])
    write(mni / "schaefer_names.tsv", "label\tname", schaefer_names(100))
    write(ENGINES / "connectivity_atlases" / "schaefer400_names.tsv",
          "label\tname", schaefer_names(400))
    write(mni / "abbreviations.tsv", "atlas\tabbreviation\tmeaning",
          ABBREVIATIONS)


if __name__ == "__main__":
    main()
