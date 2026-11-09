"""Download the public tractography atlases of F.-C. Yeh (DESIGN.md, 7.5).

Standard library only: an archive is downloaded once into a data
directory, its checksum checked, and only the files asked for are
extracted from it, next to it; both are reused by the calls after.

``hcp842``
    The HCP842 atlas (Yeh et al. 2018, NeuroImage 178:57-68), as curated
    by E. Garyfallidis for DIPY's RecoBundles ("Advanced Atlas of 80
    Bundles in MNI space", figshare 10.6084/m9.figshare.7375883, CC BY 4.0;
    cite Garyfallidis et al. 2017, NeuroImage 170:283-297, and Yeh et al.
    2018): 78 expert-labelled bundles (79 files, ``IF0F_R.trk`` being a
    copy of ``IFOF_R.trk``) and the whole-brain tractogram (144,678
    streamlines, 37 of one point), ``.trk`` in MNI152 (ICBM 2009a).
    320 MB.
``hcp1065``
    The HCP1065 atlas (Yeh 2022, Nat Commun 13:4933, CC BY-SA 4.0, with the
    acknowledgement of the Human Connectome Project): 87 bundles,
    ``.trk.gz`` in MNI152 (ICBM 2009a); no whole-brain tractogram. 588 MB.
"""

import gzip
import hashlib
import os
import shutil
import urllib.request
import zipfile
from pathlib import Path
from typing import Optional, Sequence, Union

__all__ = ["ATLASES", "DEFAULT_DATA_DIR", "fetch_tractogram"]

#: Where the archives are kept unless a ``data_dir`` is given.
DEFAULT_DATA_DIR = "~/.cache/nl_tracts"

HCP842_BUNDLES = {
    "AC": "anterior commissure",
    "AF_L": "arcuate fasciculus, left",
    "AF_R": "arcuate fasciculus, right",
    "AR_L": "acoustic radiation, left",
    "AR_R": "acoustic radiation, right",
    "AST_L": "frontal aslant tract, left",
    "AST_R": "frontal aslant tract, right",
    "C_L": "cingulum, left",
    "C_R": "cingulum, right",
    "CB_L": "cerebellum, left",
    "CB_R": "cerebellum, right",
    "CC": "corpus callosum",
    "CC_ForcepsMajor": "corpus callosum, forceps major",
    "CC_ForcepsMinor": "corpus callosum, forceps minor",
    "CCMid": "corpus callosum, body",
    "CNII_L": "optic nerve (II), left",
    "CNII_R": "optic nerve (II), right",
    "CNIII_L": "oculomotor nerve (III), left",
    "CNIII_R": "oculomotor nerve (III), right",
    "CNIV_L": "trochlear nerve (IV), left",
    "CNIV_R": "trochlear nerve (IV), right",
    "CNV_L": "trigeminal nerve (V), left",
    "CNV_R": "trigeminal nerve (V), right",
    "CNVII_L": "facial nerve (VII), left",
    "CNVII_R": "facial nerve (VII), right",
    "CNVIII_L": "vestibulocochlear nerve (VIII), left",
    "CNVIII_R": "vestibulocochlear nerve (VIII), right",
    "CS_L": "corticostriatal pathway, left",
    "CS_R": "corticostriatal pathway, right",
    "CST_L": "corticospinal tract, left",
    "CST_R": "corticospinal tract, right",
    "CT_L": "corticothalamic pathway, left",
    "CT_R": "corticothalamic pathway, right",
    "CTT_L": "central tegmental tract, left",
    "CTT_R": "central tegmental tract, right",
    "DLF_L": "dorsal longitudinal fasciculus, left",
    "DLF_R": "dorsal longitudinal fasciculus, right",
    "EMC_L": "extreme capsule, left",
    "EMC_R": "extreme capsule, right",
    "F_L_R": "fornix, both sides",
    "FPT_L": "frontopontine tract, left",
    "FPT_R": "frontopontine tract, right",
    "ICP_L": "inferior cerebellar peduncle, left",
    "ICP_R": "inferior cerebellar peduncle, right",
    "IFOF_L": "inferior fronto-occipital fasciculus, left",
    "IFOF_R": "inferior fronto-occipital fasciculus, right",
    "ILF_L": "inferior longitudinal fasciculus, left",
    "ILF_R": "inferior longitudinal fasciculus, right",
    "LL_L": "lateral lemniscus, left",
    "LL_R": "lateral lemniscus, right",
    "MCP": "middle cerebellar peduncle",
    "MdLF_L": "middle longitudinal fasciculus, left",
    "MdLF_R": "middle longitudinal fasciculus, right",
    "ML_L": "medial lemniscus, left",
    "ML_R": "medial lemniscus, right",
    "MLF_L": "medial longitudinal fasciculus, left",
    "MLF_R": "medial longitudinal fasciculus, right",
    "OPT_L": "occipitopontine tract, left",
    "OPT_R": "occipitopontine tract, right",
    "OR_L": "optic radiation, left",
    "OR_R": "optic radiation, right",
    "PC": "posterior commissure",
    "PPT_L": "parietopontine tract, left",
    "PPT_R": "parietopontine tract, right",
    "RST_L": "rubrospinal tract, left",
    "RST_R": "rubrospinal tract, right",
    "SCP": "superior cerebellar peduncle",
    "SLF_L": "superior longitudinal fasciculus, left",
    "SLF_R": "superior longitudinal fasciculus, right",
    "STT_L": "spinothalamic tract, left",
    "STT_R": "spinothalamic tract, right",
    "TPT_L": "temporopontine tract, left",
    "TPT_R": "temporopontine tract, right",
    "UF_L": "uncinate fasciculus, left",
    "UF_R": "uncinate fasciculus, right",
    "V": "vermis",
    "VOF_L": "vertical occipital fasciculus, left",
    "VOF_R": "vertical occipital fasciculus, right",
}


def _sides(names):
    return {
        f"{stem}_{s}": f"{name}, {side}"
        for stem, name in names.items()
        for s, side in (("L", "left"), ("R", "right"))
    }


HCP1065_BUNDLES = {
    **_sides({
        "AF": "arcuate fasciculus",
        "C_FPH": "cingulum, frontal parahippocampal",
        "C_FP": "cingulum, frontal parietal",
        "C_PHP": "cingulum, parahippocampal parietal",
        "C_PH": "cingulum, parahippocampal",
        "C_PO": "cingulum, parolfactory",
        "EMC": "extreme capsule",
        "FAT": "frontal aslant tract",
        "IFOF": "inferior fronto-occipital fasciculus",
        "ILF": "inferior longitudinal fasciculus",
        "MdLF": "middle longitudinal fasciculus",
        "PAT": "parietal aslant tract",
        "SLF1": "superior longitudinal fasciculus I",
        "SLF2": "superior longitudinal fasciculus II",
        "SLF3": "superior longitudinal fasciculus III",
        "UF": "uncinate fasciculus",
        "VOF": "vertical occipital fasciculus",
        "CB": "cerebellum",
        "ICP": "inferior cerebellar peduncle",
        "CNII": "optic nerve (II)",
        "CNIII": "oculomotor nerve (III)",
        "CNV": "trigeminal nerve (V)",
        "CNVII": "facial nerve (VII)",
        "CNVIII": "vestibulocochlear nerve (VIII)",
        "AR": "acoustic radiation",
        "CBT": "corticobulbar tract",
        "CPT_F": "corticopontine tract, frontal",
        "CPT_O": "corticopontine tract, occipital",
        "CPT_P": "corticopontine tract, parietal",
        "CST": "corticospinal tract",
        "CS_A": "corticostriatal tract, anterior",
        "CS_P": "corticostriatal tract, posterior",
        "CS_S": "corticostriatal tract, superior",
        "DRTT": "dentatorubrothalamic tract",
        "F": "fornix",
        "ML": "medial lemniscus",
        "OR": "optic radiation",
        "RST": "reticulospinal tract",
        "TR_A": "thalamic radiation, anterior",
        "TR_P": "thalamic radiation, posterior",
        "TR_S": "thalamic radiation, superior",
    }),
    "MCP": "middle cerebellar peduncle",
    "SCP": "superior cerebellar peduncle",
    "V": "vermis",
    "AC": "anterior commissure",
    "CC": "corpus callosum",
}

#: The atlases: the URLs of the archive (tried in turn), its md5, its file
#: name, the member of the whole-brain tractogram (or None), and the
#: bundles, by the stem of their file, with their full names.
ATLASES = {
    "hcp842": {
        "urls": (
            "https://ndownloader.figshare.com/files/13638644",
            # DIPY's mirror of the same file
            "https://workshop.dipy.org/services/data/files/13638644",
        ),
        "md5": "78331d527a10ec000d4f33bac472e099",
        "archive": "Atlas_80_Bundles.zip",
        "whole_brain": "Atlas_80_Bundles/whole_brain/whole_brain_MNI.trk",
        "bundles": HCP842_BUNDLES,
    },
    "hcp1065": {
        "urls": (
            "https://github.com/data-others/atlas/releases/download/"
            "hcp1065/hcp1065_avg_tracts_trk.zip",
        ),
        "md5": "0b4530ea43ea22394f3fecd301e0b7c4",
        "archive": "hcp1065_avg_tracts_trk.zip",
        "whole_brain": None,
        "bundles": HCP1065_BUNDLES,
    },
}


def _md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _download(urls: Sequence[str], md5: Optional[str], path: Path) -> None:
    """The first of ``urls`` that gives a file of checksum ``md5`` (any,
    if None), written to a temporary name and renamed to ``path``."""
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    errors = []
    try:
        for url in urls:
            try:
                with urllib.request.urlopen(url) as r, \
                        open(temporary, "wb") as f:
                    shutil.copyfileobj(r, f, 1 << 20)
            except OSError as e:
                errors.append(f"{url}: {e}")
                continue
            got = _md5(temporary)
            if md5 is None or got == md5:
                os.replace(temporary, path)
                return
            errors.append(f"{url}: md5 {got}, not {md5}")
    finally:
        if temporary.exists():
            temporary.unlink()
    raise OSError(f"cannot download {path.name}: " + "; ".join(errors))


def _extract(archive: Path, member: str, path: Path) -> None:
    """``member`` of the zip ``archive`` to ``path``, gunzipped if it ends
    in ``.gz``, through a temporary name."""
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with zipfile.ZipFile(archive) as z, z.open(member) as source, \
                open(temporary, "wb") as f:
            if member.endswith(".gz"):
                source = gzip.GzipFile(fileobj=source)
            shutil.copyfileobj(source, f, 1 << 20)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def fetch_tractogram(
    atlas: str = "hcp842",
    bundle: Optional[str] = None,
    data_dir: Optional[Union[str, os.PathLike]] = None,
    url: Optional[str] = None,
    md5: Optional[str] = None,
) -> Path:
    """The path of a ``.trk`` of an atlas of :data:`ATLASES`, downloaded
    and extracted if it is not in ``data_dir`` yet.

    Parameters
    ----------
    atlas : str
        ``hcp842`` or ``hcp1065``.
    bundle : str, optional
        The stem of a bundle (``"UF_L"``, any case; the keys of
        ``ATLASES[atlas]["bundles"]``), else the whole brain.
    data_dir : path, optional
        The archive is ``<data_dir>/<atlas>/<archive>`` and the files
        extracted from it are next to it: ``<stem>.trk`` (a ``.trk.gz`` of
        the archive is gunzipped) and the whole brain under its name.
        Default :data:`DEFAULT_DATA_DIR`.
    url : str, optional
        Where to download the archive from instead (a ``file://`` URL
        too); its checksum is then ``md5``.
    md5 : str, optional
        The checksum of the archive of ``url``; not checked if None.

    Raises
    ------
    ValueError
        An unknown atlas or bundle, or a whole brain the atlas has not.
    OSError
        If the archive cannot be downloaded, or its checksum differs.
    """
    if atlas not in ATLASES:
        raise ValueError(
            f"the atlases are {', '.join(ATLASES)}, not {atlas!r}")
    spec = ATLASES[atlas]
    if bundle is None:
        stem = None
        if spec["whole_brain"] is None:
            raise ValueError(
                f"{atlas} has no whole-brain tractogram: give a bundle")
    else:
        stems = {s.lower(): s for s in spec["bundles"]}
        stem = stems.get(str(bundle).lower())
        if stem is None:
            raise ValueError(
                f"{atlas} has no bundle {bundle!r}; its bundles are "
                f"{', '.join(spec['bundles'])}"
            )
    directory = Path(data_dir or DEFAULT_DATA_DIR).expanduser() / atlas
    archive = directory / spec["archive"]
    path = directory / (
        f"{stem}.trk" if stem else spec["whole_brain"].rsplit("/", 1)[-1])
    if path.is_file():
        return path
    if not archive.is_file():
        directory.mkdir(parents=True, exist_ok=True)
        _download(
            spec["urls"] if url is None else (url,),
            spec["md5"] if url is None else md5,
            archive,
        )
    with zipfile.ZipFile(archive) as z:
        member = spec["whole_brain"] if stem is None else next(
            (n for n in z.namelist()
             if n.rsplit("/", 1)[-1] in (f"{stem}.trk", f"{stem}.trk.gz")),
            None,
        )
    if member is None:
        raise OSError(f"{archive} has no bundle {stem}")
    _extract(archive, member, path)
    return path
