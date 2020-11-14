"""
Sulcal Identification Queries in Neurolang
==============================================

"""

from matplotlib import pyplot as plt
import nibabel as nib
from nilearn import datasets, plotting
from neurolang.frontend import NeurolangDL, ExplicitVBR


##################################################
# Initialise the NeuroLang probabilistic engine.

nl = NeurolangDL()


###############################################################################
# Load the Destrieux example from nilearn as a fact list


atlas_destrieux = datasets.fetch_atlas_destrieux_2009()
atlas_labels = {
    label: str(name.decode('utf8'))
    for label, name in atlas_destrieux['labels']
}


nl.add_atlas_set('destrieux_atlas', atlas_labels, nib.load(atlas_destrieux['maps']))

###############################################################################
# Add utility functions


@nl.add_symbol
def startswith(prefix: str, s: str) -> bool:
    return s.startswith(prefix)


@nl.add_symbol
def anterior_dominant(s: ExplicitVBR, r: ExplicitVBR) -> bool:
    s_xyz = s.to_xyz()
    r_xyz = r.to_xyz()
    res = (
        (s_xyz[:, 1] > r_xyz[:, 1].max()).sum() >
        max((
            (s_xyz[:, 0] > r_xyz[:, 0].max()).sum(),
            (s_xyz[:, 2] > r_xyz[:, 2].max()).sum()
        ) + tuple(
            (s_xyz[:, i] < r_xyz[:, i].min()).sum()
            for i in range(3)
        ))
    )
    return res


#############################################################
# Define all left sulci and the primary ones

with nl.environment as e:
    e.left_sulcus[e.name, e.region] = (
        e.destrieux_atlas(e.name, e.region) &
        startswith('L S', e.name)
    )

    e.left_primary_sulcus[e.name, e.region] = (
        e.destrieux_atlas(e.name, e.region) & (
            (e.name == "L S_central") |
            (e.name == "L Lat_Fis-post") |
            (e.name == "L S_pericallosal") |
            (e.name == "L S_parieto_occipital") |
            (e.name == "L S_calcarine") |
            (e.name == "L Lat_Fis-ant-Vertical")
        )
    )

    e.precentral_candidate[e.x] = (
        e.anterior_dominant(
            e.x, e.destrieux_atlas.s['L S_central']
        ) &
        e.during_x_dominant(
            e.x, e.destrieux_atlas.s['L S_central']
        )
    )


def Q_superior_frontal(nl):
    x = nl.new_region_symbol("x")
    y = nl.new_region_symbol("y")
    query1 = nl.query(
        x,
        (
            nl.symbols.isin(x, nl.symbols.Central_sulcus_anterior_dominant)
            & nl.symbols.isin(x, nl.symbols.Central_sulcus_during_x_dominant)
            & ~nl.symbols.isin(x, nl.symbols.Central_sulcus_posterior_dominant)
            & ~nl.symbols.isin(x, nl.symbols.found_sulci)
            & ~nl.symbols.isin(x, nl.symbols.primary_sulci)
        ),
    )

    q1 = query1.do()

    query2 = nl.query(
        x,
        q1(x)
        & ~nl.exists(
            y, (q1(y) & (x != y) & nl.symbols.is_more_superior_than(y, x))
        ),
    )
    return query2.do()