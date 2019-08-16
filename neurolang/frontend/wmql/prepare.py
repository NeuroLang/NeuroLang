import nibabel as nib
import numpy as np
from tract_querier import tract_label_indices, tractography

from .. import ExplicitVBR
from ...solver_datalog_naive import Symbol


def prepare_datalog_ir_program(datalog, atlas_filename, tracts_filename):
    """
    Given a Datalog Program representation initalise the following EDB sets:
    * `tract_traversals(tract_id, region_id)`: of tract id and the atlas region
                              traversed by the tract, `tracts`
    * `tracts(tract_id, region)`: of tract id and the tracts as VBR regions.
    * `endpoints_in(tract_id, region)`: of tract id and the regions where its
                                        endpoints reach.

    :param datalog DatalogBasic: datalog IR instance to add the EDB elements
                                 to.
    :param atlas_filename str: filename for the atlas image.
    :param tracts_filename str: filename for the tracts.
    """

    tr = tractography.Tractography(tracts=[])

    desikan_map = nib.load(atlas_filename)
    tracts_trk = nib.trackvis.read(tracts_filename, points_space='rasmm')

    tr = tractography.Tractography(tracts=[t[0] for t in tracts_trk[0]])
    tli = tract_label_indices.TractographySpatialIndexing(
        tr.tracts(), desikan_map.get_data(), desikan_map.affine, 0., 2.
    )

    tract_traversals_ = []
    tracts_ = []
    for tract, labels in tli.crossing_tracts_labels.items():
        tract_region = ExplicitVBR(tli.tractography[tract], np.eye(4))
        tracts_.append((tract, tract_region))
        for label in labels:
            tract_traversals_.append((tract, label))

    datalog.add_extensional_predicate_from_tuples(
        Symbol('tract_traversals'), tract_traversals_
    )
    datalog.add_extensional_predicate_from_tuples(Symbol('tracts'), tracts_)

    endpoints_in = []
    for ending in tli.ending_tracts_labels:
        for tract, label in ending.items():
            endpoints_in.append((tract, label))
    datalog.add_extensional_predicate_from_tuples(
        Symbol('endpoints_in'), endpoints_in
    )

    return datalog
