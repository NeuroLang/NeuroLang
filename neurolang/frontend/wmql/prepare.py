import nibabel as nib
import numpy as np
from tract_querier import tract_label_indices, tractography

from ...regions import ExplicitVBR, PointSet
from ...solver_datalog_naive import Symbol


def prepare_datalog_ir_program(
    datalog, atlas_filename, tracts_filename,
    tracts_symbol_name='tracts',
    regions_symbol_name='regions',
    tract_traversals_symbol_name='tract_traversals',
    endpoints_in_symbol_name='endpoints_in',
):
    """
    Given a Datalog Program representation initialise the following EDB sets:
    * `tract_traversals(tract_id, region_id)`: of tract id and the atlas region
                              traversed by the tract, `tracts`
    * `tracts(tract_id, region)`: of tract id and the tracts as `PointSet`
                                  regions.
    * `regions(region_id, region)`: of region id and the template regions
                                    as `ExplicitVBR` regions.
    * `endpoints_in(tract_id, region_id)`: of tract id and the regions where
                                        its endpoints reach.

    :param datalog DatalogBasic: datalog IR instance to add the EDB elements
                                 to.
    :param atlas_filename str: filename for the atlas image.
    :param tracts_filename str: filename for the tracts.
    """

    tr = tractography.Tractography(tracts=[])

    atlas_map = nib.load(atlas_filename)
    atlas_im = atlas_map.get_data()
    atlas_affine = atlas_map.affine
    tracts_trk = nib.trackvis.read(tracts_filename, points_space='rasmm')

    tr = tractography.Tractography(tracts=[t[0] for t in tracts_trk[0]])
    tli = tract_label_indices.TractographySpatialIndexing(
        tr.tracts(), atlas_map.get_data(), atlas_map.affine, 0., 2.
    )

    tract_traversals_ = []
    tracts_ = []
    eye_4 = np.eye(4)
    for tract, labels in tli.crossing_tracts_labels.items():
        points = tli.tractography[tract]
        if len(points) == 0:
            continue
        tract_region = PointSet(points, eye_4)
        tracts_.append((tract, tract_region))
        for label in labels:
            tract_traversals_.append((tract, label))
    datalog.add_extensional_predicate_from_tuples(
        Symbol(tract_traversals_symbol_name),
        tract_traversals_
    )
    datalog.add_extensional_predicate_from_tuples(
        Symbol(tracts_symbol_name),
        tracts_
    )

    regions = []
    for label in tli.crossing_labels_tracts:
        region = ExplicitVBR(
            np.transpose((atlas_im == label).nonzero()),
            atlas_affine
        )
        regions.append((label, region))
    datalog.add_extensional_predicate_from_tuples(
        Symbol(regions_symbol_name),
        regions
    )

    endpoints_in = []
    for ending in tli.ending_tracts_labels:
        for tract, label in ending.items():
            endpoints_in.append((tract, label))
    datalog.add_extensional_predicate_from_tuples(
        Symbol(endpoints_in_symbol_name), endpoints_in
    )

    return datalog
