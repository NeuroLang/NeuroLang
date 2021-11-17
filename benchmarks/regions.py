from itertools import product
import nibabel
import nilearn
import numpy as np
import time

from neurolang import regions
from neurolang.CD_relations import (
    cardinal_relation,
    cardinal_relation_fast,
)


class TimeDestrieuxRegions:

    params = [
        [cardinal_relation, cardinal_relation_fast],
    ]

    param_names = [
        "CR method",
    ]

    """
    Ten slowest destrieux regions overlap comparisons:
        ('R G_cuneus', 'R G_occipital_sup') 0.4985234520000006
        ('R G_and_S_frontomargin', 'R G_front_sup') 0.4842066729999992
        ('L G_and_S_frontomargin', 'L G_front_sup') 0.45386739899999995
        ('R G_and_S_frontomargin', 'R G_front_middle') 0.35684133700000054
        ('L G_and_S_paracentral', 'L G_precentral') 0.32330401400000053
        ('R G_and_S_occipital_inf', 'R S_temporal_sup') 0.3231391549999998
        ('R G_and_S_occipital_inf', 'R G_occipital_middle') 0.299535895
        ('R G_and_S_cingul-Mid-Post', 'R G_and_S_paracentral') 0.2917201779999985
        ('R G_cingul-Post-ventral', 'R G_oc-temp_med-Parahip') 0.2848894589999986
        ('R G_occipital_middle', 'R G_pariet_inf-Angular') 0.28115004900000073

    Ten slowest destrieux regions to find an overlap with a sphere of radius 2 in the center of the region:
        L G_front_sup 0.4760005439999997
        R G_front_sup 0.4160161329999994
        L G_precentral 0.26453238700000004
        L G_front_middle 0.23240793400000026
        R G_front_middle 0.22466390900000022
        R S_temporal_sup 0.21848125299999666
        R G_precentral 0.20742356899999947
        R G_temporal_middle 0.19936352000000035
        R G_and_S_cingul-Ant 0.19587059100000026
        R G_orbital 0.19517688900000074
    """

    timeout = 3 * 60

    def setup(self, cr_method):
        self.regions = self.load_destrieux()

    def load_destrieux(self):
        atlas_destrieux = nilearn.datasets.fetch_atlas_destrieux_2009()

        image = nibabel.load(atlas_destrieux["maps"])
        image_data = image.get_data()

        region_dict = {}
        for label, name in atlas_destrieux["labels"]:
            if label == 0:
                continue

            voxels = np.transpose((image_data == label).nonzero())
            if voxels.shape[0] == 0:
                continue

            r = regions.ExplicitVBR(
                voxels, image.affine, image_dim=image.shape
            )
            region_dict[name.decode("utf8")] = r
        return region_dict

    def check_regions_overlap(self):
        """
        Compare results of cardinal_relation and cardinal_relation_fast
        methods.
        """
        ok, nok = 0, 0
        checked = set()
        for n0, n1 in product(self.regions.keys(), self.regions.keys()):
            key = " ".join(sorted((n0, n1)))
            if key not in checked:
                checked.add(key)
                overlap = cardinal_relation(
                    self.regions[n0],
                    self.regions[n1],
                    "O",
                    refine_overlapping=True,
                )
                overlap2 = cardinal_relation_fast(
                    self.regions[n0],
                    self.regions[n1],
                    "O",
                    refine_overlapping=True,
                )
                if overlap == overlap2:
                    ok += 1
                else:
                    print(n0, n1, overlap, overlap2)
                    nok += 1

        print("total ok ", ok)
        print("total nok ", nok)

    def time_regions_convex(self, cr_method):
        """
        Time overlaping between convex regions and sphere that don't
        overlap.
        """
        name = "L G_parietal_sup"
        r0 = self.regions[name]
        r1 = regions.SphericalVolume((-26, -52, 58), 2)
        is_overlaping = cr_method(
            r1, r0, "O", refine_overlapping=True
        )
        print(f"Convex region {name} is overlaping: ", is_overlaping)

        name = "R S_pericallosal"
        r0 = self.regions[name]
        r1 = r1 = regions.SphericalVolume((5, 3, 16), 9)
        is_overlaping = cr_method(
            r1, r0, "O", refine_overlapping=True
        )
        print(f"Convex region {name} is overlaping: ", is_overlaping)

    def time_regions_overlap(self, cr_method):
        """
        Check whether regions are overlaping for all region combinations
        in destrieux atlas. Prints the 100 slowest times.

        Parameters
        ----------
        cr_method : function
            the cardinal_relation method to test
        """
        times = dict()
        for r0, r1 in product(self.regions.keys(), self.regions.keys()):
            comb = tuple(sorted((r0, r1)))
            if comb not in times:
                start = time.perf_counter()
                is_overlaping = cr_method(
                    self.regions[r0],
                    self.regions[r1],
                    "O",
                    refine_overlapping=True,
                )
                stop = time.perf_counter()
                times[comb] = stop - start
        # print slowest 100
        print("100 slowest times in s for region / region overlaping")
        for k, v in sorted(
            times.items(), key=lambda item: item[1], reverse=True
        )[:100]:
            print(k, v)

    def time_regions_sphere_overlap(self, cr_method):
        """
        For all regions in destrieux's atlas, check whether they overlap
        with a 2mm radius sphere in the center of the region.

        Parameters
        ----------
        cr_method : function
            the cr_method to test
        """
        times = dict()
        for n, r in self.regions.items():
            if not isinstance(r, regions.EmptyRegion):
                start = time.perf_counter()
                image = r.spatial_image()
                coords = (
                    np.transpose(image.get_fdata().nonzero())
                    .mean(0)
                    .astype(int)
                )
                coords = nibabel.affines.apply_affine(image.affine, coords)
                center = [int(c) for c in coords]
                sphere = regions.SphericalVolume(center, 2)
                is_overlaping = cr_method(
                    r, sphere, "O", refine_overlapping=True
                )
                stop = time.perf_counter()
                times[n] = stop - start
        # print slowest 100
        print("100 slowest times in s for region / sphere overlaping")
        for k, v in sorted(
            times.items(), key=lambda item: item[1], reverse=True
        )[:100]:
            print(k, v)


if __name__ == "__main__":
    ts = TimeDestrieuxRegions()
    ts.setup(None)
    start = time.perf_counter()
    # ts.check_regions_overlap()
    # ts.time_regions_sphere_overlap(cardinal_relation_fast)
    # ts.time_regions_overlap(cardinal_relation_fast)
    ts.time_regions_convex(cardinal_relation_fast)
    stop = time.perf_counter()
    print(f"Took : {stop - start: .4f} s.")
