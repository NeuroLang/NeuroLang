"""
Plotting functions for the general notebook
===========================================
"""

import matplotlib.style

import matplotlib
import nilearn
from nilearn import plotting
from nilearn.image import mean_img

matplotlib.style.use("ggplot")


###############################################################################
# Plots in surface
# ===============
#


def plot_surface_map_of_individual_folds(
    subjects, q_list, queries_dict, subject_info_dict, query_name
):
    list_of_fold_spatial_images = []
    for s in subjects:
        s_index = subjects.index(s)
        df_one_query = queries_dict[queries_dict["query"] == query_name]
        wanted_fold = df_one_query[df_one_query["subject"] == s].sulcus
        fold_results = wanted_fold.values[0]
        if "No sulcus found" not in fold_results:
            if (
                fold_results
                in subject_info_dict.iloc[s_index][
                    "destrieux_spatial_images"
                ].keys()
            ):
                list_of_fold_spatial_images.append(
                    subject_info_dict.iloc[s_index][
                        "destrieux_spatial_images"
                    ][fold_results]
                )
            else:
                pass
        else:
            pass

        for x in list_of_fold_spatial_images:
            plotting.surf_plotting.plot_surf(
                x, title=f"{s}, {query_name} result"
            )


def plot_surf_prob_map(
    subjects,
    q_list,
    queries_dict,
    subject_info_dict,
    primary_sulcus_name,
    query_name,
    plane="lateral",
    save_as=None,
    interactive=False,
):
    list_of_fold_spatial_images = []
    for s in subjects:
        s_index = subjects.index(s)
        if primary_sulcus_name is not None:
            list_of_fold_spatial_images.append(
                subject_info_dict.iloc[s_index]["destrieux_spatial_images"][
                    primary_sulcus_name
                ]
            )
        if query_name is not None:
            df_one_query = queries_dict[queries_dict["query"] == query_name]
            wanted_fold = df_one_query[df_one_query["subject"] == s].sulcus
            fold_results = wanted_fold.values[0]
            if "No sulcus found" not in fold_results:
                if (
                    fold_results
                    in subject_info_dict.iloc[s_index][
                        "destrieux_spatial_images"
                    ].keys()
                ):
                    list_of_fold_spatial_images.append(
                        subject_info_dict.iloc[s_index][
                            "destrieux_spatial_images"
                        ][fold_results]
                    )
                else:
                    pass
            else:
                pass

    mean_spatial_img = mean_img(list_of_fold_spatial_images)
    spatial_surf_from_vol = nilearn.surface.vol_to_surf(
        mean_spatial_img, "107321.L.inflated.32k_fs_LR.surf.gii"
    )

    if not interactive:
        if primary_sulcus_name is not None:
            return plotting.plot_surf_stat_map(
                "107321.L.very_inflated.32k_fs_LR.surf.gii",
                spatial_surf_from_vol,
                bg_map="107321.L.sulc.32k_fs_LR.shape.gii",
                threshold=0.1,
                cmap="viridis",
                vmax=1,
                title=f"{primary_sulcus_name}",
                darkness=0.75,
                output_file=save_as,
                view=plane,
            )
        if query_name is not None:
            return plotting.plot_surf_stat_map(
                "107321.L.very_inflated.32k_fs_LR.surf.gii",
                spatial_surf_from_vol,
                bg_map="107321.L.sulc.32k_fs_LR.shape.gii",
                threshold=0.1,
                cmap="viridis",
                vmax=1,
                title=f"{query_name}",
                darkness=0.75,
                output_file=save_as,
                view=plane,
            )
    else:
        return plotting.view_surf(
            "107321.L.very_inflated.32k_fs_LR.surf.gii",
            spatial_surf_from_vol,
            bg_map="107321.L.sulc.32k_fs_LR.shape.gii",
            threshold=0.01,
            cmap="viridis",
            vmin=0,
            vmax=1,
            symmetric_cmap=False,
        )


###############################################################################
# Plots in volume
# ===============
#


def plot_all_folds_per_subject(
    subjects,
    queries_dict,
    subject_info_dict,
    dimension="ortho",
    cut_coords=None,
    save_as=None,
):
    list_of_fold_spatial_images = []
    for x, s in enumerate(subjects):
        for k in subject_info_dict.iloc[x]["destrieux_spatial_images"].keys():
            list_of_fold_spatial_images.append(
                subject_info_dict.iloc[x]["destrieux_spatial_images"][k]
            )
        mean_spatial_img = mean_img(list_of_fold_spatial_images)
        plotting.plot_stat_map(
            mean_spatial_img,
            display_mode=dimension,
            cut_coords=cut_coords,
            threshold=0,
            colorbar=False,
            cmap="Set3",
            draw_cross=False,
            vmax=0.0005,
            title=f"{s} left",
            output_file=save_as,
        )


def plot_individual_fold_per_subject(
    subjects,
    q_list,
    queries_dict,
    subject_info_dict,
    primary_sulcus_name,
    query_name,
    dimension="ortho",
    cut_coords=None,
    save_as=None,
):
    for x, s in enumerate(subjects):
        if primary_sulcus_name is not None:
            plotting.plot_roi(
                subject_info_dict.iloc[x]["destrieux_spatial_images"][
                    primary_sulcus_name
                ],
                display_mode=dimension,
                cut_coords=cut_coords,
                threshold=0.1,
                colorbar=False,
                cmap="cool",
                draw_cross=False,
                vmax=1,
                title=f"{s}, Left {primary_sulcus_name}",
                output_file=save_as,
            )
        if query_name is not None:
            df_one_query = queries_dict[queries_dict["query"] == query_name]
            wanted_fold = df_one_query[df_one_query["subject"] == s].sulcus
            fold_results = wanted_fold.values[0]
            if "No sulcus found" not in fold_results:
                if (
                    fold_results
                    in subject_info_dict.iloc[x][
                        "destrieux_spatial_images"
                    ].keys()
                ):
                    plotting.plot_roi(
                        subject_info_dict.iloc[x]["destrieux_spatial_images"][
                            fold_results
                        ],
                        display_mode=dimension,
                        cut_coords=cut_coords,
                        threshold=0.1,
                        colorbar=False,
                        cmap="cool",
                        draw_cross=False,
                        vmax=1,
                        title=(
                            f"{s}, Left {query_name},"
                            + f" Destrieux sulcus={fold_results}",
                        ),
                        output_file=save_as,
                    )
                else:
                    pass
            else:
                pass
        else:
            pass


def plot_stat_map_of_folds(
    subjects,
    q_list,
    queries_dict,
    subject_info_dict,
    primary_sulcus_name,
    query_name,
    dimension="ortho",
    cut_coords=None,
    save_as=None,
    interactive=False,
):
    list_of_fold_spatial_images = []
    for x, s in enumerate(subjects):
        if primary_sulcus_name is not None:
            list_of_fold_spatial_images.append(
                subject_info_dict.iloc[x]["destrieux_spatial_images"][
                    primary_sulcus_name
                ]
            )
        if query_name is not None:
            df_one_query = queries_dict[queries_dict["query"] == query_name]
            wanted_fold = df_one_query[df_one_query["subject"] == s].sulcus
            fold_results = wanted_fold.values[0]
            if "No sulcus found" not in fold_results:
                if (
                    fold_results
                    in subject_info_dict.iloc[x][
                        "destrieux_spatial_images"
                    ].keys()
                ):
                    list_of_fold_spatial_images.append(
                        subject_info_dict.iloc[x]["destrieux_spatial_images"][
                            fold_results
                        ]
                    )
                else:
                    pass
            else:
                pass
        else:
            pass
    mean_spatial_img = mean_img(list_of_fold_spatial_images)
    print(
        "Value of voxel with maximum overlap: ", mean_spatial_img.dataobj.max()
    )
    proportion = round(len(list_of_fold_spatial_images) / len(subjects), 2)
    if not interactive:
        return plotting.plot_stat_map(
            mean_spatial_img,
            display_mode=dimension,
            cut_coords=cut_coords,
            threshold=0.1,
            colorbar=True,
            cmap="tab10",
            draw_cross=False,
            vmax=1,
            title=(
                "Results found in"
                + f" {len(list_of_fold_spatial_images)} / {len(subjects)}"
                + f" subjects, proportion={proportion}"
            ),
            output_file=save_as,
        )
    else:
        return plotting.view_img(
            mean_spatial_img,
            threshold=0.01,
            cmap="viridis",
            vmin=0,
            vmax=1,
            title=(
                f"{primary_sulcus_name},"
                + f" {len(list_of_fold_spatial_images)} / {len(subjects)}"
                + f" subjects"
            ),
            symmetric_cmap=False,
        )
