"""
python -m hcp_connectivity COMMAND ... (from tools/, or with tools/ on
PYTHONPATH); see README.md.

The workers take a share of the work, ``--task I --tasks N``; by default
those of a Slurm array (SLURM_ARRAY_TASK_ID, SLURM_ARRAY_TASK_COUNT), so
that ``sbatch --array=0-39 --wrap "python -m hcp_connectivity functional
..."`` runs them on a cluster, and a loop of ``--task`` on one machine.
"""

import argparse
import os
from pathlib import Path


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(prog="python -m hcp_connectivity",
                                     description=__doc__.split("\n\n")[1])
    commands = parser.add_subparsers(dest="command", required=True)

    def command(name, help_text, *paths):
        sub = commands.add_parser(name, help=help_text)
        for path in paths:
            sub.add_argument(path, type=Path)
        return sub

    def tasks(sub, share):
        sub.add_argument("--task", type=int, default=int(
            os.environ.get("SLURM_ARRAY_TASK_ID", 0)))
        if share:
            sub.add_argument("--tasks", type=int, default=int(
                os.environ.get("SLURM_ARRAY_TASK_COUNT", 1)))

    command("atlases", "the atlases on the grayordinates (WORK/atlases.npz;"
            " needs the internet)", "work")
    sub = command("download", "the structural connectomes into STAGING "
                  "(needs the internet)", "staging", "out")
    sub.add_argument("--jobs", type=int, default=6)
    sub.add_argument("--most", type=int, default=40)
    sub = command("osf", "the files of an OSF project into FOLDER, checked "
                  "by their md5 (needs the internet)")
    sub.add_argument("project")
    sub.add_argument("folder", type=Path)
    sub.add_argument("--jobs", type=int, default=6)
    sub = command("check-space", "whether a tractogram is in the MNI space "
                  "of its HCP subject's segmentation", "tractogram", "hcp")
    sub.add_argument("--subject")
    sub.add_argument("--shift", type=int, default=3)
    sub = command("fit-space", "the affine that maps HCP subjects' "
                  "tractograms onto their MNINonLinear segmentations",
                  "hcp", "out")
    sub.add_argument("tractograms", type=Path, nargs="+")
    sub = command("dense", "a dense tractogram of at most --megabytes MB "
                  "and its subjects' segmentation", "trk_dir", "hcp", "out")
    sub.add_argument("--name", default="disconnectome7t")
    sub.add_argument("--subjects", type=int, default=20)
    sub.add_argument("--affine", type=Path)
    sub.add_argument("--minimum-length", type=float, default=30)
    sub.add_argument("--tolerance", type=float, default=0.5)
    sub.add_argument("--megabytes", type=float, default=480)
    sub.add_argument("--count", type=int)
    sub.add_argument("--brain", action="store_true")
    tasks(command("structural", "a worker parcellating the connectomes of "
                  "STAGING into OUT", "work", "staging", "out"), False)
    tasks(command("functional", "a task of the functional connectivity of "
                  "the HCP_DIR subjects into OUT", "work", "hcp", "out"), True)
    tasks(command("voxel-functional", "a task of the partial functional "
                  "connectivity of the cortical voxels of the HCP_DIR "
                  "subjects with Schaefer's parcels into OUT", "work", "hcp",
                  "out"), True)
    command("volume-atlas", "the atlas in MNI space, for tractography "
            "(WORK/atlases; needs the internet)", "work")
    sub = command("tractography", "a task of the tractography of the HCP_DIR "
                  "subjects into OUT", "work", "hcp", "out")
    tasks(sub, True)
    sub.add_argument("--streamlines", type=int, default=5_000_000)
    sub.add_argument("--limit", type=int, help="the first LIMIT subjects "
                     "with diffusion data")
    sub.add_argument("--scratch", type=Path, help="the intermediate files "
                     "(default WORK/tractography)")
    sub.add_argument("--keep", action="store_true",
                     help="keep the intermediate files")
    sub = command("combine", "the group connectivity of the parts of PARTS, "
                  "its tables and gradients, in DATASET")
    sub.add_argument("kind", choices=["structural", "functional",
                                      "tractography", "voxel-functional"])
    for path in ("work", "parts", "dataset"):
        sub.add_argument(path, type=Path)
    sub.add_argument("--threshold", type=float, default=0.05,
                     help="voxel-functional: the engine table keeps the "
                     "pairs whose r is above it (the counts at some cutoffs "
                     "are printed)")
    sub = command("population", "the connectivity of the cortex and the "
                  "subcortex from a population tractogram, into OUT "
                  "(WORK: its atlases; needs the internet)", "work", "out")
    sub.add_argument("tractograms", nargs="+", type=Path)
    sub.add_argument("--radius", type=float, default=3.0)
    command("engine-tables", "the engines' connectivity tables, of the "
            "matrices of DATASET", "dataset")
    command("gradients", "the gradients again, of the matrices of DATASET",
            "work", "dataset")
    args = parser.parse_args(argv)

    if args.command == "atlases":
        from .atlases import build
        build(args.work)
    elif args.command == "download":
        from .download import run
        run(args.staging, args.out, args.jobs, args.most)
    elif args.command == "check-space":
        from .space import run
        run(args.tractogram, args.hcp, args.subject, args.shift)
    elif args.command == "dense":
        from .dense import run
        run(args.trk_dir, args.hcp, args.out, args.name, args.subjects,
            args.affine, args.minimum_length, args.tolerance, args.megabytes,
            count=args.count, brain=args.brain)
    elif args.command == "fit-space":
        import numpy as np

        from .space import fit
        matrix = fit(args.tractograms, args.hcp)
        np.savetxt(args.out, matrix, fmt="%.6f")
        print(matrix)
    elif args.command == "osf":
        from .osf import run
        run(args.project, args.folder, args.jobs)
    elif args.command == "structural":
        from .structural import run
        run(args.work, args.staging, args.out, args.task)
    elif args.command == "functional":
        from .functional import run
        run(args.work, args.hcp, args.out, args.task, args.tasks)
    elif args.command == "voxel-functional":
        from .voxel_functional import run
        run(args.work, args.hcp, args.out, args.task, args.tasks)
    elif args.command == "volume-atlas":
        from .tractography import volume_atlas
        volume_atlas(args.work / "atlases")
    elif args.command == "tractography":
        from .tractography import run
        run(args.work, args.hcp, args.out, args.task, args.tasks,
            args.streamlines, args.limit, args.scratch, args.keep)
    elif args.command == "combine":
        from . import combine
        if args.kind == "structural":
            combine.structural(args.work, args.parts, args.dataset)
        elif args.kind == "tractography":
            combine.tractography(args.work, args.parts, args.dataset)
        elif args.kind == "voxel-functional":
            combine.voxel_functional(args.work, args.parts, args.dataset,
                                     args.threshold)
        else:
            from .atlases import tian_names
            combine.functional(args.work, tian_names(args.work / "atlases"),
                               args.parts, args.dataset)
    elif args.command == "population":
        from .population import run
        run(args.work, args.out, args.tractograms, args.radius)
    elif args.command == "engine-tables":
        from .combine import engine_tables
        engine_tables(args.dataset)
    elif args.command == "gradients":
        from .combine import regradient
        regradient(args.work, args.dataset)


if __name__ == "__main__":
    main()
