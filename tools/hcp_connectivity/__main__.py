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
    tasks(command("structural", "a worker parcellating the connectomes of "
                  "STAGING into OUT", "work", "staging", "out"), False)
    tasks(command("functional", "a task of the functional connectivity of "
                  "the HCP_DIR subjects into OUT", "work", "hcp", "out"), True)
    sub = command("combine", "the group connectivity of the parts of PARTS, "
                  "its tables and gradients, in DATASET")
    sub.add_argument("kind", choices=["structural", "functional"])
    for path in ("work", "parts", "dataset"):
        sub.add_argument(path, type=Path)
    sub = command("population", "the connectivity of the cortex and the "
                  "subcortex from a population tractogram, into OUT "
                  "(WORK: its atlases; needs the internet)", "work", "out")
    sub.add_argument("tractograms", nargs="+", type=Path)
    sub.add_argument("--radius", type=float, default=3.0)
    command("gradients", "the gradients again, of the matrices of DATASET",
            "work", "dataset")
    args = parser.parse_args(argv)

    if args.command == "atlases":
        from .atlases import build
        build(args.work)
    elif args.command == "download":
        from .download import run
        run(args.staging, args.out, args.jobs, args.most)
    elif args.command == "structural":
        from .structural import run
        run(args.work, args.staging, args.out, args.task)
    elif args.command == "functional":
        from .functional import run
        run(args.work, args.hcp, args.out, args.task, args.tasks)
    elif args.command == "combine":
        from . import combine
        if args.kind == "structural":
            combine.structural(args.work, args.parts, args.dataset)
        else:
            from .atlases import tian_names
            combine.functional(args.work, tian_names(args.work / "atlases"),
                               args.parts, args.dataset)
    elif args.command == "population":
        from .population import run
        run(args.work, args.out, args.tractograms, args.radius)
    elif args.command == "gradients":
        from .combine import regradient
        regradient(args.work, args.dataset)


if __name__ == "__main__":
    main()
