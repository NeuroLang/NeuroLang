import configparser
import os
import shutil
import subprocess
from pathlib import Path

from hatchling.builders.hooks.plugin.interface import BuildHookInterface


class CustomBuildHook(BuildHookInterface):
    """Custom hatchling build hook.

    Runs two optional steps before the wheel is assembled:

    1. npm build — builds the legacy neurolang-web frontend.
       Skipped if the dist/ directory already exists.
       Warns (does not fail) if npm is not installed.

    2. Dask backend config — if the environment variable
       NEUROLANG_DASK=1 is set, mutates
       neurolang/config/config.ini to set [RAS] backend = dask.
    """

    def initialize(self, version, build_data):
        self._run_npm_build()
        self._maybe_set_dask_backend()

    # ------------------------------------------------------------------
    # npm build
    # ------------------------------------------------------------------

    def _run_npm_build(self):
        web_dir = (
            Path(self.root)
            / "neurolang"
            / "utils"
            / "server"
            / "neurolang-web"
        )
        dist_dir = web_dir / "dist"

        if dist_dir.exists():
            self.app.display_warning(
                "Frontend dist/ already exists — skipping npm build."
            )
            return

        npm = shutil.which("npm")
        if not npm:
            self.app.display_warning(
                "npm not found — skipping frontend build. "
                "Install Node.js from https://nodejs.org/ to build "
                "the frontend."
            )
            return

        self.app.display_info("Running: npm install")
        subprocess.check_call([npm, "install"], cwd=web_dir)

        self.app.display_info("Running: npm run build -- --mode dev")
        subprocess.check_call(
            [npm, "run", "build", "--", "--mode", "dev"], cwd=web_dir
        )

    # ------------------------------------------------------------------
    # Dask backend config
    # ------------------------------------------------------------------

    def _maybe_set_dask_backend(self):
        if not os.environ.get("NEUROLANG_DASK"):
            return

        config_file = (
            Path(self.root) / "neurolang" / "config" / "config.ini"
        )
        config = configparser.ConfigParser(
            allow_no_value=True, comment_prefixes=("//", "#")
        )
        config.optionxform = str
        config.read(config_file)
        config["RAS"]["backend"] = "dask"
        with open(config_file, "w") as fh:
            config.write(fh)

        self.app.display_info(
            "NEUROLANG_DASK=1 — set [RAS] backend = dask in config.ini"
        )
