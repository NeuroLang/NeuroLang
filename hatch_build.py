import shutil
import subprocess
from pathlib import Path

from hatchling.builders.hooks.plugin.interface import BuildHookInterface


class CustomBuildHook(BuildHookInterface):
    """Custom hatchling build hook.

    Runs an optional npm build before the wheel is assembled: it builds the
    legacy neurolang-web frontend. Skipped if the dist/ directory already
    exists. Warns (does not fail) if npm is not installed.
    """

    def initialize(self, version, build_data):
        self._run_npm_build()

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
