"""
Make the images and figures of doc/tutorial_decoding.rst.

Two images of activations are loaded with ``.image`` in the REPL on the
neurosynth engine and decoded with the query ``.image`` offers: a mask of
the right amygdala, a sphere of 8 mm, and nilearn's sample motor z-map
(NeuroVault image 10426, left vs right button press). For each, it writes
in the output directory the image, a picture of it on the brain and a
chart of its ten largest lifts, in the light and the dark theme of the
documentation.

    python doc/tools/make_decoding_figures.py -d DATA_DIR [-o OUTPUT]
"""

import argparse
import io
import shutil
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import nibabel as nib  # noqa: E402
import numpy as np  # noqa: E402
from rich.console import Console  # noqa: E402

AMYGDALA_MNI = (24.0, -2.0, -20.0)  # the right amygdala, in mm
AMYGDALA_RADIUS = 8.0
SHOWN_TERMS = 10
# The 95% interval of the lift of a term, over the replicates of the
# studies (doc/tutorial_decoding.rst).
INTERVAL = (
    "lift_replicate(b, x) :- "
    "RESAMPLE[boot_studies](image_decoding('TERM', x, n)) = b\n"
    "interval(lift, percentile(x, 2.5), percentile(x, 97.5)) :- "
    "image_decoding('TERM', lift, n), lift_replicate(b, x)\n"
    "ans(lift, ci_low, ci_high) :- interval(lift, ci_low, ci_high)\n"
)

# The chart's colours, on the surfaces of the documentation's themes.
THEMES = {
    "light": {"bar": "#3b6fd8", "ink": "#1f2328", "muted": "#59636e",
              "grid": "#d8dee4", "surface": "#ffffff"},
    "dark": {"bar": "#7aa2f7", "ink": "#e6edf3", "muted": "#9198a1",
             "grid": "#30363d", "surface": "#14181f"},
}


def amygdala_mask(template: nib.Nifti1Image) -> nib.Nifti1Image:
    """Return the sphere of the right amygdala on the grid of *template*."""
    ijk = np.indices(template.shape[:3]).reshape(3, -1).T
    xyz = ijk @ template.affine[:3, :3].T + template.affine[:3, 3]
    inside = np.linalg.norm(xyz - AMYGDALA_MNI, axis=1) <= AMYGDALA_RADIUS
    data = inside.reshape(template.shape[:3]).astype(np.uint8)
    return nib.Nifti1Image(data, template.affine)


def bar_chart(frame, title: str, path: Path, theme: str) -> None:
    """Draw the largest lifts of *frame* as horizontal bars in *path*."""
    colours = THEMES[theme]
    top = frame.nlargest(SHOWN_TERMS, "lift").iloc[::-1]
    fig, ax = plt.subplots(figsize=(6.4, 0.32 * len(top) + 0.9), dpi=150)
    fig.patch.set_facecolor(colours["surface"])
    ax.set_facecolor(colours["surface"])
    bars = ax.barh(top["term"], top["lift"], height=0.55,
                   color=colours["bar"], zorder=2)
    for bar, (lift, studies) in zip(bars, zip(top["lift"], top["studies"])):
        ax.text(bar.get_width() + 0.04, bar.get_y() + bar.get_height() / 2,
                f"{lift:.2f}  ({studies} studies)", va="center",
                fontsize=8, color=colours["muted"])
    ax.axvline(1, color=colours["muted"], linewidth=0.8, linestyle=":",
               zorder=1)
    ax.set_xlim(0, top["lift"].max() * 1.35)
    ax.set_xlabel("lift: P(peak in the image | term) / P(peak in the image)",
                  color=colours["muted"], fontsize=8)
    ax.set_title(title, color=colours["ink"], fontsize=10, loc="left")
    ax.tick_params(colors=colours["ink"], labelsize=8, length=0)
    ax.grid(axis="x", color=colours["grid"], linewidth=0.6, zorder=0)
    for side in ax.spines.values():
        side.set_visible(False)
    fig.tight_layout()
    fig.savefig(path, facecolor=colours["surface"])
    plt.close(fig)


def brain_picture(image_path: Path, path: Path, mask: bool) -> None:
    """Draw the image *image_path* on the MNI template in *path*."""
    from nilearn import plotting

    if mask:
        display = plotting.plot_roi(
            str(image_path), cut_coords=AMYGDALA_MNI, draw_cross=False,
            cmap="autumn", black_bg=True,
        )
    else:  # its positive tail, the activations .image decodes
        image = nib.load(image_path)
        values = np.asanyarray(image.dataobj)
        display = plotting.plot_stat_map(
            nib.Nifti1Image(np.where(values > 0, values, 0), image.affine),
            threshold=3.1, cut_coords=(38, -22, 56), draw_cross=False,
            black_bg=True, cmap="hot", symmetric_cbar=False, vmin=0,
        )
    display.savefig(str(path), dpi=150)
    display.close()


def main() -> None:
    from nilearn import datasets

    from neurolang_apps.utils import engine_registry
    from neurolang_apps.utils.tui import InteractiveTuiApp, TuiSettings
    from neurolang_apps.utils.tui import nifti

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("-d", "--data-dir", type=Path, required=True)
    parser.add_argument("-o", "--output", type=Path,
                        default=Path(__file__).parents[1] / "_static"
                        / "decoding")
    parser.add_argument("-r", "--resolution", type=float, default=2.0)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    nl = engine_registry.build_engine(
        "neurosynth", args.data_dir, args.resolution
    )
    nifti.webbrowser.open = lambda *a, **k: None  # no viewer here
    console = Console(file=io.StringIO(), width=200)
    app = InteractiveTuiApp(nl, TuiSettings(engine_name="neurosynth"),
                            console=console, err_console=console)

    mask_path = args.output / "amygdala_mask.nii.gz"
    nib.save(amygdala_mask(nl.template), mask_path)
    zmap_path = args.output / "motor_zmap.nii.gz"
    shutil.copy(datasets.load_sample_motor_activation_image(), zmap_path)

    for stem, path, title in (
        ("amygdala", mask_path, "A mask of the right amygdala"),
        ("motor", zmap_path, "The motor z-map, left vs right button press"),
    ):
        app._handle_dot_command(f".image {path} --name image")
        query = nifti.image_decoding(app, "image")["query"]
        start = time.perf_counter()
        app._execute_and_display(query)
        seconds = time.perf_counter() - start
        frame = app.results[-1].frame.sort_values("lift", ascending=False)
        for theme in THEMES:
            chart = args.output / f"{stem}_decoding_{theme}.png"
            bar_chart(frame, title, chart, theme)
        brain_picture(path, args.output / f"{stem}_image.png",
                      mask=stem == "amygdala")
        print(f"{stem}: {seconds:.1f} s, {len(frame)} terms")
        print(frame.head(SHOWN_TERMS).to_string(index=False))
        top = frame["term"].iloc[0]
        start = time.perf_counter()
        app._execute_and_display(INTERVAL.replace("TERM", top))
        seconds = time.perf_counter() - start
        print(f"{stem}: interval of {top} in {seconds:.1f} s")
        print(app.results[-1].frame.to_string(index=False))
    print(console.file.getvalue())


if __name__ == "__main__":
    main()
