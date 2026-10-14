# SPDX-License-Identifier: GPL-3.0-or-later
"""Draw the paper's figures as vector PDF into figures/, from the same tables and drawing code as the write-up.

    python make_figures.py

It imports ../writeup/figures/make_figures.py and changes only the presentation for print: no in-figure title,
subtitle or source line (the caption carries them), a white surface, a smaller canvas so text prints at about
7-8 pt at text width, and embedded fonts. Numbers come from ../writeup/data/*.csv as in the write-up.
"""
import importlib.util
import sys
from pathlib import Path

sys.dont_write_bytecode = True   # the import below would leave a __pycache__ in writeup/figures/

HERE = Path(__file__).resolve().parent
SRC = HERE.parent / "writeup" / "figures" / "make_figures.py"
OUT = HERE / "figures"

spec = importlib.util.spec_from_file_location("writeup_figures", SRC)
mf = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mf)
plt = mf.plt

SCALE = 0.75  # canvas factor: a 10 in figure printed at the 5.6 in text width keeps text near 7.5 pt
SCALE_BY_FIGURE = {"fig_pipeline": 0.95, "fig_timeline": 0.85}  # boxes and labels placed in data units need the full canvas
FIGURES = ["fig_timeline", "fig_regimes", "fig_concentration", "fig_mixture", "fig_copy_reliability", "fig_pipeline",
           "fig_progress", "fig_tail", "fig_airports", "fig_months", "fig_rehearsal", "fig_coverage", "fig_calibration", "fig_error_budget"]

mf.SURFACE = "white"
plt.rcParams.update({"figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
                     "pdf.fonttype": 42, "svg.fonttype": "path"})
mf.header = lambda *a, **k: None
mf.footer = lambda *a, **k: None

_subplots = plt.subplots


def subplots(*args, figsize=None, **kw):
    if figsize is not None:
        figsize = (figsize[0] * _scale[0], figsize[1] * _scale[0])
    return _subplots(*args, figsize=figsize, **kw)


def save(fig, name):
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / f"{name.replace('_', '-')}.pdf", bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    print("wrote", name)


_scale = [SCALE]
plt.subplots = subplots
mf.save = save

if __name__ == "__main__":
    for name in FIGURES:
        _scale[0] = SCALE_BY_FIGURE.get(name, SCALE)
        getattr(mf, name)()
