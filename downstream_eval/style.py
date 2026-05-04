from __future__ import annotations
import shutil
import matplotlib as mpl
SERIF_STACK: list[str] = ['Times New Roman', 'STIX Two Text', 'DejaVu Serif', 'Liberation Serif', 'serif']
SANS_STACK: list[str] = ['Helvetica', 'Arial', 'DejaVu Sans', 'Liberation Sans', 'sans-serif']

def apply_publication_style(use_latex: bool=False) -> None:
    latex_available = shutil.which('latex') is not None
    effective_latex = use_latex and latex_available
    if use_latex and (not latex_available):
        print('[downstream_eval.style] latex executable not found, falling back to matplotlib built-in renderer.')
    mpl.rcParams.update({'font.family': 'serif', 'font.serif': SERIF_STACK, 'font.sans-serif': SANS_STACK, 'text.usetex': effective_latex, 'pdf.fonttype': 42, 'ps.fonttype': 42, 'font.size': 10, 'axes.labelsize': 10, 'axes.titlesize': 11, 'xtick.labelsize': 9, 'ytick.labelsize': 9, 'legend.fontsize': 9, 'axes.linewidth': 0.8, 'axes.edgecolor': '#222222', 'grid.color': '#cccccc', 'grid.linewidth': 0.5, 'savefig.dpi': 300, 'savefig.bbox': 'tight', 'savefig.pad_inches': 0.05, 'savefig.facecolor': 'white', 'image.interpolation': 'antialiased'})
