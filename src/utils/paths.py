"""
*** notebook ***
- exploratory 圖 -
fig_path = get_figure_path("price_series_0050.png", category="exploratory")
plt.savefig(fig_path, dpi=300, bbox_inches="tight")
print(fig_path)

- final 展示圖 -
fig_path = get_figure_path("conditional_high_state_win_rate.png", category="final")
plt.savefig(fig_path, dpi=300, bbox_inches="tight")
print(fig_path)
"""


# src/utils/paths.py
from __future__ import annotations

from pathlib import Path
from typing import Literal

# 假設此檔案位置為: all-in-vs-dca/src/utils/paths.py
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Core folders
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
CACHE_DIR = DATA_DIR / "cache"

NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
REPORTS_DIR = PROJECT_ROOT / "reports"

# Reports subfolders
FIGURES_DIR = REPORTS_DIR / "figures"
TABLES_DIR = REPORTS_DIR / "tables"
ONE_PAGER_DIR = REPORTS_DIR / "one_pager"

FIGURES_EXPLORATORY_DIR = FIGURES_DIR / "exploratory"
FIGURES_MODEL_OUTPUTS_DIR = FIGURES_DIR / "model_outputs"
FIGURES_FINAL_DIR = FIGURES_DIR / "final"

TABLES_EXPLORATORY_DIR = TABLES_DIR / "exploratory"
TABLES_MODEL_OUTPUTS_DIR = TABLES_DIR / "model_outputs"
TABLES_FINAL_DIR = TABLES_DIR / "final"


def ensure_project_dirs() -> None:
    """
    Create all key project directories if they do not exist.
    Safe to call multiple times.
    """
    dirs = [
        DATA_DIR,
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        CACHE_DIR,
        NOTEBOOKS_DIR,
        REPORTS_DIR,
        FIGURES_DIR,
        TABLES_DIR,
        ONE_PAGER_DIR,
        FIGURES_EXPLORATORY_DIR,
        FIGURES_MODEL_OUTPUTS_DIR,
        FIGURES_FINAL_DIR,
        TABLES_EXPLORATORY_DIR,
        TABLES_MODEL_OUTPUTS_DIR,
        TABLES_FINAL_DIR,
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)


ReportKind = Literal["figures", "tables"]
ReportCategory = Literal["exploratory", "model_outputs", "final"]


def get_report_dir(kind: ReportKind, category: ReportCategory) -> Path:
    """
    Return the output directory for reports.

    Parameters
    ----------
    kind : {"figures", "tables"}
        Output type.
    category : {"exploratory", "model_outputs", "final"}
        Output category.

    Returns
    -------
    Path
        Directory path, already created.
    """
    ensure_project_dirs()

    mapping = {
        ("figures", "exploratory"): FIGURES_EXPLORATORY_DIR,
        ("figures", "model_outputs"): FIGURES_MODEL_OUTPUTS_DIR,
        ("figures", "final"): FIGURES_FINAL_DIR,
        ("tables", "exploratory"): TABLES_EXPLORATORY_DIR,
        ("tables", "model_outputs"): TABLES_MODEL_OUTPUTS_DIR,
        ("tables", "final"): TABLES_FINAL_DIR,
    }

    out_dir = mapping[(kind, category)]
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def get_figure_path(filename: str, category: ReportCategory = "final") -> Path:
    """
    Return a full figure path under reports/figures/<category>/filename
    """
    out_dir = get_report_dir("figures", category)
    return out_dir / filename


def get_table_path(filename: str, category: ReportCategory = "final") -> Path:
    """
    Return a full table path under reports/tables/<category>/filename
    """
    out_dir = get_report_dir("tables", category)
    return out_dir / filename
