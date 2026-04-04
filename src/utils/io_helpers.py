# src/utils/io_helpers.py
from __future__ import annotations

from pathlib import Path
import pandas as pd
from matplotlib.figure import Figure

from src.utils.paths import get_figure_path, get_table_path


def save_figure(
    fig: Figure,
    filename: str,
    category: str = "final",
    dpi: int = 300,
    bbox_inches: str = "tight",
) -> Path:
    path = get_figure_path(filename, category=category)
    fig.savefig(path, dpi=dpi, bbox_inches=bbox_inches)
    return path


def save_csv(
    df: pd.DataFrame,
    filename: str,
    category: str = "final",
    index: bool = False,
    encoding: str = "utf-8-sig",
) -> Path:
    path = get_table_path(filename, category=category)
    df.to_csv(path, index=index, encoding=encoding)
    return path


def save_markdown_table(
    df: pd.DataFrame,
    filename: str,
    category: str = "final",
    index: bool = False,
) -> Path:
    path = get_table_path(filename, category=category)
    path.write_text(df.to_markdown(index=index), encoding="utf-8")
    return path
