"""Shared paths, I/O helpers, formatting and plotting style for the HVIA project."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
FIGURES = ROOT / "outputs" / "figures"
TABLES = ROOT / "outputs" / "tables"
MODELS = ROOT / "outputs" / "models"

# Data covers Oct-2023 .. Sep-2026, i.e. exactly three Oct-Sep fiscal years.
AS_OF_DATE = pd.Timestamp("2026-09-30")
FISCAL_YEAR_START_MONTH = 10

PALETTE = {
    "primary": "#1F3A5F",
    "accent": "#D1495B",
    "good": "#2A9D8F",
    "warn": "#E9A23B",
    "muted": "#8D99AE",
    "light": "#E5E9F0",
}
CHANNEL_COLORS = {"Wholesale": "#1F3A5F", "Retail": "#2A9D8F", "Online": "#E9A23B"}


def ensure_dirs() -> None:
    for folder in (PROCESSED, FIGURES, TABLES, MODELS):
        folder.mkdir(parents=True, exist_ok=True)


def read_raw(name: str, **kwargs) -> pd.DataFrame:
    """Read a raw CSV (files carry a UTF-8 BOM)."""
    return pd.read_csv(RAW / f"{name}.csv", encoding="utf-8-sig", **kwargs)


def read_processed(name: str) -> pd.DataFrame:
    return pd.read_parquet(PROCESSED / f"{name}.parquet")


def write_processed(df: pd.DataFrame, name: str) -> None:
    df.to_parquet(PROCESSED / f"{name}.parquet", index=False)


def save_table(df: pd.DataFrame, name: str, index: bool = False) -> Path:
    path = TABLES / f"{name}.csv"
    df.to_csv(path, index=index, encoding="utf-8-sig")
    return path


def safe_div(numerator, denominator):
    """Element-wise division returning NaN instead of inf for zero denominators."""
    num = np.asarray(numerator, dtype="float64")
    den = np.asarray(denominator, dtype="float64")
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(den == 0, np.nan, num / den)
    if isinstance(numerator, pd.Series):
        return pd.Series(out, index=numerator.index)
    return out


def fiscal_year(dates: pd.Series) -> pd.Series:
    """Fiscal year label (FY24 = Oct-2023..Sep-2024)."""
    year = dates.dt.year + (dates.dt.month >= FISCAL_YEAR_START_MONTH).astype(int)
    return "FY" + (year % 100).astype(str).str.zfill(2)


# ---------------------------------------------------------------- formatting
def fmt_egp(value: float, decimals: int = 1) -> str:
    """Compact EGP format: 1.2M, 340K."""
    if pd.isna(value):
        return "n/a"
    sign = "-" if value < 0 else ""
    value = abs(value)
    if value >= 1e9:
        return f"{sign}EGP {value / 1e9:.{decimals}f}B"
    if value >= 1e6:
        return f"{sign}EGP {value / 1e6:.{decimals}f}M"
    if value >= 1e3:
        return f"{sign}EGP {value / 1e3:.{decimals}f}K"
    return f"{sign}EGP {value:,.0f}"


def fmt_pct(value: float, decimals: int = 1) -> str:
    return "n/a" if pd.isna(value) else f"{value * 100:.{decimals}f}%"


def fmt_int(value: float) -> str:
    return "n/a" if pd.isna(value) else f"{value:,.0f}"


# ------------------------------------------------------------------ plotting
def set_style() -> None:
    plt.rcParams.update(
        {
            "figure.figsize": (9, 4.8),
            "figure.dpi": 110,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": "#E5E9F0",
            "grid.linewidth": 0.8,
            "axes.axisbelow": True,
            "axes.titlesize": 13,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.labelcolor": "#333333",
            "font.size": 10,
            "legend.frameon": False,
        }
    )


def save_fig(fig: plt.Figure, name: str) -> Path:
    FIGURES.mkdir(parents=True, exist_ok=True)
    path = FIGURES / f"{name}.png"
    fig.savefig(path, dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def millions(x, _pos=None) -> str:
    return f"{x / 1e6:.0f}M" if abs(x) >= 1e6 else f"{x / 1e3:.0f}K"
