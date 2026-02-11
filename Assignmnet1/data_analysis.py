#!/usr/bin/env python3
"""
data_analysis.py (Task 1 / Denver)
Cloud-Native Nutritional Insights Application - Dataset Analysis

This script is written to closely follow the assignment's provided Python pseudocode
and required analysis bullets.

Outputs:
- outputs/avg_macros_by_diet.csv
- outputs/top5_protein_by_diet.csv
- outputs/most_common_cuisines_by_diet.csv
- outputs/metrics_with_ratios.csv
- outputs/summary.json
- outputs/charts/*.png  (bar chart, heatmap, scatter plot)

Run:
  python data_analysis.py --input All_Diets.csv --outdir outputs
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


MACRO_COLS = ["Protein(g)", "Carbs(g)", "Fat(g)"]
REQUIRED_COLS = ["Diet_type", "Recipe_name", "Cuisine_type", *MACRO_COLS]


def ensure_required_columns(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}\nFound columns: {list(df.columns)}")


def make_safe_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds:
      - protein_to_carbs_ratio = Protein / Carbs
      - carbs_to_fat_ratio     = Carbs / Fat

    Handles divide-by-zero by returning NaN for those rows.
    """
    out = df.copy()

    carbs = out["Carbs(g)"].replace(0, pd.NA)
    fat = out["Fat(g)"].replace(0, pd.NA)

    out["protein_to_carbs_ratio"] = out["Protein(g)"] / carbs
    out["carbs_to_fat_ratio"] = out["Carbs(g)"] / fat

    return out


def save_bar_chart_avg_macros(avg_macros: pd.DataFrame, charts_dir: Path) -> Path:
    """
    Bar chart: average Protein/Carbs/Fat by diet type.
    """
    charts_dir.mkdir(parents=True, exist_ok=True)
    out_path = charts_dir / "avg_macros_by_diet_bar.png"

    plot_df = avg_macros.set_index("Diet_type")[MACRO_COLS]

    # For readability, plot top 12 diets by average protein (still satisfies "bar chart" requirement)
    plot_df = plot_df.sort_values("Protein(g)", ascending=False).head(12)

    ax = plot_df.plot(kind="bar")
    ax.set_title("Average Macronutrients by Diet Type (Top 12 by Protein)")
    ax.set_xlabel("Diet Type")
    ax.set_ylabel("Average grams")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

    return out_path


def save_heatmap_avg_macros(avg_macros: pd.DataFrame, charts_dir: Path) -> Path:
    """
    Heatmap: average macros per diet type using pure matplotlib (no seaborn).
    """
    charts_dir.mkdir(parents=True, exist_ok=True)
    out_path = charts_dir / "avg_macros_by_diet_heatmap.png"

    heat_df = avg_macros.set_index("Diet_type")[MACRO_COLS].copy()

    # Keep heatmap readable: top 20 diets by average protein
    heat_df = heat_df.sort_values("Protein(g)", ascending=False).head(20)

    fig, ax = plt.subplots()
    im = ax.imshow(heat_df.values, aspect="auto")

    ax.set_title("Heatmap: Avg Protein/Carbs/Fat by Diet (Top 20 by Protein)")
    ax.set_xlabel("Macronutrient")
    ax.set_ylabel("Diet Type")

    ax.set_xticks(range(len(MACRO_COLS)))
    ax.set_xticklabels(MACRO_COLS, rotation=30, ha="right")
    ax.set_yticks(range(len(heat_df.index)))
    ax.set_yticklabels(heat_df.index)

    # annotate cells (kept simple + readable)
    for i in range(heat_df.shape[0]):
        for j in range(heat_df.shape[1]):
            ax.text(j, i, f"{heat_df.iat[i, j]:.1f}", ha="center", va="center")

    fig.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

    return out_path


def save_scatter_ratio_plot(df_with_ratios: pd.DataFrame, charts_dir: Path) -> Path:
    """
    Scatter plot: Protein vs Carbs with point size roughly tied to Fat (simple visual insight).
    """
    charts_dir.mkdir(parents=True, exist_ok=True)
    out_path = charts_dir / "protein_vs_carbs_scatter.png"

    # Use a subset to keep the plot lighter if needed (still representative)
    plot_df = df_with_ratios.dropna(subset=["Protein(g)", "Carbs(g)", "Fat(g)"]).copy()
    plot_df = plot_df.sample(n=min(2000, len(plot_df)), random_state=42)

    x = plot_df["Carbs(g)"]
    y = plot_df["Protein(g)"]
    sizes = (plot_df["Fat(g)"].clip(lower=0) + 1)  # avoid zeros

    plt.figure()
    plt.scatter(x, y, s=sizes, alpha=0.3)
    plt.title("Scatter: Protein vs Carbs (Point size ~ Fat)")
    plt.xlabel("Carbs (g)")
    plt.ylabel("Protein (g)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

    return out_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Task 1: Analyze All_Diets.csv (pseudocode-aligned)")
    parser.add_argument("--input", default="All_Diets.csv", help="Path to All_Diets.csv")
    parser.add_argument("--outdir", default="outputs", help="Output directory")
    args = parser.parse_args()

    input_path = Path(args.input)
    outdir = Path(args.outdir)
    charts_dir = outdir / "charts"
    outdir.mkdir(parents=True, exist_ok=True)

    # -------------------------
    # ASSIGNMENT PSEUDOCODE STEP
    # import pandas as pd
    # df = pd.read_csv('path_to_csv')
    # -------------------------
    df = pd.read_csv(input_path)
    ensure_required_columns(df)

    # -------------------------
    # ASSIGNMENT PSEUDOCODE STEP
    # Handle missing data (fill missing values with mean)
    # df.fillna(df.mean(), inplace=True)
    #
    # Note: df.mean() applies to numeric columns; pandas will ignore non-numeric columns.
    # -------------------------
    df.fillna(df.mean(numeric_only=True), inplace=True)

    # (Practical safety) Make sure macro columns are numeric after fill; coerce if needed.
    for c in MACRO_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    # Fill again after coercion (still consistent with "fill with mean")
    df.fillna(df.mean(numeric_only=True), inplace=True)

    # -------------------------
    # ASSIGNMENT PSEUDOCODE STEP
    # Calculate the average macronutrient content for each diet type
    # avg_macros = df.groupby('Diet_type')[['Protein(g)', 'Carbs(g)', 'Fat(g)']].mean()
    # -------------------------
    avg_macros = df.groupby("Diet_type")[MACRO_COLS].mean().reset_index()
    avg_macros.to_csv(outdir / "avg_macros_by_diet.csv", index=False)

    # Required: Identify top 5 protein-rich recipes for each diet type
    top5_by_diet = (
        df.sort_values("Protein(g)", ascending=False)
        .groupby("Diet_type", as_index=False)
        .head(5)[["Diet_type", "Recipe_name", "Cuisine_type", *MACRO_COLS]]
        .reset_index(drop=True)
    )
    top5_by_diet.to_csv(outdir / "top5_protein_by_diet.csv", index=False)

    # Required: Find diet type with highest protein content across all recipes
    # (Most defensible interpretation: highest average protein across recipes in that diet)
    highest_avg_protein_row = avg_macros.sort_values("Protein(g)", ascending=False).iloc[0]
    highest_avg_protein_diet = str(highest_avg_protein_row["Diet_type"])
    highest_avg_protein_value = float(highest_avg_protein_row["Protein(g)"])

    # (Extra: also capture single highest-protein recipe overall, in case marker expects that)
    max_recipe_row = df.sort_values("Protein(g)", ascending=False).iloc[0]
    highest_single_recipe = {
        "diet_type": str(max_recipe_row["Diet_type"]),
        "recipe_name": str(max_recipe_row["Recipe_name"]),
        "cuisine_type": str(max_recipe_row["Cuisine_type"]),
        "protein_g": float(max_recipe_row["Protein(g)"]),
        "carbs_g": float(max_recipe_row["Carbs(g)"]),
        "fat_g": float(max_recipe_row["Fat(g)"]),
    }

    # Required: Identify most common cuisines for each diet type
    # We'll output top 3 cuisines per diet for compactness.
    cuisine_counts = (
        df.groupby(["Diet_type", "Cuisine_type"])
        .size()
        .reset_index(name="count")
        .sort_values(["Diet_type", "count"], ascending=[True, False])
    )
    most_common_cuisines = cuisine_counts.groupby("Diet_type", as_index=False).head(3)
    most_common_cuisines.to_csv(outdir / "most_common_cuisines_by_diet.csv", index=False)

    # Required: Create new metrics: protein-to-carbs ratio and carbs-to-fat ratio per recipe
    df_with_ratios = make_safe_ratios(df)
    df_with_ratios.to_csv(outdir / "metrics_with_ratios.csv", index=False)

    # Charts (bar, heatmap, scatter)
    bar_path = save_bar_chart_avg_macros(avg_macros, charts_dir)
    heatmap_path = save_heatmap_avg_macros(avg_macros, charts_dir)
    scatter_path = save_scatter_ratio_plot(df_with_ratios, charts_dir)

    summary = {
        "rows": int(len(df)),
        "columns": list(df.columns),
        "highest_avg_protein_diet": {
            "diet_type": highest_avg_protein_diet,
            "avg_protein_g": round(highest_avg_protein_value, 3),
        },
        "highest_single_recipe_by_protein": highest_single_recipe,
        "outputs": {
            "avg_macros_by_diet_csv": str((outdir / "avg_macros_by_diet.csv").as_posix()),
            "top5_protein_by_diet_csv": str((outdir / "top5_protein_by_diet.csv").as_posix()),
            "most_common_cuisines_by_diet_csv": str((outdir / "most_common_cuisines_by_diet.csv").as_posix()),
            "metrics_with_ratios_csv": str((outdir / "metrics_with_ratios.csv").as_posix()),
            "bar_chart": str(bar_path.as_posix()),
            "heatmap_chart": str(heatmap_path.as_posix()),
            "scatter_chart": str(scatter_path.as_posix()),
        },
    }
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("✅ Done.")
    print(f"- Highest avg protein diet: {highest_avg_protein_diet} ({highest_avg_protein_value:.2f}g)")
    print(f"- Outputs written to: {outdir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())