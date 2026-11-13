"""Explore the Iris flower dataset and save beginner-friendly charts."""

from pathlib import Path

import matplotlib

# A non-interactive backend lets the script save charts on computers without a GUI.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


PROJECT_DIR = Path(__file__).resolve().parent
SOURCE_FILE = PROJECT_DIR / "data" / "iris.data"
CSV_FILE = PROJECT_DIR / "iris.csv"
OUTPUT_DIR = PROJECT_DIR / "outputs"

FEATURES = [
    "sepal_length_cm",
    "sepal_width_cm",
    "petal_length_cm",
    "petal_width_cm",
]
DATA_COLUMNS = FEATURES + ["species"]


def load_data() -> pd.DataFrame:
    """Read the local UCI data file with Pandas and save a clean CSV copy."""
    if not SOURCE_FILE.exists():
        raise FileNotFoundError(
            f"Could not find {SOURCE_FILE}. Keep the included data/iris.data file."
        )

    # UCI's source file has no header row, so we provide clear column names here.
    iris = pd.read_csv(SOURCE_FILE, header=None, names=DATA_COLUMNS)
    iris = iris.dropna(how="all")  # The source file may end with a blank line.
    iris["species"] = iris["species"].str.replace("Iris-", "", regex=False).str.title()

    # NumPy checks that every measurement is a real, finite number.
    measurements = iris[FEATURES].to_numpy(dtype=float)
    if not np.isfinite(measurements).all():
        raise ValueError("The measurement columns contain a missing or invalid number.")

    CSV_FILE.parent.mkdir(parents=True, exist_ok=True)
    iris.to_csv(CSV_FILE, index=False)
    return iris


def print_overview(iris: pd.DataFrame) -> None:
    """Print a preview, dataset size, column types, and data quality checks."""
    print("\n1. First five rows")
    print(iris.head())

    print("\n2. Dataset shape (rows, columns)")
    print(iris.shape)

    print("\n3. Column information")
    iris.info()

    print("\n4. Missing values in each column")
    print(iris.isna().sum())

    print("\n5. Duplicate rows")
    print(f"{iris.duplicated().sum()} duplicate row(s)")


def print_statistics(iris: pd.DataFrame) -> None:
    """Show common summary statistics for every numeric measurement."""
    summary = iris[FEATURES].agg(["mean", "median", "std", "min", "max"])
    print("\n6. Measurement summary statistics")
    print(summary.round(2))

    print("\n7. Species-wise measurement means")
    species_means = iris.groupby("species", sort=True)[FEATURES].mean()
    print(species_means.round(2))

    print("\nNumber of flowers per species")
    print(iris["species"].value_counts().sort_index())


def save_charts(iris: pd.DataFrame) -> None:
    """Create and save histograms, scatter plots, box plots, bars, and heatmap."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="notebook")

    # One histogram per measurement helps show its range and distribution.
    for feature in FEATURES:
        fig, ax = plt.subplots(figsize=(7, 5))
        sns.histplot(data=iris, x=feature, hue="species", bins=12, element="step", ax=ax)
        ax.set_title(f"Distribution of {feature.replace('_', ' ').title()}")
        ax.set_xlabel("Measurement (cm)")
        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / f"histogram_{feature}.png", dpi=150)
        plt.close(fig)

    # The scatter plots compare petal size and sepal size, colored by species.
    scatter_pairs = [
        ("sepal_length_cm", "sepal_width_cm", "sepal_measurements_scatter.png"),
        ("petal_length_cm", "petal_width_cm", "petal_measurements_scatter.png"),
    ]
    for x_feature, y_feature, filename in scatter_pairs:
        fig, ax = plt.subplots(figsize=(7, 5))
        sns.scatterplot(
            data=iris,
            x=x_feature,
            y=y_feature,
            hue="species",
            s=65,
            ax=ax,
        )
        ax.set_title(f"{y_feature.replace('_', ' ').title()} vs. {x_feature.replace('_', ' ').title()}")
        ax.set_xlabel(f"{x_feature.replace('_', ' ').title()} (cm)")
        ax.set_ylabel(f"{y_feature.replace('_', ' ').title()} (cm)")
        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / filename, dpi=150)
        plt.close(fig)

    # A single grouped box plot compares all four measurements by species.
    long_iris = iris.melt(
        id_vars="species", value_vars=FEATURES, var_name="measurement", value_name="cm"
    )
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(data=long_iris, x="measurement", y="cm", hue="species", ax=ax)
    ax.set_title("Measurement ranges by species")
    ax.set_xlabel("Measurement")
    ax.set_ylabel("Length (cm)")
    ax.tick_params(axis="x", rotation=15)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "box_plots_by_species.png", dpi=150)
    plt.close(fig)

    # The bar chart shows how many examples of each species are in the dataset.
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.countplot(data=iris, x="species", hue="species", legend=False, ax=ax)
    ax.set_title("Number of flowers per species")
    ax.set_xlabel("Species")
    ax.set_ylabel("Number of flowers")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "species_counts.png", dpi=150)
    plt.close(fig)

    # Correlation describes how strongly numeric measurements move together.
    fig, ax = plt.subplots(figsize=(8, 6))
    correlation = iris[FEATURES].corr()
    sns.heatmap(correlation, annot=True, cmap="coolwarm", center=0, fmt=".2f", ax=ax)
    ax.set_title("Correlation between measurements")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "correlation_heatmap.png", dpi=150)
    plt.close(fig)


def main() -> None:
    """Run the full exploratory data analysis from start to finish."""
    iris = load_data()
    print(f"Loaded {len(iris)} rows from {SOURCE_FILE.relative_to(PROJECT_DIR)}")
    print(f"Saved a cleaned copy to {CSV_FILE.relative_to(PROJECT_DIR)}")
    print_overview(iris)
    print_statistics(iris)
    save_charts(iris)
    print(f"\n8. Charts saved in: {OUTPUT_DIR.relative_to(PROJECT_DIR)}")


if __name__ == "__main__":
    main()
