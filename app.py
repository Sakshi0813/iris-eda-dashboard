"""Interactive Streamlit dashboard for the local Iris EDA project."""

from pathlib import Path

import inspect
import os

# Keep numerical-library startup within modest memory limits on high-core PCs.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import altair as alt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

import ml_model


PROJECT_DIR = Path(__file__).resolve().parent
CSV_FILE = PROJECT_DIR / "iris.csv"
OUTPUT_DIR = PROJECT_DIR / "outputs"
SPECIES_COLUMN = "species"
MEASUREMENT_FEATURES = [
    "sepal_length_cm",
    "sepal_width_cm",
    "petal_length_cm",
    "petal_width_cm",
]
SPECIES_MISSING_LABEL = "(Missing species)"
SPECIES_COLORS = ["#2A9D8F", "#E9A23B", "#7568D8", "#D65A72", "#3977A8"]
PAGES = [
    "Overview",
    "Data explorer",
    "Statistical analysis",
    "Species comparison",
    "Advanced analysis",
    "Charts & gallery",
    "Machine Learning",
    "Iris Prediction",
]

st.set_page_config(
    page_title="Iris EDA | Data Explorer",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data(show_spinner="Loading the Iris dataset…")
def load_dataset(csv_path: str, modified_ns: int, file_size: int) -> tuple[pd.DataFrame, int]:
    """Load the CSV without changing it, and report numeric values that need coercion."""
    del modified_ns, file_size  # These values refresh cached data when the CSV changes.
    data = pd.read_csv(csv_path)
    missing_columns = [column for column in [SPECIES_COLUMN, *MEASUREMENT_FEATURES] if column not in data.columns]
    if missing_columns:
        raise ValueError("Missing required columns: " + ", ".join(missing_columns))

    # Work on an in-memory copy. The original iris.csv is never rewritten.
    data = data.copy()
    invalid_numeric_cells = 0
    for feature in MEASUREMENT_FEATURES:
        original_values = data[feature]
        numeric_values = pd.to_numeric(original_values, errors="coerce")
        invalid_numeric_cells += int((original_values.notna() & numeric_values.isna()).sum())
        data[feature] = numeric_values.replace([np.inf, -np.inf], np.nan)

    data[SPECIES_COLUMN] = (
        data[SPECIES_COLUMN]
        .astype("string")
        .str.replace("^Iris-", "", regex=True)
        .str.strip()
        .str.title()
    )
    data.loc[data[SPECIES_COLUMN] == "", SPECIES_COLUMN] = pd.NA
    return data, invalid_numeric_cells


@st.cache_resource(show_spinner="Loading the trained model…")
def load_dashboard_model(
    model_name: str,
    csv_modified_ns: int,
    csv_file_size: int,
    model_modified_ns: int,
    model_file_size: int,
) -> dict:
    """Cache the project-local trained artifact until the CSV or model changes."""
    del csv_modified_ns, csv_file_size, model_modified_ns, model_file_size
    return ml_model.load_saved_model(model_name, model_dir=PROJECT_DIR / "models", csv_path=CSV_FILE)


def feature_label(feature: str) -> str:
    """Return a readable axis or table label for a measurement column."""
    return feature.replace("_", " ").replace(" cm", "").title() + " (cm)"


def filter_species(data: pd.DataFrame, selected_species: list[str]) -> pd.DataFrame:
    """Return rows for selected species; keep every matching record, including duplicates."""
    if not selected_species:
        return data.iloc[0:0].copy()
    mask = data[SPECIES_COLUMN].isin(selected_species)
    if SPECIES_MISSING_LABEL in selected_species:
        mask = mask | data[SPECIES_COLUMN].isna()
    return data.loc[mask].copy()


def chart_data(data: pd.DataFrame) -> pd.DataFrame:
    """Prepare labels for charts without changing the table or source CSV."""
    result = data.copy()
    result["species_display"] = result[SPECIES_COLUMN].fillna(SPECIES_MISSING_LABEL).astype(str)
    return result


def csv_download_bytes(data: pd.DataFrame) -> bytes:
    """Encode the filtered table as a UTF-8 CSV download without an index column."""
    return data.to_csv(index=False).encode("utf-8")


def render_full_width(element, *args, **kwargs):
    """Use the current responsive width API with a fallback for older Streamlit."""
    parameters = inspect.signature(element).parameters
    if "width" in parameters:
        kwargs["width"] = "stretch"
    elif "use_container_width" in parameters:
        kwargs["use_container_width"] = True
    return element(*args, **kwargs)


def apply_theme() -> None:
    """Set a restrained, responsive dashboard theme with readable contrast."""
    st.markdown(
        """
        <style>
        .stApp { background: #f4f7fb; color: #17243b; }
        .block-container { max-width: 1500px; padding-top: 1.4rem; padding-bottom: 2.5rem; }
        [data-testid="stHeader"] { background: rgba(244,247,251,.92); }
        [data-testid="stSidebar"] { background: #17243b; }
        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"],
        [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] p,
        [data-testid="stSidebar"] span { color: #eaf0f7; }
        [data-testid="stSidebar"] [data-testid="stCaptionContainer"] { color: #b8c6d8; }
        .dashboard-hero { padding: 1.65rem 1.8rem; border-radius: 18px;
          background: linear-gradient(118deg,#17243b 0%,#28566b 66%,#2a9d8f 125%);
          color: white; margin: .2rem 0 1.2rem; }
        .dashboard-hero .eyebrow { color: #a8e2d7; font-size: .76rem; font-weight: 700;
          letter-spacing: .13em; text-transform: uppercase; margin-bottom: .5rem; }
        .dashboard-hero h1 { color: white; font-size: clamp(1.8rem, 3vw, 2.5rem);
          line-height: 1.15; margin: 0 0 .5rem; }
        .dashboard-hero p { color: #e5f0f3; font-size: 1rem; margin: 0; max-width: 850px; }
        .section-intro { color: #5e6d80; margin-top: -.45rem; margin-bottom: 1rem; }
        div[data-testid="stMetric"] { background: white; border: 1px solid #e4eaf1;
          padding: 1rem 1.05rem; border-radius: 14px; box-shadow: 0 4px 14px rgba(23,36,59,.045); }
        div[data-testid="stMetricLabel"] { color: #64748b; }
        div[data-testid="stMetricValue"] { color: #17243b; }
        div[data-testid="stDataFrame"] { border: 1px solid #e4eaf1; border-radius: 12px; overflow: hidden; }
        h2, h3 { color: #17243b; }
        @media (max-width: 700px) {
          .block-container { padding-left: 1rem; padding-right: 1rem; }
          .dashboard-hero { padding: 1.25rem; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def make_species_comparison(
    data: pd.DataFrame,
    features: list[str],
    species_domain: list[str],
    chart_type: str = "Grouped bar chart",
) -> alt.Chart:
    """Build an interactive grouped bar or dot chart of species averages."""
    frame = chart_data(data)
    means = (
        frame.groupby("species_display", dropna=False)[features]
        .mean()
        .reset_index()
        .melt(id_vars="species_display", var_name="feature", value_name="mean_cm")
    )
    means["feature_label"] = means["feature"].map(feature_label)
    color = alt.Color(
        "species_display:N",
        title="Species",
        scale=alt.Scale(domain=species_domain, range=SPECIES_COLORS[: max(1, len(species_domain))]),
    )
    selection = alt.selection_point(fields=["species_display"], bind="legend")
    common_encoding = {
        "x": alt.X("feature_label:N", title="Measurement", sort=[feature_label(f) for f in features]),
        "y": alt.Y("mean_cm:Q", title="Average (cm)"),
        "color": alt.condition(selection, color, alt.value("#d5dce5")),
        "tooltip": [
            alt.Tooltip("species_display:N", title="Species"),
            alt.Tooltip("feature_label:N", title="Measurement"),
            alt.Tooltip("mean_cm:Q", title="Average (cm)", format=".2f"),
        ],
    }
    if chart_type == "Dot plot":
        chart = alt.Chart(means).mark_point(filled=True, shape="diamond", size=150).encode(
            **common_encoding,
            detail=alt.Detail("species_display:N"),
        )
    else:
        chart = alt.Chart(means).mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4).encode(
            **common_encoding,
            xOffset=alt.XOffset("species_display:N", sort=species_domain),
        )
    return chart.add_params(selection).interactive().properties(height=360)


def make_species_distribution(data: pd.DataFrame, chart_type: str, species_domain: list[str]) -> alt.Chart:
    """Build a species count chart in bar or donut form."""
    counts = (
        chart_data(data)["species_display"]
        .value_counts()
        .rename_axis("species_display")
        .reset_index(name="records")
    )
    color = alt.Color(
        "species_display:N",
        title="Species",
        scale=alt.Scale(domain=species_domain, range=SPECIES_COLORS[: max(1, len(species_domain))]),
    )
    if chart_type == "Donut chart":
        return (
            alt.Chart(counts)
            .mark_arc(innerRadius=70, outerRadius=130, stroke="white", strokeWidth=2)
            .encode(
                theta=alt.Theta("records:Q", title="Records"),
                color=color,
                tooltip=[alt.Tooltip("species_display:N", title="Species"), alt.Tooltip("records:Q", title="Records")],
            )
            .properties(height=380)
        )
    return (
        alt.Chart(counts)
        .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
        .encode(
            x=alt.X("species_display:N", title="Species", sort=species_domain),
            y=alt.Y("records:Q", title="Number of records"),
            color=color,
            tooltip=[alt.Tooltip("species_display:N", title="Species"), alt.Tooltip("records:Q", title="Records")],
        )
        .properties(height=380)
        .interactive()
    )


def feature_summary_table(data: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Calculate descriptive statistics and distribution shape for each feature."""
    if data.empty or not features:
        return pd.DataFrame()
    summary = data[features].describe(percentiles=[0.25, 0.50, 0.75]).T
    summary["median"] = data[features].median()
    summary["iqr"] = summary["75%"] - summary["25%"]
    summary["skewness"] = data[features].skew()
    summary = summary[["count", "mean", "median", "std", "min", "25%", "50%", "75%", "max", "iqr", "skewness"]]
    summary.index = [feature_label(feature) for feature in summary.index]
    summary.index.name = "Feature"
    return summary.reset_index().rename(columns={
        "count": "Count", "mean": "Mean", "median": "Median",
        "std": "Standard deviation", "min": "Minimum", "25%": "25th percentile",
        "50%": "50th percentile", "75%": "75th percentile", "max": "Maximum",
        "iqr": "IQR", "skewness": "Skewness",
    })


def species_statistics(data: pd.DataFrame, features: list[str], statistic: str) -> pd.DataFrame:
    """Calculate a selected statistic for every species in the current data view."""
    if data.empty or not features:
        return pd.DataFrame()
    frame = chart_data(data)
    grouped = frame.groupby("species_display", dropna=False)[features]
    calculations = {
        "Mean": grouped.mean,
        "Median": grouped.median,
        "Standard deviation": grouped.std,
        "Minimum": grouped.min,
        "Maximum": grouped.max,
        "25th percentile": lambda: grouped.quantile(0.25),
        "75th percentile": lambda: grouped.quantile(0.75),
    }
    result = calculations[statistic]()
    result.columns = [feature_label(feature) for feature in result.columns]
    result.index.name = "Species"
    return result.reset_index()


def iqr_outlier_summary(data: pd.DataFrame, features: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return per-feature IQR bounds and the original rows outside those bounds."""
    summaries = []
    outlier_records = []
    for feature in features:
        values = data[feature].dropna()
        if values.empty:
            summaries.append({
                "Feature": feature_label(feature), "Count": 0, "Q1": np.nan,
                "Q3": np.nan, "IQR": np.nan, "Lower bound": np.nan,
                "Upper bound": np.nan, "Potential outliers": 0,
            })
            continue

        q1, q3 = values.quantile([0.25, 0.75])
        spread = q3 - q1
        lower_bound = q1 - 1.5 * spread
        upper_bound = q3 + 1.5 * spread
        mask = data[feature].notna() & ((data[feature] < lower_bound) | (data[feature] > upper_bound))
        summaries.append({
            "Feature": feature_label(feature), "Count": int(values.count()),
            "Q1": q1, "Q3": q3, "IQR": spread,
            "Lower bound": lower_bound, "Upper bound": upper_bound,
            "Potential outliers": int(mask.sum()),
        })
        if mask.any():
            rows = pd.DataFrame({
                "CSV row index": data.index[mask],
                "Species": data.loc[mask, SPECIES_COLUMN].fillna(SPECIES_MISSING_LABEL).astype(str).to_numpy(),
                "Feature": feature_label(feature),
                "Value": data.loc[mask, feature].to_numpy(),
                "Lower bound": lower_bound,
                "Upper bound": upper_bound,
            })
            outlier_records.append(rows)

    summary_frame = pd.DataFrame(summaries)
    records_frame = pd.concat(outlier_records, ignore_index=True) if outlier_records else pd.DataFrame(
        columns=["CSV row index", "Species", "Feature", "Value", "Lower bound", "Upper bound"]
    )
    return summary_frame, records_frame


def key_insights(data: pd.DataFrame, features: list[str]) -> dict[str, str]:
    """Calculate headline relationships and observations from the filtered dataset."""
    insights: dict[str, str] = {}
    if data.empty:
        return insights

    if features:
        averages = data[features].mean().dropna()
        if not averages.empty:
            feature = averages.idxmax()
            insights["Highest average feature"] = f"{feature_label(feature)} · {averages[feature]:.2f} cm"

        if len(features) >= 2:
            matrix = data[features].corr()
            pairs = [
                (left, right, matrix.loc[left, right])
                for index, left in enumerate(features)
                for right in features[index + 1:]
                if pd.notna(matrix.loc[left, right])
            ]
            positive_pairs = [pair for pair in pairs if pair[2] > 0]
            if positive_pairs:
                left, right, value = max(positive_pairs, key=lambda pair: pair[2])
                insights["Strongest positive correlation"] = f"{feature_label(left)} × {feature_label(right)} · r = {value:.2f}"
            else:
                insights["Strongest positive correlation"] = "No positive pairwise correlation in this view"
            if pairs:
                left, right, value = min(pairs, key=lambda pair: abs(pair[2]))
                insights["Weakest correlation"] = f"{feature_label(left)} × {feature_label(right)} · r = {value:.2f}"
            else:
                insights["Weakest correlation"] = "Not enough valid feature pairs"

    if "petal_length_cm" in data.columns:
        petal_means = chart_data(data).groupby("species_display")["petal_length_cm"].mean().dropna()
        if not petal_means.empty:
            species = petal_means.idxmax()
            insights["Highest mean petal length"] = f"{species} · {petal_means[species]:.2f} cm"

    counts = chart_data(data)["species_display"].value_counts()
    if not counts.empty:
        if counts.nunique() == 1:
            insights["Species distribution"] = f"Balanced · {int(counts.iloc[0])} rows per selected species"
        else:
            insights["Species distribution"] = f"Largest group · {counts.index[0]} ({int(counts.iloc[0])} rows)"
    return insights


def render_pair_plot(data: pd.DataFrame, features: list[str], species_domain: list[str]) -> None:
    """Render a Seaborn pair plot without changing any source records."""
    if len(features) < 2:
        st.info("Select at least two numerical features in the sidebar to build a pair plot.")
        return
    if data[features].notna().sum().sum() == 0:
        st.info("There are no valid numeric measurements for the selected features.")
        return

    frame = chart_data(data)[["species_display", *features]]
    palette = {
        species: SPECIES_COLORS[index % len(SPECIES_COLORS)]
        for index, species in enumerate(species_domain)
    }
    try:
        pair_grid = sns.pairplot(
            frame,
            vars=features,
            hue="species_display",
            hue_order=species_domain or None,
            palette=palette or None,
            corner=True,
            diag_kind="hist",
            height=2.2,
            aspect=1.05,
            plot_kws={"alpha": 0.75, "s": 32, "edgecolor": "white", "linewidth": 0.25},
            diag_kws={"bins": 12},
        )
    except (ValueError, TypeError) as error:
        st.info(f"The pair plot is unavailable for this selection: {error}")
        return
    render_full_width(st.pyplot, pair_grid.figure)
    plt.close(pair_grid.figure)


def render_metrics(data: pd.DataFrame, total_rows: int, selected_features: list[str]) -> None:
    """Show data quality and size metrics for the current species selection."""
    species_count = int(data[SPECIES_COLUMN].nunique(dropna=True))
    if data[SPECIES_COLUMN].isna().any():
        species_count += 1
    metrics = st.columns(5)
    metrics[0].metric("Records in view", f"{len(data):,}", help=f"Selected rows out of {total_rows:,} source records. Duplicates are preserved.")
    metrics[1].metric("Species in view", species_count, help="Counts observed species in the current selection, including a missing label if present.")
    metrics[2].metric("Numeric features", len(selected_features), help="Measurement features selected in the sidebar.")
    metrics[3].metric("Missing cells", f"{int(data.isna().sum().sum()):,}", help="Missing values across all columns in the current species selection.")
    metrics[4].metric("Duplicate rows", f"{int(data.duplicated().sum()):,}", help="Repeated copies within the current selection, counted across all columns. None are removed.")


def show_no_data_message(data: pd.DataFrame, selected_features: list[str]) -> bool:
    """Display guidance when filters leave no records or no numeric features."""
    if data.empty:
        st.info("No rows match the current species selection. Choose one or more species in the sidebar.")
        return True
    if not selected_features:
        st.info("Select at least one numerical feature in the sidebar to view statistics and charts.")
        return True
    return False


def render_existing_chart_gallery() -> None:
    """Display the project's saved EDA images without regenerating them."""
    st.markdown("---")
    st.markdown("### Existing EDA chart gallery")
    st.markdown("These are the original Matplotlib/Seaborn images in `outputs/`; the dashboard displays them without regenerating or changing them.")
    if not OUTPUT_DIR.is_dir():
        st.info("The project's `outputs` folder was not found.")
        return

    image_files = sorted(
        path for path in OUTPUT_DIR.iterdir()
        if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}
    )
    if not image_files:
        st.info("No chart images were found in the outputs folder.")
        return

    gallery_columns = st.columns(3)
    for index, image_path in enumerate(image_files):
        with gallery_columns[index % 3]:
            title = image_path.stem.replace("_", " ").replace("cm", "").title()
            render_full_width(st.image, str(image_path), caption=title)


apply_theme()

if not CSV_FILE.is_file():
    st.error(f"Dataset not found: `{CSV_FILE}`. Keep `iris.csv` beside `app.py`.")
    st.stop()

try:
    iris, invalid_numeric_cells = load_dataset(
        str(CSV_FILE), CSV_FILE.stat().st_mtime_ns, CSV_FILE.stat().st_size
    )
except (OSError, pd.errors.ParserError, UnicodeError, ValueError) as error:
    st.error(f"The Iris dataset could not be loaded: {error}")
    st.stop()

available_species = sorted(iris[SPECIES_COLUMN].dropna().astype(str).unique().tolist())
if iris[SPECIES_COLUMN].isna().any():
    available_species.append(SPECIES_MISSING_LABEL)

st.sidebar.markdown("# 🌸 Iris Explorer")
st.sidebar.caption("A clear, interactive view of the flower measurements")
page = st.sidebar.radio("Navigate", PAGES, key="page_navigation", label_visibility="collapsed")
st.sidebar.markdown("---")

reset_keys = [
    "species_filter", "feature_filter", "hist_feature", "hist_bins",
    "scatter_x", "scatter_y", "box_feature", "box_whiskers", "violin_feature",
    "species_statistic", "species_compare_chart_type", "species_distribution_chart_type",
    "visual_species_distribution_type",
]
if st.sidebar.button("Reset filters", type="secondary"):
    for key in reset_keys:
        st.session_state.pop(key, None)
    st.rerun()

selected_species = st.sidebar.multiselect(
    "Select Iris species",
    options=available_species,
    default=available_species,
    key="species_filter",
    help="Choose one or more species. Duplicate records remain included.",
)
selected_features = st.sidebar.multiselect(
    "Numerical features",
    options=MEASUREMENT_FEATURES,
    default=MEASUREMENT_FEATURES,
    format_func=feature_label,
    key="feature_filter",
    help="Choose which measurements appear in summaries and feature-based charts.",
)
st.sidebar.caption("Measurement units: centimeters")

filtered = filter_species(iris, selected_species)
filtered_chart = chart_data(filtered)
species_domain = [
    label for label in available_species
    if label in selected_species
]
if not species_domain and selected_species:
    species_domain = selected_species

st.markdown(
    "<div class='dashboard-hero'><div class='eyebrow'>Exploratory data analysis</div>"
    "<h1>Iris Flower Data Explorer</h1>"
    "<p>Explore the dataset, compare species, and inspect measurement patterns with interactive charts."
    " All views respond to the sidebar filters.</p></div>",
    unsafe_allow_html=True,
)
st.caption(f"Current view: {len(filtered):,} of {len(iris):,} records · {len(selected_features)} selected measurement features")
render_metrics(filtered, len(iris), selected_features)

if invalid_numeric_cells:
    st.warning(
        f"{invalid_numeric_cells} non-empty measurement value(s) could not be read as numbers. "
        "They are treated as missing in this dashboard; the source CSV was not changed."
    )

if page == "Overview":
    st.subheader("Dataset overview")
    st.markdown("A quick summary of the data currently selected in the sidebar.", help="The default view includes all rows and all three species.")
    if not filtered.empty:
        left, right = st.columns([1.1, 1])
        with left:
            st.markdown("#### Filtered dataset preview")
            render_full_width(st.dataframe, filtered.head(8), hide_index=True)
            st.caption("Preview of the first eight matching rows; no duplicate records were removed.")
        with right:
            st.markdown("#### Species distribution")
            counts = filtered_chart["species_display"].value_counts().rename_axis("species_display").reset_index(name="records")
            count_chart = (
                alt.Chart(counts)
                .mark_arc(innerRadius=62, outerRadius=118, stroke="white", strokeWidth=2)
                .encode(
                    theta=alt.Theta("records:Q", title="Records"),
                    color=alt.Color("species_display:N", title="Species", scale=alt.Scale(domain=species_domain, range=SPECIES_COLORS[: max(1, len(species_domain))])),
                    tooltip=[alt.Tooltip("species_display:N", title="Species"), alt.Tooltip("records:Q", title="Records")],
                )
                .properties(height=300)
            )
            render_full_width(st.altair_chart, count_chart)
        st.markdown("#### Dataset structure and quality")
        quality = pd.DataFrame({
            "Check": ["Rows in source", "Rows in current view", "Columns", "Missing cells in view", "Duplicate rows in view", "Species in source"],
            "Result": [len(iris), len(filtered), len(iris.columns), int(filtered.isna().sum().sum()), int(filtered.duplicated().sum()), int(iris[SPECIES_COLUMN].nunique(dropna=True))],
        })
        render_full_width(st.dataframe, quality, hide_index=True)
        st.caption("The dashboard reads `iris.csv` from the same folder as this app and never writes back to it.")
    else:
        st.info("Choose at least one species in the sidebar to see the overview.")

elif page == "Data explorer":
    st.subheader("Filtered data explorer")
    st.markdown("Inspect matching records, column types, missing values, and duplicates.")
    render_full_width(st.dataframe, filtered, hide_index=True)
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("#### Column profile")
        profile = pd.DataFrame({
            "Column": filtered.columns,
            "Data type": [str(dtype) for dtype in filtered.dtypes],
            "Non-missing values": filtered.notna().sum().values,
            "Missing values": filtered.isna().sum().values,
        })
        render_full_width(st.dataframe, profile, hide_index=True)
    with col_b:
        st.markdown("#### Data quality notes")
        st.write(f"**Missing cells:** {int(filtered.isna().sum().sum())}")
        st.write(f"**Duplicate rows in this view:** {int(filtered.duplicated().sum())}")
        st.write(f"**Invalid numeric values converted for display:** {invalid_numeric_cells}")
        st.caption("Missing or invalid values are shown as missing for analysis. The source CSV and repeated records are preserved.")
    render_full_width(st.download_button, 
        "Download filtered dataset (CSV)",
        data=csv_download_bytes(filtered),
        file_name="iris_filtered.csv",
        mime="text/csv",
        type="primary",
        help="Downloads all columns for the currently selected species.",
    )

elif page == "Statistical analysis":
    st.subheader("Descriptive statistics")
    st.markdown("Review center, spread, quartiles, and distribution shape for each selected measurement.")
    if not show_no_data_message(filtered, selected_features):
        feature_stats = feature_summary_table(filtered, selected_features)
        st.markdown("#### Feature distribution profile")
        render_full_width(st.dataframe, feature_stats.round(3), hide_index=True)
        st.caption("Includes mean, median, sample standard deviation, minimum, maximum, 25th/50th/75th percentiles, IQR, and skewness. Missing values are excluded feature by feature.")
        render_full_width(st.download_button,
            "Download feature statistics (CSV)",
            data=feature_stats.to_csv(index=False).encode("utf-8"),
            file_name="iris_feature_statistics.csv",
            mime="text/csv",
            type="primary",
        )

        st.markdown("#### Species-wise statistical comparison")
        statistic = st.selectbox(
            "Statistic to compare",
            ["Mean", "Median", "Standard deviation", "Minimum", "Maximum", "25th percentile", "75th percentile"],
            key="species_statistic",
        )
        species_stats = species_statistics(filtered, selected_features, statistic)
        render_full_width(st.dataframe, species_stats.round(3), hide_index=True)
        render_full_width(st.download_button,
            "Download species comparison (CSV)",
            data=species_stats.to_csv(index=False).encode("utf-8"),
            file_name="iris_species_comparison.csv",
            mime="text/csv",
        )
        st.caption("The selected statistic is recalculated for the filtered species and measurements.")

elif page == "Species comparison":
    st.subheader("Compare species measurements")
    st.markdown("Compare species counts and measurement averages. Use the sidebar to choose species and features.")
    if not show_no_data_message(filtered, selected_features):
        distribution_type = st.selectbox(
            "Species distribution chart", ["Bar chart", "Donut chart"], key="species_distribution_chart_type"
        )
        st.markdown("#### Species distribution")
        distribution_chart = make_species_distribution(filtered, distribution_type, species_domain)
        render_full_width(st.altair_chart, distribution_chart)
        st.caption("Counts use the current species selection; repeated records remain included.")
        comparison_type = st.selectbox(
            "Feature comparison chart", ["Grouped bar chart", "Dot plot"], key="species_compare_chart_type"
        )
        st.markdown("#### Feature comparison")
        comparison_chart = make_species_comparison(filtered, selected_features, species_domain, comparison_type)
        render_full_width(st.altair_chart, comparison_chart)
        means = species_statistics(filtered, selected_features, "Mean")
        render_full_width(st.dataframe, means.round(3), hide_index=True)
        st.caption("Hover for exact values; click the chart legend to focus on a species.")

elif page == "Advanced analysis":
    st.subheader("Key Insights")
    st.markdown("Calculated from the current filters. Insights update when selected species or features change.")
    st.caption("Weakest correlation means the feature pair closest to zero (the smallest absolute correlation).")
    if filtered.empty:
        st.info("Select at least one species to calculate insights and outlier bounds.")
    else:
        insights = key_insights(filtered, selected_features)
        if insights:
            insight_columns = st.columns(min(4, len(insights)))
            for index, (label, value) in enumerate(insights.items()):
                insight_columns[index % len(insight_columns)].metric(label, value)

        observations = []
        current_counts = chart_data(filtered)["species_display"].value_counts()
        if not current_counts.empty:
            if current_counts.nunique() == 1:
                observations.append(f"The filtered sample is balanced, with {int(current_counts.iloc[0])} rows in each selected species.")
            else:
                observations.append(f"{current_counts.index[0]} is the largest selected group with {int(current_counts.iloc[0])} rows.")
        petal_means = chart_data(filtered).groupby("species_display")["petal_length_cm"].mean().dropna()
        if len(petal_means) >= 2:
            highest = petal_means.idxmax()
            lowest = petal_means.idxmin()
            observations.append(f"Mean petal length spans {petal_means[lowest]:.2f}–{petal_means[highest]:.2f} cm, from {lowest} to {highest}.")
        if selected_features:
            profile = feature_summary_table(filtered, selected_features).dropna(subset=["IQR"])
            if not profile.empty:
                widest = profile.loc[profile["IQR"].idxmax()]
                observations.append(f"{widest['Feature']} has the widest middle 50% range in this view (IQR {widest['IQR']:.2f} cm).")
        if observations:
            st.markdown("#### Observations from the current charts")
            for observation in observations:
                st.markdown(f"- {observation}")

    st.markdown("---")
    st.markdown("#### Potential outlier analysis (IQR method)")
    st.markdown("The interquartile range is Q3 − Q1. A measurement is flagged as a potential outlier when it is below Q1 − 1.5 × IQR or above Q3 + 1.5 × IQR.")
    st.caption("Bounds and counts use the current filtered rows. A flag is a review prompt, not a cleaning instruction; no rows or values are removed.")
    if filtered.empty:
        st.info("No rows are available for outlier analysis under the current filters.")
    elif not selected_features:
        st.info("Select one or more numerical features in the sidebar to calculate outlier bounds.")
    else:
        outlier_summary, potential_outliers = iqr_outlier_summary(filtered, selected_features)
        render_full_width(st.dataframe, outlier_summary.round(3), hide_index=True)
        feature_stats = feature_summary_table(filtered, selected_features)
        analysis_results = feature_stats.merge(
            outlier_summary[["Feature", "Lower bound", "Upper bound", "Potential outliers"]],
            on="Feature",
            how="left",
        )
        render_full_width(st.download_button,
            "Download analysis results (CSV)",
            data=analysis_results.to_csv(index=False).encode("utf-8"),
            file_name="iris_analysis_results.csv",
            mime="text/csv",
            type="primary",
        )
        if not potential_outliers.empty:
            render_full_width(st.download_button,
                "Download potential outlier rows (CSV)",
                data=potential_outliers.to_csv(index=False).encode("utf-8"),
                file_name="iris_potential_outliers.csv",
                mime="text/csv",
            )
        with st.expander(f"Show potential outlier records ({len(potential_outliers)})"):
            if potential_outliers.empty:
                st.success("No potential outlier records were identified for these selected features.")
            else:
                render_full_width(st.dataframe, potential_outliers.round(3), hide_index=True)
                st.caption("CSV row index is zero-based. Every flagged source row is shown, including repeated records.")

elif page == "Machine Learning":
    st.subheader("Machine Learning model evaluation")
    st.markdown(
        "Two classifiers are trained on a reproducible, stratified train/test split. "
        "The held-out test rows are used only for evaluation, not for fitting either model."
    )
    st.info(
        "The split keeps exact duplicate records together, while retaining all source rows. "
        "Preprocessing is inside each training pipeline and is fitted only on the training fold. "
        "To train or refresh the saved models, run `python ml_model.py` from the project folder."
    )
    artifacts = {}
    model_errors = []
    for model_name, filename in ml_model.MODEL_FILES.items():
        artifact_path = PROJECT_DIR / "models" / filename
        if not artifact_path.is_file():
            model_errors.append(f"{model_name}: saved model is not present yet.")
            continue
        try:
            artifacts[model_name] = load_dashboard_model(
                model_name,
                CSV_FILE.stat().st_mtime_ns,
                CSV_FILE.stat().st_size,
                artifact_path.stat().st_mtime_ns,
                artifact_path.stat().st_size,
            )
        except (OSError, ValueError, EOFError) as error:
            model_errors.append(f"{model_name}: {error}")
    for message in model_errors:
        st.warning(message)

    if artifacts:
        first_artifact = next(iter(artifacts.values()))
        st.caption(
            f"Dataset rows: {first_artifact['dataset_rows']} · "
            f"Training rows: {first_artifact['train_rows']} · "
            f"Test rows: {first_artifact['test_rows']} · "
            f"Duplicate rows retained: {first_artifact['duplicate_rows_retained']} · "
            f"Random state: {first_artifact['random_state']}"
        )
        comparison_rows = []
        for model_name, artifact in artifacts.items():
            metrics = artifact["metrics"]
            comparison_rows.append({
                "Model": model_name,
                "Accuracy": metrics["accuracy"],
                "Precision (macro)": metrics["precision_macro"],
                "Recall (macro)": metrics["recall_macro"],
                "F1-score (macro)": metrics["f1_macro"],
            })
        comparison = pd.DataFrame(comparison_rows)
        st.markdown("#### Test-set performance comparison")
        render_full_width(st.dataframe, comparison.style.format({
            "Accuracy": "{:.3f}", "Precision (macro)": "{:.3f}",
            "Recall (macro)": "{:.3f}", "F1-score (macro)": "{:.3f}",
        }), hide_index=True)
        metric_long = comparison.melt(id_vars="Model", var_name="Metric", value_name="Score")
        score_chart = alt.Chart(metric_long).mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4).encode(
            x=alt.X("Metric:N", title=None, axis=alt.Axis(labelAngle=-20)),
            y=alt.Y("Score:Q", title="Test score", scale=alt.Scale(domain=[0, 1])),
            color=alt.Color("Model:N", scale=alt.Scale(range=["#3977A8", "#2A9D8F"])),
            xOffset="Model:N",
            tooltip=["Model", "Metric", alt.Tooltip("Score:Q", format=".3f")],
        ).properties(height=320)
        render_full_width(st.altair_chart, score_chart)

        evaluation_tabs = st.tabs(list(artifacts))
        for tab, (model_name, artifact) in zip(evaluation_tabs, artifacts.items()):
            with tab:
                metrics = artifact["metrics"]
                report = pd.DataFrame(metrics["classification_report"]).T
                st.markdown(f"#### {model_name} classification report")
                render_full_width(st.dataframe, report.style.format("{:.3f}"), hide_index=False)
                matrix = np.asarray(metrics["confusion_matrix"])
                matrix_data = pd.DataFrame(matrix, index=metrics["class_labels"], columns=metrics["class_labels"])
                st.markdown("#### Confusion matrix")
                st.caption("Rows are actual species; columns are predicted species. Correct predictions appear on the diagonal.")
                render_full_width(st.dataframe, matrix_data, hide_index=False)
                matrix_chart_data = matrix_data.rename_axis("Actual").reset_index().melt(
                    id_vars="Actual", var_name="Predicted", value_name="Records"
                )
                matrix_chart = alt.Chart(matrix_chart_data).mark_rect().encode(
                    x=alt.X("Predicted:N", title="Predicted species"),
                    y=alt.Y("Actual:N", title="Actual species"),
                    color=alt.Color("Records:Q", scale=alt.Scale(scheme="blues")),
                    tooltip=["Actual", "Predicted", "Records"],
                )
                matrix_text = alt.Chart(matrix_chart_data).mark_text(fontSize=15, fontWeight="bold").encode(
                    x="Predicted:N", y="Actual:N", text="Records:Q",
                    color=alt.condition("datum.Records > 0", alt.value("white"), alt.value("#17243b")),
                )
                render_full_width(st.altair_chart, (matrix_chart + matrix_text).properties(height=260))

        if "Random Forest" in artifacts:
            st.markdown("#### Random Forest feature importance")
            rf_classifier = artifacts["Random Forest"]["pipeline"].named_steps["classifier"]
            importance = pd.DataFrame({
                "Feature": [feature_label(name) for name in ml_model.FEATURES],
                "Importance": rf_classifier.feature_importances_,
            }).sort_values("Importance", ascending=False)
            render_full_width(st.dataframe, importance.style.format({"Importance": "{:.3f}"}), hide_index=True)
            importance_chart = alt.Chart(importance).mark_bar(cornerRadiusEnd=4, color="#2A9D8F").encode(
                x=alt.X("Importance:Q", title="Relative importance"),
                y=alt.Y("Feature:N", sort="-x", title=None),
                tooltip=["Feature", alt.Tooltip("Importance:Q", format=".3f")],
            ).properties(height=240)
            render_full_width(st.altair_chart, importance_chart)
            most_important = importance.iloc[0]
            st.caption(
                f"The forest assigns the largest relative importance to {most_important['Feature']} "
                f"({most_important['Importance']:.3f}). Importances describe this fitted model; "
                "they do not prove that a feature causes a species difference."
            )
        st.caption("Macro precision, recall and F1 give each species equal weight. Scores summarize this one held-out split and do not guarantee future performance.")

elif page == "Iris Prediction":
    st.subheader("Iris Prediction")
    st.markdown(
        "Enter four flower measurements in centimeters. The saved classifier returns an estimate "
        "of the species based on patterns learned from the project dataset."
    )
    model_choice = st.selectbox("Model", list(ml_model.MODEL_FILES), key="prediction_model")
    model_filename = ml_model.MODEL_FILES[model_choice]
    model_path = PROJECT_DIR / "models" / model_filename
    if not model_path.is_file():
        st.warning("No trained model is available yet. From this project folder, run `python ml_model.py` and reopen this page.")
    else:
        try:
            artifact = load_dashboard_model(
                model_choice,
                CSV_FILE.stat().st_mtime_ns,
                CSV_FILE.stat().st_size,
                model_path.stat().st_mtime_ns,
                model_path.stat().st_size,
            )
        except (OSError, ValueError, EOFError) as error:
            st.error(f"The saved model could not be loaded: {error}")
            artifact = None
        if artifact is not None:
            st.caption("Input limits use the minimum and maximum values in iris.csv. A prediction is an estimate, not a guarantee of correctness.")
            with st.form("iris_prediction_form"):
                columns = st.columns(2)
                values = {}
                for index, feature in enumerate(ml_model.FEATURES):
                    lower, upper = artifact["input_bounds"][feature]
                    default = float(iris[feature].median())
                    values[feature] = columns[index % 2].number_input(
                        feature_label(feature), min_value=lower, max_value=upper,
                        value=min(max(default, lower), upper), step=0.1,
                        format="%.1f", key=f"prediction_{feature}",
                    )
                submitted = st.form_submit_button("Predict species", type="primary")
            if submitted:
                try:
                    prediction = ml_model.predict_species(artifact, values)
                    st.success(f"Predicted species: **{prediction}**")
                    st.caption(f"Model used: {model_choice}. This result is a model estimate and may be incorrect.")
                except (TypeError, ValueError) as error:
                    st.error(f"Please check the measurement values: {error}")

else:
    st.subheader("Interactive visualizations")
    st.markdown("Select chart settings below. Hover for values and use chart gestures to zoom or pan.")
    if show_no_data_message(filtered, selected_features):
        render_existing_chart_gallery()
        st.stop()

    histogram_tab, scatter_tab, box_tab, violin_tab, pair_tab, correlation_tab, species_tab = st.tabs(
        ["Histogram", "Scatter plot", "Box plot", "Violin plot", "Pair plot", "Correlation heatmap", "Species distribution"]
    )
    with histogram_tab:
        h1, h2 = st.columns([1, 1])
        hist_feature = h1.selectbox(
            "Measurement",
            selected_features,
            format_func=feature_label,
            key="hist_feature",
        )
        hist_bins = h2.slider("Histogram bins", min_value=5, max_value=30, value=12, key="hist_bins")
        hist = (
            alt.Chart(filtered_chart)
            .mark_bar(opacity=0.72, binSpacing=1)
            .encode(
                x=alt.X(f"{hist_feature}:Q", bin=alt.Bin(maxbins=hist_bins), title=feature_label(hist_feature)),
                y=alt.Y("count():Q", title="Number of records", stack=None),
                color=alt.Color("species_display:N", title="Species", scale=alt.Scale(domain=species_domain, range=SPECIES_COLORS[: max(1, len(species_domain))])),
                tooltip=[alt.Tooltip("species_display:N", title="Species"), alt.Tooltip("count():Q", title="Records")],
            )
            .properties(height=380)
            .interactive()
        )
        render_full_width(st.altair_chart, hist)
        st.caption("The histogram groups measurements into ranges. Color separates species; changing the feature or bin count updates the chart.")

    with scatter_tab:
        if len(selected_features) < 2:
            st.info("Select at least two numerical features in the sidebar to compare them in a scatter plot.")
        else:
            scatter_left, scatter_right = st.columns(2)
            scatter_x = scatter_left.selectbox(
                "Horizontal feature", selected_features,
                index=min(2, len(selected_features) - 1),
                format_func=feature_label, key="scatter_x",
            )
            y_options = [feature for feature in selected_features if feature != scatter_x]
            default_y = next((feature for feature in ["petal_width_cm", "sepal_width_cm"] if feature in y_options), y_options[0])
            scatter_y = scatter_right.selectbox(
                "Vertical feature", y_options, index=y_options.index(default_y),
                format_func=feature_label, key="scatter_y",
            )
            selection = alt.selection_point(fields=["species_display"], bind="legend")
            scatter = (
                alt.Chart(filtered_chart)
                .mark_circle(size=92, opacity=0.82, stroke="white", strokeWidth=0.8)
                .encode(
                    x=alt.X(f"{scatter_x}:Q", title=feature_label(scatter_x)),
                    y=alt.Y(f"{scatter_y}:Q", title=feature_label(scatter_y)),
                    color=alt.Color("species_display:N", title="Species", scale=alt.Scale(domain=species_domain, range=SPECIES_COLORS[: max(1, len(species_domain))])),
                    opacity=alt.condition(selection, alt.value(0.88), alt.value(0.08)),
                    tooltip=[alt.Tooltip("species_display:N", title="Species"), alt.Tooltip(f"{scatter_x}:Q", title=feature_label(scatter_x), format=".2f"), alt.Tooltip(f"{scatter_y}:Q", title=feature_label(scatter_y), format=".2f")],
                )
                .add_params(selection)
                .properties(height=420)
                .interactive()
            )
            render_full_width(st.altair_chart, scatter)
            st.caption("Each point is one flower. Hover for measurements, use the legend to focus species, and scroll or drag to zoom and pan.")

    with box_tab:
        box_feature = st.selectbox(
            "Measurement", selected_features,
            format_func=feature_label, key="box_feature",
        )
        whisker_rule = st.selectbox(
            "Whisker rule",
            ["1.5 × IQR (flag potential outliers)", "Full minimum to maximum"],
            key="box_whiskers",
        )
        box = (
            alt.Chart(filtered_chart)
            .mark_boxplot(extent=1.5 if whisker_rule.startswith("1.5") else "min-max", size=56)
            .encode(
                x=alt.X("species_display:N", title="Species", sort=species_domain),
                y=alt.Y(f"{box_feature}:Q", title=feature_label(box_feature)),
                color=alt.Color("species_display:N", title="Species", scale=alt.Scale(domain=species_domain, range=SPECIES_COLORS[: max(1, len(species_domain))]), legend=None),
                tooltip=[alt.Tooltip("species_display:N", title="Species")],
            )
            .properties(height=400)
            .interactive()
        )
        render_full_width(st.altair_chart, box)
        st.caption("The center line is the median and the box spans the middle half of values. With the 1.5 × IQR rule, points beyond the whiskers indicate potential outliers.")

    with violin_tab:
        violin_feature = st.selectbox(
            "Measurement", selected_features,
            format_func=feature_label, key="violin_feature",
        )
        violin_fig, violin_ax = plt.subplots(figsize=(9, 5))
        violin_frame = filtered_chart.dropna(subset=[violin_feature])
        if violin_frame.empty:
            st.info("No valid values are available for this measurement in the current view.")
        else:
            violin_palette = {
                species: SPECIES_COLORS[index % len(SPECIES_COLORS)]
                for index, species in enumerate(species_domain)
            }
            sns.violinplot(
                data=violin_frame,
                x="species_display",
                y=violin_feature,
                hue="species_display",
                order=species_domain,
                hue_order=species_domain,
                palette=violin_palette or None,
                dodge=False,
                inner="quart",
                cut=0,
                legend=False,
                ax=violin_ax,
            )
            violin_ax.set_title(f"{feature_label(violin_feature)} distribution by species", loc="left", fontweight="bold")
            violin_ax.set_xlabel("Species")
            violin_ax.set_ylabel(feature_label(violin_feature))
            violin_ax.grid(axis="y", color="#e8edf3", linewidth=0.8)
            violin_ax.set_axisbelow(True)
            sns.despine(ax=violin_ax)
            render_full_width(st.pyplot, violin_fig)
            st.caption("The width of each violin reflects the density of values; inner quartile marks show the median and quartiles. Missing measurements are omitted from this chart only.")
        plt.close(violin_fig)

    with pair_tab:
        st.markdown("Compare pairwise feature relationships and the distribution of each feature on the diagonal.")
        render_pair_plot(filtered, selected_features, species_domain)
        st.caption("Pairwise chart rendering skips only missing points needed for each plotted pair; source records and duplicate rows remain unchanged.")

    with correlation_tab:
        if len(selected_features) < 2:
            st.info("Select at least two numerical features in the sidebar to calculate and display correlations.")
        else:
            corr = filtered[selected_features].corr().rename_axis("row_feature").reset_index()
            corr = corr.melt(id_vars="row_feature", var_name="column_feature", value_name="correlation")
            corr["row_label"] = corr["row_feature"].map(feature_label)
            corr["column_label"] = corr["column_feature"].map(feature_label)
            heatmap = alt.Chart(corr).mark_rect().encode(
                x=alt.X("column_label:N", title=None, sort=[feature_label(f) for f in selected_features]),
                y=alt.Y("row_label:N", title=None, sort=[feature_label(f) for f in selected_features]),
                color=alt.Color("correlation:Q", title="Correlation", scale=alt.Scale(domain=[-1, 1], scheme="redblue")),
                tooltip=[alt.Tooltip("row_label:N", title="Row feature"), alt.Tooltip("column_label:N", title="Column feature"), alt.Tooltip("correlation:Q", title="Correlation", format=".3f")],
            )
            labels = alt.Chart(corr).mark_text(fontSize=13, fontWeight="bold").encode(
                x=alt.X("column_label:N", sort=[feature_label(f) for f in selected_features]),
                y=alt.Y("row_label:N", sort=[feature_label(f) for f in selected_features]),
                text=alt.Text("correlation:Q", format=".2f"),
                color=alt.condition("abs(datum.correlation) > 0.55", alt.value("white"), alt.value("#17243b")),
            )
            render_full_width(st.altair_chart, (heatmap + labels).properties(height=440).interactive())
            st.caption("Correlation values range from −1 to +1. Values near either end indicate stronger relationships; correlation does not prove cause and effect.")

    with species_tab:
        species_chart_type = st.selectbox(
            "Species distribution style", ["Donut chart", "Bar chart"], key="visual_species_distribution_type"
        )
        species_chart = make_species_distribution(filtered, species_chart_type, species_domain)
        render_full_width(st.altair_chart, species_chart)
        st.caption("This chart counts every row in the current filtered view, including exact duplicate records.")

    render_existing_chart_gallery()

st.sidebar.markdown("---")
st.sidebar.caption("Source: project-local iris.csv · Original rows and duplicates are preserved")
