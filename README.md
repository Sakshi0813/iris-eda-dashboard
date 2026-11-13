# Iris EDA and Machine Learning Dashboard

A beginner-friendly Data Science project that combines exploratory data analysis (EDA), interactive visualizations, and supervised Iris species classification in a Streamlit dashboard. The project includes both a command-line EDA workflow and saved scikit-learn models for evaluation and interactive predictions.

> **Project status:** Local project. No deployed application or public dashboard link is currently configured.

## Table of contents

- [Project overview](#project-overview)
- [Project objectives](#project-objectives)
- [Technologies used](#technologies-used)
- [Dataset information](#dataset-information)
- [Project workflow](#project-workflow)
- [Features](#features)
- [Machine learning models](#machine-learning-models)
- [Project folder structure](#project-folder-structure)
- [Installation and setup](#installation-and-setup)
- [How to use](#how-to-use)
- [Results and key insights](#results-and-key-insights)
- [Future improvements](#future-improvements)
- [Author](#author)
- [License](#license)

## Project overview

Exploratory Data Analysis helps us understand a dataset before drawing conclusions or building a model. This project loads Iris flower measurements, checks the data, calculates summary statistics, and presents patterns in charts. It also trains two classification models to estimate a flower's species from its measurements.

The Streamlit application brings these tasks together in a filterable dashboard. It is intended as a learning project and portfolio example; its predictions are estimates and are not guaranteed to be correct.

## Project objectives

- Load and inspect the supplied Iris dataset.
- Check data types, missing values, and repeated records without silently deleting duplicates.
- Calculate descriptive and species-wise statistics.
- Explore measurement distributions and relationships through charts.
- Train and evaluate Decision Tree and Random Forest classifiers.
- Provide an interactive dashboard for exploration, model comparison, and species prediction.

## Technologies used

| Technology | Use in this project |
| --- | --- |
| Python | Project scripts and dashboard logic |
| Pandas | Reading, filtering, grouping, and summarizing tabular data |
| NumPy | Numeric validation and array operations |
| Matplotlib | EDA chart generation and dashboard figures |
| Seaborn | Statistical plots, including histograms, pair plots, and violin plots |
| Altair | Interactive dashboard charts |
| Scikit-learn | Data splitting, preprocessing pipelines, classifiers, and evaluation metrics |
| Joblib | Saving and loading trained model artifacts |
| Streamlit | Interactive web dashboard |

## Dataset information

The project uses the [Iris dataset from the UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/53/iris). The supplied `data/iris.data` file is included, so the EDA script does not need to download it. The project CSV, `iris.csv`, contains **150 records** and **5 columns**. The three species have 50 records each.

| Column | Description | Data role |
| --- | --- | --- |
| `sepal_length_cm` | Sepal length, in centimeters | Numerical feature |
| `sepal_width_cm` | Sepal width, in centimeters | Numerical feature |
| `petal_length_cm` | Petal length, in centimeters | Numerical feature |
| `petal_width_cm` | Petal width, in centimeters | Numerical feature |
| `species` | Setosa, Versicolor, or Virginica | Target label |

The current CSV has **0 missing cells** and **3 duplicate rows** (extra copies across two exact-match groups). The EDA and model workflow retain all 150 records. For model evaluation, identical rows are kept in the same split group so an exact copy cannot appear in both the training and test sets.

## Project workflow

1. **Load the dataset:** `iris_eda.py` reads `data/iris.data` with Pandas, assigns column names, normalizes the species labels, and writes `iris.csv`. The dashboard and `ml_model.py` read `iris.csv` without writing to it.
2. **Prepare and inspect data:** The EDA script checks numeric values and reports shape, column information, missing values, and duplicates. The dashboard offers interactive filters and data-quality details.
3. **Explore and visualize:** The scripts calculate overall and species-wise summaries and create EDA charts. Dashboard charts respond to species and feature selections.
4. **Train models:** `ml_model.py` creates a reproducible, stratified 80/20 train/test split and trains a Decision Tree and a Random Forest.
5. **Evaluate models:** The held-out test rows are used for accuracy, macro precision, macro recall, macro F1-score, classification reports, and confusion matrices.
6. **Explore predictions:** The Streamlit prediction page accepts four measurements and uses a saved model to return an estimated species.

## Features

- **Dashboard overview:** Shows the filtered dataset preview, species distribution, and data-quality summary.
- **Dataset exploration:** Displays filtered records, column types, missing counts, duplicate counts, and a CSV download.
- **Statistical analysis:** Shows mean, median, sample standard deviation, minimum, maximum, quartiles, IQR, and skewness, plus species-wise comparisons. Summary tables can be downloaded as CSV.
- **Interactive visualizations:** Includes histograms, scatter plots, box plots, violin plots, a Seaborn pair plot, a correlation heatmap, and species distribution charts. Species and numerical-feature filters update the applicable views.
- **Saved chart gallery:** Displays the existing PNG charts from `outputs/`.
- **Species comparison:** Compares species counts and measurement averages with bar, donut, grouped-bar, or dot chart styles.
- **Key insights and outliers:** Calculates filter-dependent correlation and average-value insights, plus IQR-based potential outlier bounds and records. Outlier rows are flagged for review, not removed.
- **Machine learning comparison:** Compares both saved classifiers in a table and chart; displays classification reports, confusion matrices, and Random Forest feature importance.
- **Iris prediction:** Accepts sepal and petal measurements in centimeters, lets the user choose a model, and displays its estimated species. Input controls use the observed dataset ranges.
- **Analysis downloads:** Exports filtered records, statistical comparisons, and outlier analysis data as CSV files.

## Machine learning models

The two scikit-learn models use the four measurement columns to predict `species`:

- **Decision Tree Classifier:** Learns a sequence of feature-based splits.
- **Random Forest Classifier:** Combines predictions from an ensemble of decision trees.

`ml_model.py` uses a fixed random state of 42 and a stratified 80/20 split. Exact duplicate rows are assigned as groups so they do not cross between training and test data. In the current split, 120 rows are used for training and 30 for testing, with 10 test rows per species. A preprocessing pipeline contains median imputation and standardization; it is fitted on the training data only. The test set is held out from model fitting and is used for evaluation.

The current saved model results are:

| Model | Accuracy | Macro precision | Macro recall | Macro F1-score |
| --- | ---: | ---: | ---: | ---: |
| Decision Tree | 0.933 | 0.944 | 0.933 | 0.933 |
| Random Forest | 0.933 | 0.944 | 0.933 | 0.933 |

The confusion matrix for each model is the same in this saved evaluation: all 10 Setosa and all 10 Versicolor test records were classified correctly; 8 of 10 Virginica records were classified correctly, and 2 were predicted as Versicolor. These values describe one fixed test split of this small dataset. They do not guarantee the same performance on new data.

The saved Random Forest assigns its highest feature importance to petal width (about 0.452), followed by petal length (about 0.411). Feature importance describes how this fitted model uses its inputs; it does not show that a measurement causes a species difference.

## Project folder structure

```text
Iris_EDA_Project/
├── data/
│   └── iris.data
├── outputs/                         # Nine saved EDA charts
│   ├── box_plots_by_species.png
│   ├── correlation_heatmap.png
│   ├── histogram_*.png              # Four measurement histograms
│   ├── petal_measurements_scatter.png
│   ├── sepal_measurements_scatter.png
│   └── species_counts.png
├── models/
│   ├── decision_tree.joblib
│   └── random_forest.joblib
├── app.py                           # Streamlit EDA and ML dashboard
├── iris.csv                         # 150-row, 5-column dataset
├── iris_eda.py                      # EDA and saved chart generation
├── main.py                          # EDA entry point
├── ml_model.py                      # Model training, evaluation, and prediction helpers
├── INTERNSHIP_REPORT.md             # Project internship report
├── requirements.txt
├── .gitignore
└── README.md
```

## Installation and setup

### 1. Clone the repository

```bash
git clone https://github.com/Sakshi0813/iris-eda-dashboard.git
cd iris-eda-dashboard
```

### 2. Create and activate a virtual environment

**Windows PowerShell:**

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

**macOS or Linux:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Train or refresh the saved models

```bash
python ml_model.py
```

This creates or refreshes the two files in `models/`. The dashboard does not retrain models during normal page interactions. If a model file is missing or was trained against a different CSV, the dashboard asks you to run this step.

### 5. Start the dashboard

```bash
streamlit run app.py
```

To run the original command-line EDA and regenerate `iris.csv` and the saved charts, run `python main.py`. That command writes those outputs from `data/iris.data`; the dashboard itself reads the existing CSV without modifying it.

## How to use

1. Start the dashboard using `streamlit run app.py`.
2. Use the sidebar navigation to open **Overview**, **Data explorer**, **Statistical analysis**, **Species comparison**, **Advanced analysis**, **Charts & gallery**, **Machine Learning**, or **Iris Prediction**.
3. Select one or more species and measurement features in the sidebar. The relevant data summaries and EDA charts update to reflect those selections. Use **Reset filters** to restore the defaults.
4. Download filtered records or analysis tables from the pages that provide CSV download buttons.
5. Open **Machine Learning** to review saved model scores and diagnostics. If the models are missing or stale, train them from the project folder with `python ml_model.py`.
6. Open **Iris Prediction**, select Decision Tree or Random Forest, enter all four measurements in centimeters within the displayed dataset ranges, and choose **Predict species**. Treat the result as an estimate, not a guaranteed answer.

## Results and key insights

- The dataset has 150 records, 50 for each species, with no missing cells.
- There are 3 extra exact duplicate copies in two groups. They are reported and retained; they are not automatically removed.
- The petal length and petal width measurements have a strong positive correlation in this dataset (Pearson correlation about **0.963**).
- In the recorded data, Virginica has the highest average petal length (**5.552 cm**) and petal width (**2.026 cm**). Setosa has the lowest average petal length (**1.464 cm**) and width (**0.244 cm**).
- Both saved classifiers achieved **0.933 accuracy** on the same 30-row held-out test split. The result is useful for demonstrating the workflow, but additional validation would be needed before drawing broader conclusions.

## Dashboard Screenshots

No dashboard screenshot files are currently available in this project. The PNG files in `outputs/` are individual EDA charts, not captures of the Streamlit dashboard, so they are not shown here as screenshots. Add actual dashboard captures to a `screenshots/` folder and replace these TODOs with relative Markdown image links and captions.

> **TODO — Dashboard overview:** Add the overview screenshot and a short caption.

> **TODO — Machine Learning:** Add the model comparison screenshot and a short caption.

> **TODO — Iris Prediction:** Add the prediction page screenshot and a short caption.

## Future improvements

- Deploy the dashboard and document the hosting setup once deployment is completed.
- Evaluate models with repeated cross-validation and additional metrics, while continuing to prevent duplicate-row leakage.
- Add more datasets or a user-provided dataset workflow with validation and clear data-quality reporting.
- Extend analysis with model calibration, decision-boundary views, or explainability methods.

## Author

**Sakshi** · [GitHub profile](https://github.com/Sakshi0813)

## License

No `LICENSE` file is currently included in this project, so no open-source license is specified. If you plan to publish or reuse the project, add a license that reflects your intended terms.
