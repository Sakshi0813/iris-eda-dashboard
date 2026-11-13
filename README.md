# Iris Flower Dataset: Exploratory Data Analysis

A beginner-friendly Python project that explores measurements of three Iris flower species. It loads the data with Pandas, checks its quality, prints useful summaries, and saves charts for visual exploration.

## Project structure

```text
Iris_EDA_Project/
├── data/
│   └── iris.data          # Original UCI data file used as the input
├── outputs/               # Charts created by the script
├── iris.csv               # Clean CSV created when the script runs
├── main.py                # Project entry point
├── iris_eda.py            # Main analysis script
├── ml_model.py            # Train, evaluate, save, and load classifiers
├── models/                # Generated model artifacts (created by training)
│   ├── decision_tree.joblib
│   └── random_forest.joblib
├── app.py                 # Interactive EDA and machine learning dashboard
├── requirements.txt       # Python packages needed
└── README.md
```

## Dataset

The data comes from the [UCI Machine Learning Repository Iris dataset](https://archive.ics.uci.edu/dataset/53/iris). It contains 150 flower records, four measurements in centimeters, and a species label. The included `data/iris.data` file means the analysis can run without downloading the dataset again.

The script uses Pandas to read the source data and writes a tidy, headed copy named `iris.csv` in the project folder. The species names in that CSV are `Setosa`, `Versicolor`, and `Virginica`.

## Machine learning workflow

The project includes two scikit-learn classifiers: a Decision Tree and a Random Forest. Both use the four centimeter measurements to predict the species. `ml_model.py` reads the existing `iris.csv` without changing it or removing duplicate records.

The data is divided into training and test portions with a fixed random seed and stratification by species. Exact duplicate rows are assigned to the same split group, so a copy of an identical row cannot leak from training into testing. All source rows are retained. The test set is held out from model fitting and is used once to calculate the reported evaluation scores. Each model uses a pipeline for median imputation and scaling; those preprocessing steps are fitted as part of training on the training portion only.

The dashboard reports accuracy, macro precision, macro recall, macro F1-score, per-species classification reports, and confusion matrices. Macro averaging weights each species equally. Random Forest feature importance shows how much the fitted forest relied on each measurement; it is a model-specific description and does not show causation. Scores are based on this fixed holdout and are not a guarantee of future performance.

### Train models

Install project requirements, then run:

```bash
python ml_model.py
```

This creates `models/decision_tree.joblib` and `models/random_forest.joblib`. Run this command again after changing the CSV. The dashboard checks that the loaded model was trained from the current CSV and displays an instruction if a model is missing or stale. Joblib model files use Python serialization; only load artifacts created by this project or otherwise trusted sources.

### Run the dashboard and make a prediction

```bash
streamlit run app.py
```

Use the **Machine Learning** page to compare model scores, classification reports, confusion matrices, and Random Forest feature importance. Open **Iris Prediction**, choose a saved model, enter sepal length, sepal width, petal length, and petal width in centimeters, and submit the form. Input fields use the dataset's observed minimum and maximum values. The output is a prediction estimate and may be incorrect.

### Expanded project structure

```text
Iris_EDA_Project/
├── data/iris.data
├── outputs/                         # Existing EDA charts
├── models/                          # Created by `python ml_model.py`
│   ├── decision_tree.joblib
│   └── random_forest.joblib
├── iris.csv                         # Existing data; kept unchanged by the ML module
├── ml_model.py                      # Training and evaluation code
├── app.py                           # EDA, model evaluation, and prediction pages
├── main.py / iris_eda.py            # Existing EDA scripts
├── requirements.txt
└── README.md
```

## What the analysis does

1. Loads the local dataset and saves `iris.csv`.
2. Prints the first five rows, dataset shape, and column data types.
3. Checks each column for missing values and counts duplicate rows.
4. Calculates mean, median, standard deviation, minimum, and maximum for each measurement.
5. Prints average measurements and flower counts for each species.
6. Saves histograms, scatter plots, box plots, a species count bar chart, and a correlation heatmap to `outputs/`.

## Duplicate record review

The included CSV has **3 repeated copies** according to Pandas' default `duplicated()` count. Those copies belong to **2 exact-match groups**, involving 5 rows total. The records below match in all five columns (four measurements and species). Row numbers are one-based; CSV indices are zero-based.

| CSV indices (row numbers) | Sepal length (cm) | Sepal width (cm) | Petal length (cm) | Petal width (cm) | Species | Copies |
| --- | ---: | ---: | ---: | ---: | --- | ---: |
| 9, 34, 37 (10, 35, 38) | 4.9 | 3.1 | 1.5 | 0.1 | Setosa | 3 |
| 101, 142 (102, 143) | 5.8 | 2.7 | 5.1 | 1.9 | Virginica | 2 |

Matching measurements and labels do not prove that two rows describe the same physical flower: this dataset has no unique flower identifier, and different flowers can share measurements at the recorded precision. Repeated rows can also arise when data is copied or combined more than once. For this descriptive EDA, the project keeps the original records so its analysis reflects the supplied dataset. The script reports duplicates but does not remove them. If the data is later used for a model or a different analysis, decide whether to deduplicate based on the data's provenance and purpose, and document that choice because repeated rows affect summary statistics and plots.

### Charts created

- Four measurement histograms
- Two scatter plots (sepal measurements and petal measurements)
- One box plot comparing all measurements by species
- One bar chart showing the number of flowers per species
- One correlation heatmap for the four numeric measurements

## Setup and run

Use Python 3.9 or newer. From this folder, create and activate a virtual environment if you want to keep project packages separate:

```bash
python -m venv .venv
```

On Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Install the packages and run the analysis:

```bash
python -m pip install -r requirements.txt
python main.py
```

The summary appears in the terminal. The script creates or refreshes `iris.csv` and the PNG charts in `outputs/` each time it runs.

## Beginner notes

- A **row** is one flower; a **column** is a measurement or its species label.
- The **shape** is reported as `(rows, columns)`.
- The **median** is the middle value after sorting. The **standard deviation** describes how spread out the values are.
- A **histogram** groups measurements into ranges. A **scatter plot** compares two measurements. A **box plot** summarizes spread. A **bar chart** counts categories. A **correlation heatmap** shows how numeric measurements move together; correlation does not prove cause and effect.
- A missing-value count above zero means that column has missing data. Duplicate rows are counted, not removed.

## Tools used

Python, Pandas, NumPy, Matplotlib, and Seaborn.
