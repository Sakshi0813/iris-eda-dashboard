# Iris Flower Dataset: Exploratory Data Analysis

**Project type:** Beginner Data Science internship project  
**Analysis file:** `iris.csv`  
**Records analyzed:** 150

## 1. Introduction

Exploratory Data Analysis (EDA) is an early step in working with a dataset. It helps describe the data, check its quality, summarize measurements, and look for patterns using charts. This project explores measurements for three Iris flower species using Python.

## 2. Project objective

The objective was to load the Iris data with Pandas, create a CSV copy, inspect its structure and quality, calculate descriptive statistics, compare measurements by species, and create charts. Duplicate records are reported but not removed.

## 3. Dataset description

The project uses the Iris dataset from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/53/iris). `iris.csv` has **150 rows and 5 columns**: four numeric measurements in centimeters and a species label. There are 50 records for each of Setosa, Versicolor, and Virginica.

| Column | Description | Type |
| --- | --- | --- |
| `sepal_length_cm` | Sepal length in centimeters | Numeric |
| `sepal_width_cm` | Sepal width in centimeters | Numeric |
| `petal_length_cm` | Petal length in centimeters | Numeric |
| `petal_width_cm` | Petal width in centimeters | Numeric |
| `species` | Iris species name | Category |

## 4. Tools and technologies

- **Python** runs the analysis.
- **Pandas** loads, checks, and summarizes tabular data.
- **NumPy** supports numeric validation and calculations.
- **Matplotlib** and **Seaborn** create the charts.

## 5. Data loading process

The script reads `data/iris.data` with Pandas. As the source has no header row, it assigns names to the four measurements and species column. It drops a possible blank trailing line, converts labels such as `Iris-setosa` to `Setosa`, checks that measurements are finite numbers, and writes the resulting 150 rows to `iris.csv`. No rows were removed in preparing this report.

## 6. Exploratory data analysis

The dataset contains **150 records and 5 columns**. Every column has **0 missing values**. Pandas reports **3 repeated copies** in **2 exact duplicate groups**; five rows total belong to these groups. The first five rows, shape, column types, missing counts, duplicate count, species summaries, and visualizations were reviewed.

## 7. Statistical analysis

These values were calculated from all 150 rows in `iris.csv`. Standard deviation is the sample standard deviation used by Pandas (`ddof=1`). Values are in centimeters and rounded to three decimal places.

| Statistic | Sepal length | Sepal width | Petal length | Petal width |
| --- | ---: | ---: | ---: | ---: |
| Mean | 5.843 | 3.054 | 3.759 | 1.199 |
| Median | 5.800 | 3.000 | 4.350 | 1.300 |
| Standard deviation | 0.828 | 0.434 | 1.764 | 0.763 |
| Minimum | 4.300 | 2.000 | 1.000 | 0.100 |
| Maximum | 7.900 | 4.400 | 6.900 | 2.500 |

### Species-wise averages

| Species | Flowers | Sepal length | Sepal width | Petal length | Petal width |
| --- | ---: | ---: | ---: | ---: | ---: |
| Setosa | 50 | 5.006 | 3.418 | 1.464 | 0.244 |
| Versicolor | 50 | 5.936 | 2.770 | 4.260 | 1.326 |
| Virginica | 50 | 6.588 | 2.974 | 5.552 | 2.026 |

## 8. Explanation of all nine charts

All images are PNG files in `outputs/`. Histograms use color for species, and scatter plots color points by species.

1. **Sepal length histogram** — [`histogram_sepal_length_cm.png`](outputs/histogram_sepal_length_cm.png) shows sepal length distributions for the three species.
2. **Sepal width histogram** — [`histogram_sepal_width_cm.png`](outputs/histogram_sepal_width_cm.png) shows sepal width distributions by species.
3. **Petal length histogram** — [`histogram_petal_length_cm.png`](outputs/histogram_petal_length_cm.png) compares petal length distributions; Setosa appears in a lower range than the other groups.
4. **Petal width histogram** — [`histogram_petal_width_cm.png`](outputs/histogram_petal_width_cm.png) shows petal width distributions by species.
5. **Sepal scatter plot** — [`sepal_measurements_scatter.png`](outputs/sepal_measurements_scatter.png) compares sepal length with sepal width; each point is a record.
6. **Petal scatter plot** — [`petal_measurements_scatter.png`](outputs/petal_measurements_scatter.png) compares petal length with petal width, with points colored by species.
7. **Box plots** — [`box_plots_by_species.png`](outputs/box_plots_by_species.png) compares the spread and medians of all four measurements for each species. Points beyond whiskers may be potential outliers.
8. **Species count bar chart** — [`species_counts.png`](outputs/species_counts.png) shows the count of records per species: 50 each.
9. **Correlation heatmap** — [`correlation_heatmap.png`](outputs/correlation_heatmap.png) displays pairwise correlations among measurements. Petal length and width have the strongest positive correlation, approximately **0.963**. Correlation does not prove cause and effect.

## 9. Duplicate record investigation

The default Pandas duplicate count is **3**: these are extra rows repeating an earlier record. There are two exact-match groups across all measurements and the species label:

| Zero-based CSV indices (one-based row numbers) | Sepal length | Sepal width | Petal length | Petal width | Species | Group size |
| --- | ---: | ---: | ---: | ---: | --- | ---: |
| 9, 34, 37 (10, 35, 38) | 4.9 | 3.1 | 1.5 | 0.1 | Setosa | 3 |
| 101, 142 (102, 143) | 5.8 | 2.7 | 5.1 | 1.9 | Virginica | 2 |

Exact matches do not prove that rows refer to the same physical flower: there is no unique flower identifier, and measurements have limited precision. Repeated rows may also result from copying or combining data more than once. This descriptive EDA keeps the supplied records so all 150 observations are represented. The script reports duplicates and does not delete them. Removing rows would require a separate decision based on data provenance and analysis purpose.

## 10. Key observations

- The dataset is balanced, with 50 records per species, and has no missing values.
- Setosa has the smallest average petal measurements: 1.464 cm long and 0.244 cm wide.
- Virginica has the largest average sepal length (6.588 cm), petal length (5.552 cm), and petal width (2.026 cm). Versicolor has the smallest average sepal width (2.770 cm).
- Petal length and petal width have a strong positive correlation in this dataset (about 0.963).
- Three extra exact repeated copies occur in two groups. They remain in the dataset and are included in the summaries and charts.

These observations describe this dataset and do not establish causes or describe every Iris flower.

## 11. Conclusion

This project completed a basic EDA of the 150-row Iris dataset: it loaded the source with Pandas, saved a CSV, checked data structure and quality, calculated overall and species-wise statistics, and generated nine visualizations. Two groups of exact duplicate records were documented and all 150 rows were kept. The results provide a clear starting point for further study while staying within what this dataset shows.
