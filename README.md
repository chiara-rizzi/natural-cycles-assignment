# Natural Cycles Data Scientist Assignment

This repository contains the complete analysis pipeline for a time-to-pregnancy study for the Natural Cycles Data Scientist Assignment.
It includes data preprocessing, imputation, survival analysis, and modeling using logistic regression and Cox proportional hazards models. 

## Repository Structure

```
├── code/        # All core analysis scripts
├── notebooks/   # Development and exploratory notebooks (not used in final analysis)
├── latex/       # LaTeX report files
├── requirements.txt
```

## Setup Instructions

1. Clone the repository:
   ```bash
   git clone https://github.com/chiara-rizzi/natural-cycles-assignment.git
   cd natural-cycles-assignment
   ```

2. Create and activate a virtual environment (optional but recommended):
   ```bash
   python -m venv venvnc
   source venvnc/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Running the Pipeline

All analysis scripts must be run from within the `code/` directory.

### Run the Full Pipeline

This performs:
1. Exploratory data analysis
2. Imputation
3. Kaplan-Meier survival analysis
4. Logistic regression
5. Cox proportional hazards modeling

From the `code/` directory:
```bash
cd code
python run_pipeline.py --csv ../data/ncdatachallenge-2021-v1.csv
```

If `--csv` is not specified, it defaults to `../data/ncdatachallenge-2021-v1.csv`.

### Run Individual Steps

Each step can be run independently using the `--step` argument:

```bash
# Step 1: Exploratory Data Analysis
python run_pipeline.py --step eda --csv ../data/ncdatachallenge-2021-v1.csv

# Step 2: Imputation
python run_pipeline.py --step impute --csv ../data/modified_numeric.csv

# Step 3: Kaplan-Meier Survival Analysis
python run_pipeline.py --step survival --csv ../data/modified_small_imputed.csv

# Step 4: Logistic Regression
python run_pipeline.py --step regression --csv ../data/modified_small_imputed.csv

# Step 5: Cox Proportional Hazards Model
python run_pipeline.py --step cox --csv ../data/modified_small_imputed.csv
```

## Code Overview

- **eda.py**: Performs exploratory analysis and variable binning; generates summary plots and saves cleaned data.
- **imputation.py**: Applies MICE imputation to numeric columns and reconstructs categorical variables.
- **survival_analysis.py**: Computes Kaplan-Meier survival curves and probability of pregnancy over time.
- **survival_comparison.py**: Compares Kaplan-Meier survival curves for different selections. 
- **regression.py**: Fits a discrete-time logistic regression model using long-format data to estimate odds ratios.
- **cox_model.py**: Fits a Cox proportional hazards model to assess time-to-pregnancy risk factors.
- **grouping_config.py**: Defines variable grouping logic and baseline categories for modeling.
- **utils.py**: Provides reusable utilities for plotting and data inspection.
- **run_pipeline.py**: Entry point for executing the full pipeline or any individual step.

## Output

- Plots are saved to the `../plots/` directory automatically.
- Processed and imputed datasets are saved in `../data/`.

## Requirements

All dependencies are listed in `requirements.txt`. The environment includes:
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- scipy
- statsmodels
- lifelines
- miceforest
- plotnine
