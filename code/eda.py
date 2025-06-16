import pandas as pd
import numpy as np
import os
from utils import (
    plot_value_counts,
    plot_numerical_distribution,
    plot_missing_values,
    plot_correlation_matrix
)
from grouping_config import (
    education_map,
    sleeping_map,
    pregnancy_map,
    age_bins,
    intercourse_freq_bins,
    dedication_bins,
    bmi_bins,
    cycle_length_range
)

DATA_DIR = '../data'
PLOT_DIR = '../plots'

def load_data(path):
    """Load the dataset and replace string 'NA' with actual None values."""
    df = pd.read_csv(path, index_col=0)
    df = df.replace("NA", None)
    return df

def compute_pregnancy_bounds(df):
    """Compute lower and upper estimates of probability of pregnancy within 13 cycles."""
    n_pregnant_13 = df[(df.outcome=='pregnant') & (df.n_cycles_trying<=13)].shape[0]
    n_not_pregnant_less13 = df[(df.outcome=='not_pregnant') & (df.n_cycles_trying<13)].shape[0]
    n_tot = df.shape[0]
    p13_low = n_pregnant_13 / n_tot
    p13_high = (n_pregnant_13 + n_not_pregnant_less13) / n_tot
    return p13_low, p13_high

def recode_and_group_variables(df):
    """Recode categorical variables and apply binning logic for group analysis."""
    outcome_map = {"pregnant": 1, "not_pregnant": 0}

    university_education = {k: v >= 3 for k, v in education_map.items()}
    regular_sleep = {k: v >= 3 for k, v in sleeping_map.items()}
    pregnancy_binary = {k: v > 0 for k, v in pregnancy_map.items()}

    df['education_numeric'] = df['education'].map(education_map)
    df['sleeping_pattern_numeric'] = df['sleeping_pattern'].map(sleeping_map)
    df['been_pregnant_before_numeric'] = df['been_pregnant_before'].map(pregnancy_map)
    df['university_education'] = df['education'].map(university_education)
    df['regular_sleep'] = df['sleeping_pattern'].map(regular_sleep)
    df['been_pregnant_before_binary'] = df['been_pregnant_before'].map(pregnancy_binary)

    df['age_group'] = df['age'].apply(lambda x: age_bins(x))
    df['intercourse_frequency_group'] = df['intercourse_frequency'].apply(intercourse_freq_bins)
    df['dedication_group'] = df['dedication'].apply(dedication_bins)
    df['bmi_group'] = df['bmi'].apply(bmi_bins)
    df['average_cycle_length_group'] = df['average_cycle_length'].apply(cycle_length_range)
    df['outcome_pregnant'] = df['outcome'].map(outcome_map)

    # Reorder columns to have key outcomes at the top
    priority_cols = ['n_cycles_trying', 'outcome_pregnant']
    other_cols = [col for col in df.columns if col not in priority_cols]
    df = df[priority_cols + other_cols]

    return df

def run_eda(data_path, output_folder=PLOT_DIR):
    """Run the full exploratory data analysis pipeline."""

    # Create the directory if it doesn't exist
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(PLOT_DIR, exist_ok=True)

    df = load_data(data_path)
    p_low, p_high = compute_pregnancy_bounds(df)
    print(f"Estimated bounds for pregnancy within 13 cycles: Low={p_low:.2%}, High={p_high:.2%}")

    df = recode_and_group_variables(df)

    # Plot missing values and correlations
    plot_missing_values(df, output_folder=output_folder)
    plot_correlation_matrix(df, plot_name="correlation_matrix_new_variables", output_folder=output_folder)
    plot_correlation_matrix(df, plot_name="correlation_matrix_new_variables_outcome_pregnant", output_folder=output_folder, query_string='outcome=="pregnant"')

    # Plot distributions
    for col in df.select_dtypes(include='object').columns:
        #print(f"Plotting categorical: {col}")
        plot_value_counts(df, col, output_folder=output_folder)

    for col in df.select_dtypes(include='number').columns:
        #print(f"Plotting numeric: {col}")
        plot_numerical_distribution(df, col, output_folder=output_folder)

    df.to_csv("../data/modified_full.csv", index=False)

    # Save numeric-only columns for imputation
    numeric_columns = [
        'n_cycles_trying', 'outcome_pregnant', 
        'bmi', 'age', 'education_numeric', 'been_pregnant_before_numeric',
        'sleeping_pattern_numeric', 'average_cycle_length', 'cycle_length_std',
        'intercourse_frequency', 'dedication']
    df[numeric_columns].to_csv("../data/modified_numeric.csv", index=False)
    
    return df
