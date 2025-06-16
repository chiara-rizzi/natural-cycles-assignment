import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import miceforest as mf
from scipy.stats import gaussian_kde

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

PLOT_DIR = "../plots"

def load_numeric_data(path):
    """Load numeric-only data for imputation."""
    df = pd.read_csv(path)
    return df

def run_mice_imputation(df, iterations=50, random_state=100):
    """Perform MICE imputation on numeric data."""
    kernel = mf.ImputationKernel(
        data=df,
        random_state=random_state,
        save_all_iterations_data=True,
        num_datasets=1
    )
    kernel.mice(iterations=iterations)
    imputed_df = kernel.complete_data(dataset=0)
    for column in df.columns:
        if df[column].isnull().any():
            n_unique = df[column].nunique(dropna=True)
            var_type = 'categorical' if n_unique < 10 or df[column].dtype == 'object' else 'continuous'
            plot_imputed_vs_observed(kernel, df, column, variable_type=var_type)

    return imputed_df, kernel

def restore_target_columns(imputed_df, original_df, target_cols):
    """Reinsert target columns (which were excluded from imputation)."""
    imputed_df[target_cols] = original_df[target_cols]
    return imputed_df

def postprocess_imputed_data(imputed_df):
    """Derive grouped and binary columns from imputed values."""
    education_inv = {v: k for k, v in education_map.items()}
    sleeping_inv = {v: k for k, v in sleeping_map.items()}
    pregnancy_inv = {v: k for k, v in pregnancy_map.items()}

    imputed_df['education'] = imputed_df['education_numeric'].map(education_inv)
    imputed_df['sleeping_pattern'] = imputed_df['sleeping_pattern_numeric'].map(sleeping_inv)
    imputed_df['been_pregnant_before'] = imputed_df['been_pregnant_before_numeric'].map(pregnancy_inv)

    imputed_df['university_education'] = imputed_df['education'].map(lambda x: education_map.get(x, -1) >= 3)
    imputed_df['regular_sleep'] = imputed_df['sleeping_pattern'].map(lambda x: sleeping_map.get(x, -1) >= 3)
    imputed_df['been_pregnant_before_binary'] = imputed_df['been_pregnant_before'].map(lambda x: pregnancy_map.get(x, -1) > 0)

    imputed_df['regular_cycle'] = imputed_df['cycle_length_std'] < 5
    imputed_df['age_group'] = imputed_df['age'].apply(age_bins)
    imputed_df['intercourse_frequency_group'] = imputed_df['intercourse_frequency'].apply(intercourse_freq_bins)
    imputed_df['dedication_group'] = imputed_df['dedication'].apply(dedication_bins)
    imputed_df['bmi_group'] = imputed_df['bmi'].apply(bmi_bins)
    imputed_df['average_cycle_length_group'] = imputed_df['average_cycle_length'].apply(cycle_length_range)

    return imputed_df

def plot_imputed_vs_observed(kernel, df_original, variable, variable_type='continuous', dataset=0):
    os.makedirs(PLOT_DIR, exist_ok=True)
    df_imputed = kernel.complete_data(dataset=dataset)
    missing_idx = df_original[df_original[variable].isnull()].index
    observed_idx = df_original[df_original[variable].notnull()].index

    imputed_vals = df_imputed.loc[missing_idx, variable].dropna()
    observed_vals = df_imputed.loc[observed_idx, variable].dropna()

    if variable_type == 'continuous':
        grid = np.linspace(min(observed_vals.min(), imputed_vals.min()), max(observed_vals.max(), imputed_vals.max()), 100)

        def bootstrap_kde_ci(data, n_boot=100):
            kde_values = []
            for _ in range(n_boot):
                sample = np.random.choice(data, size=len(data), replace=True)
                kde = gaussian_kde(sample)
                kde_values.append(kde(grid))
            kde_values = np.array(kde_values)
            return grid, np.mean(kde_values, axis=0), np.percentile(kde_values, 2.5, axis=0), np.percentile(kde_values, 97.5, axis=0)

        x_obs, y_obs, low_obs, up_obs = bootstrap_kde_ci(observed_vals)
        x_imp, y_imp, low_imp, up_imp = bootstrap_kde_ci(imputed_vals)

        plt.figure(figsize=(10, 5))
        plt.plot(x_obs, y_obs, label='Observed', color='#277CB7')
        plt.fill_between(x_obs, low_obs, up_obs, color='#C6DCEC', alpha=1.0, label='Observed 95% CI')
        plt.plot(x_imp, y_imp, label='Imputed', color='#F28E2B')
        plt.fill_between(x_imp, low_imp, up_imp, color='#FDD9B5', alpha=1.0, label='Imputed 95% CI')
        plt.title(f"Distribution of '{variable}': Observed vs Imputed")
        plt.ylabel("Density")
    else:
        obs_freq = observed_vals.value_counts(normalize=True).sort_index()
        imp_freq = imputed_vals.value_counts(normalize=True).sort_index()
        categories = sorted(set(obs_freq.index).union(set(imp_freq.index)))
        obs_freq = obs_freq.reindex(categories, fill_value=0)
        imp_freq = imp_freq.reindex(categories, fill_value=0)

        def binom_ci(p, n, z=1.96):
            se = np.sqrt(p * (1 - p) / n) if n > 0 else 0
            return p - z * se, p + z * se

        obs_ci = np.array([binom_ci(p, len(observed_vals)) for p in obs_freq])
        imp_ci = np.array([binom_ci(p, len(imputed_vals)) for p in imp_freq])

        x = np.arange(len(categories))
        width = 0.35

        plt.figure(figsize=(10, 5))
        plt.bar(x - width / 2, obs_freq.values, width, yerr=[obs_freq.values - obs_ci[:, 0], obs_ci[:, 1] - obs_freq.values],
                capsize=5, label='Observed', color='#C6DCEC', edgecolor='#277CB7', linewidth=1.5)
        plt.bar(x + width / 2, imp_freq.values, width, yerr=[imp_freq.values - imp_ci[:, 0], imp_ci[:, 1] - imp_freq.values],
                capsize=5, label='Imputed', color='#FDD9B5', edgecolor='#F28E2B', linewidth=1.5)
        plt.xticks(x, categories)
        plt.ylabel("Relative Frequency")
        plt.title(f"Distribution of '{variable}': Observed vs Imputed")

    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    filename = os.path.join(PLOT_DIR, f"mf_{variable}.jpg")
    plt.savefig(filename, format='jpg')
    plt.close()
    print(f"Saved: {filename}")

def run_imputation(numeric_path, output_path_small, output_path_full=None):
    """Main function to run the full imputation and postprocessing pipeline."""

    print("Please wait — this may take a few seconds.")
    df = load_numeric_data(numeric_path)
    target_cols = ['n_cycles_trying', 'outcome_pregnant']
    features = df.drop(columns=target_cols)

    imputed_df, kernel = run_mice_imputation(features)
    imputed_df = restore_target_columns(imputed_df, df, target_cols)
    imputed_df = postprocess_imputed_data(imputed_df)

    selected_cols = [
        'n_cycles_trying', 'outcome_pregnant', 
        'age_group','bmi_group','been_pregnant_before_binary', 
        'average_cycle_length_group', 'regular_cycle',
        'university_education', 'regular_sleep',
        'intercourse_frequency_group', 'dedication_group']

    imputed_df[selected_cols].to_csv(output_path_small, index=False)
    if output_path_full:
        imputed_df.to_csv(output_path_full, index=False)

    return imputed_df
