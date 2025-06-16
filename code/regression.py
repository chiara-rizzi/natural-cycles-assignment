import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import os
from survival_comparison import run_survival_comparison
from grouping_config import baseline_categories

PLOT_DIR = "../plots"

def expand_to_long_format(df, time_var="n_cycles_trying", outcome="outcome_pregnant"):
    """Expand wide-format dataset to long-format for discrete-time survival analysis."""
    # Loop over each row and expand into multiple rows, one per cycle
    rows = []
    for idx, row in df.iterrows():
        n_cycles = int(row[time_var])
        # Mark the cycle when the event occurred (if any)
        for cycle in range(1, n_cycles + 1):
            new_row = row.copy()
            new_row["cycle"] = cycle
            new_row["event_this_cycle"] = int((cycle == n_cycles) and row[outcome] == 1)
            rows.append(new_row)
    return pd.DataFrame(rows)

def drop_from_summary(summary, col):
    """Drop a variable from the summary table if it exists."""
    if col in summary.index:
        summary.drop(col, inplace=True)
    return summary

def plot_odds_ratios(result, output_folder, plot_name="OR", figsize=(8, None), title="Odds Ratios with 95% CI", format="jpg", log_scale=True, show_cycle=False):
    """Plot odds ratios with 95% confidence intervals from a fitted logistic model."""
    if isinstance(format, str):
        format = [format]

    os.makedirs(output_folder, exist_ok=True)
    # Extract model summary and compute odds ratios and confidence intervals
    summary = result.summary2().tables[1]
    summary["Odds Ratio"] = np.exp(summary["Coef."])
    summary["CI Lower"] = np.exp(summary["Coef."] - 1.96 * summary["Std.Err."])
    summary["CI Upper"] = np.exp(summary["Coef."] + 1.96 * summary["Std.Err."])
    summary["Significant"] = summary["P>|z|"] < 0.05

    # Determine which variables to exclude from the OR plot
    exclude_vars = ['const']
    if not show_cycle:
        exclude_vars += ['cycle', 'cycle_centered',
                         'cycle_cat_4-6', 'cycle_cat_7-9', 'cycle_cat_10-12', 'cycle_cat_13+']
    for val in exclude_vars:
        summary = drop_from_summary(summary, val)

    # Sort variables by OR magnitude for display
    summary = summary.sort_values("Odds Ratio")
    fig_height = figsize[1] if figsize[1] is not None else 0.4 * len(summary)
    fig, ax = plt.subplots(figsize=(figsize[0], fig_height))

    for i, (name, row) in enumerate(summary.iterrows()):
        color = "red" if row["Significant"] else "#0072B2"
        ax.errorbar(
            x=row["Odds Ratio"],
            y=i,
            xerr=[[row["Odds Ratio"] - row["CI Lower"]], [row["CI Upper"] - row["Odds Ratio"]]],
            fmt='o',
            color=color,
            ecolor=color,
            capsize=3,
            markersize=5
        )

    ax.set_yticks(range(len(summary)))
    ax.set_yticklabels(summary.index)
    for tick, name in zip(ax.get_yticklabels(), summary.index):
        tick.set_color("red" if summary.loc[name, "Significant"] else "#0072B2")

    ax.axvline(1, color="gray", linestyle="--")
    ax.set_xlabel("Odds Ratio" + (" (log scale)" if log_scale else ""))
    if log_scale:
        ax.set_xscale("log")
    ax.set_title(title)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
    ax.invert_yaxis()

    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='red', label='Significant (p < 0.05)', linestyle=''),
        Line2D([0], [0], marker='o', color='#0072B2', label='Not significant', linestyle='')
    ]
    ax.legend(handles=legend_elements, loc='lower left')

    plt.tight_layout()
    for ext in format:
        path = os.path.join(output_folder, f"{plot_name}.{ext}")
        plt.savefig(path, format=ext, dpi=300)
        print(f"Saved: {path}")
    plt.show()

def prepare_model_matrix(df, categorical_baselines={}, center_numeric=[], leave_unchanged=[], drop_first=True):
    """Prepares model matrix with specified categorical baselines."""
    df_processed = df.copy()

    # Determine relevant columns
    allowed_columns = set(categorical_baselines.keys()) | set(center_numeric) | set(leave_unchanged)
    df_processed = df_processed[[col for col in df_processed.columns if col in allowed_columns]]
    
    # Apply categorical baselines
    for col, baseline in categorical_baselines.items():
        if col not in df_processed.columns:
            continue
        if baseline not in df_processed[col].unique():
            raise ValueError(f"Baseline '{baseline}' not found in column '{col}'")
        categories = [baseline] + [c for c in df_processed[col].dropna().unique() if c != baseline]
        df_processed[col] = pd.Categorical(df_processed[col], categories=categories, ordered=False)

    # Create dummy variables for categorical columns
    cat_cols = list(categorical_baselines.keys())
    dummies = pd.get_dummies(df_processed[cat_cols], drop_first=drop_first, dtype=int)

    # Center numeric variables
    for col in center_numeric:
        if col in df_processed.columns:
            df_processed[col + '_centered'] = df_processed[col] - df_processed[col].mean()

    # Combine final matrix
    other_vars = [col for col in df_processed.columns if col not in cat_cols and col not in center_numeric]
    X = pd.concat([df_processed[other_vars], dummies], axis=1)
    
    return X

def run_logistic_regression(data):
    """
    Fit a discrete-time logistic regression model to assess the influence of predictors on time-to-pregnancy.

    Parameters:
        data (str or pd.DataFrame): Path to dataset or dataframe.

    Returns:
        result: Fitted logistic regression model.
    """
    # Load data if path provided
    if isinstance(data, str):
        df = pd.read_csv(data)
    else:
        df = data.copy()
        
    # Run comparison of survival curves to visually verify proportional odds assumption
    run_survival_comparison(df)

    print("Starting logistic regression analysis")
    print("Original data shape:", df.shape)
    # Transform to long format for discrete-time analysis
    df_long = expand_to_long_format(df)
    print("Expanded data shape (long format):", df_long.shape)

    # Categorize cycle time into intervals
    df_long["cycle_cat"] = pd.cut(
        df_long["cycle"],
        bins=[0, 3, 6, 9, 12, np.inf],
        labels=["1-3", "4-6", "7-9", "10-12", "13+"]
    )

    print("Preparing categorical variables with specified baselines")
    # Prepare model matrix with dummy encoding based on categorical baselines
    X = prepare_model_matrix(df_long.drop(columns=['event_this_cycle', 'outcome_pregnant', 'n_cycles_trying']), 
                             categorical_baselines=baseline_categories)
    
    y = df_long["event_this_cycle"]

    # Remove missing values
    mask = X.notnull().all(axis=1) & y.notnull()
    X = X[mask]
    y = y[mask]

    # Add intercept and fit model
    X = sm.add_constant(X)

    # Ensure X and y are numeric
    #X = X.apply(pd.to_numeric)
    #y = pd.to_numeric(y)
    
    print("Final design matrix shape:", X.shape)
    print("Final outcome vector shape:", y.shape)
    
    model = sm.Logit(y, X)
    result = model.fit()
    print("Model fitting complete.")
    print(result.summary())

    # Plot odds ratios
    os.makedirs(PLOT_DIR, exist_ok=True)
    plot_odds_ratios(result, output_folder=PLOT_DIR, log_scale=False, show_cycle=False)

    return result
