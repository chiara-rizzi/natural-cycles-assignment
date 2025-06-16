import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import os
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import pandas as pd
import re
import seaborn as sns


def plot_numerical_distribution(df, column_name, output_folder="../plots/", file_name="", query_string=None, bins=30):
    # Apply filtering if a query string is provided
    if query_string:
        try:
            df = df.query(query_string)
        except Exception as e:
            print(f"Invalid query string: {e}")
            return

    # Ensure the output directory exists
    os.makedirs(output_folder, exist_ok=True)

    # Drop NA for plotting histogram
    data = df[column_name].dropna()

    if data.empty:
        print(f"No data available in column '{column_name}' after filtering.")
        return

    # Determine if all values are integers (even if dtype is float)
    is_all_int = np.all(np.equal(np.mod(data, 1), 0))

    # Choose binning strategy
    if is_all_int and data.nunique() < 100:
        bin_edges = np.arange(data.min(), data.max() + 2) - 0.5  # center bins on integers
    else:
        bin_edges = bins  # fallback to default

    # Create plot
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(data, bins=bin_edges, color="#C6DCEC", edgecolor="#277CB7")

    # Format title
    title = f'Distribution of "{column_name}"'
    if query_string:
        title += f'\n[Filter: {query_string}]'
    ax.set_title(title, fontsize=14)

    ax.set_ylabel("Frequency")
    ax.set_xlabel(column_name)
    ax.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()

    # If file name is not provided, generate one
    if not file_name:
        safe_query = re.sub(r"[^\w\-]+", "_", query_string) if query_string else ""
        file_name = f"{column_name}__{safe_query}.jpg" if safe_query else f"{column_name}.jpg"

    output_path = os.path.join(output_folder, file_name)
    plt.savefig(output_path, format='jpg')

    # Show and close plot
    plt.show()
    plt.close()
    print(f"Plot saved to: {output_path}") 

def plot_value_counts(df, column_name, output_folder="../plots/", file_name="", query_string=None):
    # Apply filtering if a query string is provided
    if query_string:
        try:
            df = df.query(query_string)
        except Exception as e:
            print(f"Invalid query string: {e}")
            return

    # Ensure the output directory exists
    os.makedirs(output_folder, exist_ok=True)

    # Compute value counts including NaN
    value_counts = df[column_name].value_counts(dropna=False)

    # Replace NaN label with "NA" for plotting
    labels = value_counts.index.to_series().astype(str).replace("nan", "NA")

    # Total for percentage
    total = value_counts.sum()

    # Create plot
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(
        labels,
        value_counts.values,
        color="#C6DCEC",
        edgecolor="#277CB7"
    )

    # Format title
    title = f'Distribution of Values in "{column_name}"'
    if query_string:
        title += f'\n[Filter: {query_string}]'
    ax.set_title(title, fontsize=14)

    ax.set_ylabel("Count")
    ax.set_xlabel("")
    ax.yaxis.grid(True, linestyle='--', alpha=0.6)
    
    # Extend y-axis
    y_max = max(value_counts.values)
    ax.set_ylim(top=y_max * 1.12)

    # Add count and percentage on top of bars
    for i, count in enumerate(value_counts.values):
        percentage = 100 * count / total
        label = f"{count} ({percentage:.2g}%)"
        ax.text(i, count + total * 0.02, label, ha='center', va='bottom', fontsize=10)

    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    # If file name is not provided, generate one
    if not file_name:
        safe_query = re.sub(r"[^\w\-]+", "_", query_string) if query_string else ""
        file_name = f"{column_name}__{safe_query}.jpg" if safe_query else f"{column_name}.jpg"

    output_path = os.path.join(output_folder, file_name)
    plt.savefig(output_path, format='jpg')

    # Show and close plot
    plt.show()
    plt.close()
    print(f"Plot saved to: {output_path}")

    
def plot_missing_values(df, output_folder="../plots/", file_name="", query_string=None):
    # Apply query if specified
    if query_string:
        try:
            df = df.query(query_string)
        except Exception as e:
            print(f"Invalid query string: {e}")
            return

    # Ensure output directory exists
    os.makedirs(output_folder, exist_ok=True)

    # Count and sort missing values
    total_rows = len(df)
    missing_counts = df.isna().sum()
    missing_counts = missing_counts[missing_counts > 0].sort_values(ascending=False)

    # Exit if no missing values
    if missing_counts.empty:
        print("No missing values to plot.")
        return

    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(
        missing_counts.index,
        missing_counts.values,
        color="#C6DCEC",
        edgecolor="#277CB7"
    )

    # Add text labels above bars
    for i, val in enumerate(missing_counts.values):
        pct = 100 * val / total_rows
        label = f"{val} ({pct:.2g}%)"
        ax.text(i, val + total_rows * 0.01, label, ha='center', va='bottom', fontsize=10)

    # Title and styling
    title = "Missing Values Per Column"
    if query_string:
        title += f"\n[Filter: {query_string}]"
    ax.set_title(title, fontsize=14)
    ax.set_ylabel("Number of Missing Values")
    ax.set_xlabel("")
    ax.set_ylim(top=max(missing_counts.values) * 1.15)
    ax.yaxis.grid(True, linestyle='--', alpha=0.6)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    # Save plot
    if not file_name:
        safe_query = re.sub(r"[^\w\-]+", "_", query_string) if query_string else ""
        file_name = f"missing_values__{safe_query}.jpg" if safe_query else "missing_values.jpg"
    output_path = os.path.join(output_folder, file_name)
    plt.savefig(output_path, format='jpg')

    # Show and close
    plt.show()
    plt.close()
    print(f"Plot saved to: {output_path}")
    
def plot_correlation_matrix(df, plot_name="correlation_matrix", output_folder="../plots", query_string=None):
    # Apply query if provided
    if query_string:
        df = df.query(query_string)
        
    # Ensure output directory exists
    os.makedirs(output_folder, exist_ok=True)
    
    # Create the correlation matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(df.select_dtypes(include='number').corr(), annot=True, cmap='coolwarm', fmt=".2f", vmin=-1, vmax=1)
    plt.title("Correlation Heatmap (Numeric Features)")
    
    # Save the plot
    plot_path = f"{output_folder}/{plot_name}.jpg"
    plt.savefig(plot_path, format='jpg', bbox_inches='tight')
    
    # Show the plot
    plt.show()

    print(f"Plot saved as {plot_path}")

def expand_to_long_format(df, time_var="n_cycles_trying", outcome="outcome_pregnant"):
    rows = []
    for idx, row in df.iterrows():
        n_cycles = int(row[time_var])
        for cycle in range(1, n_cycles + 1):
            new_row = row.copy()
            new_row["cycle"] = cycle
            new_row["event_this_cycle"] = int((cycle == n_cycles) and row[outcome] == 1)
            rows.append(new_row)
    return pd.DataFrame(rows)


def prepare_model_matrix(df, 
                         categorical_baselines={}, 
                         center_numeric=[], 
                         leave_unchanged=[], 
                         drop_first=True):
    """
    Prepare a model matrix with specified categorical baselines, centered numeric variables, 
    and columns to leave unchanged.
    
    Parameters:
        df (pd.DataFrame): Original DataFrame
        categorical_baselines (dict): {column_name: reference_category}
        center_numeric (list): List of numeric variable names to center
        leave_unchanged (list): List of variable names to leave unchanged
        drop_first (bool): Whether to drop the baseline dummy (recommended for statsmodels)
    
    Returns:
        pd.DataFrame: X matrix ready for modeling
    """
    df_processed = df.copy()

    # Determine which columns should remain
    allowed_columns = set(categorical_baselines.keys()) | set(center_numeric) | set(leave_unchanged)

    # Keep only the allowed columns in the DataFrame
    df_processed = df_processed[[col for col in df_processed.columns if col in allowed_columns]]

    # Set baselines for categorical variables
    for col, baseline in categorical_baselines.items():
        categories = df_processed[col].dropna().unique().tolist()
        if baseline not in categories:
            raise ValueError(f"Baseline '{baseline}' not found in column '{col}'")
        categories = [baseline] + [c for c in categories if c != baseline]
        df_processed[col] = pd.Categorical(df_processed[col], categories=categories, ordered=False)

    # Create dummy variables for categorical columns
    cat_cols = list(categorical_baselines.keys())
    dummies = pd.get_dummies(df_processed[cat_cols], drop_first=drop_first)

    # Center numeric variables
    for col in center_numeric:
        if col in df_processed.columns:
            df_processed[col + "_centered"] = df_processed[col] - df_processed[col].mean()

    # Combine all variables
    other_vars = [col for col in df_processed.columns if col not in cat_cols and col not in center_numeric]
    final_df = pd.concat([df_processed[other_vars], dummies], axis=1)

    return final_df

def drop_from_summary(summary, col):
    if col in summary.index:
        summary.drop(col, inplace=True)
    return summary

def plot_odds_ratios(result, output_folder="../plots", plot_name="OR",
                     figsize=(8, None), title="Odds Ratios with 95% CI",
                     format="jpg", log_scale=True, show_cycle=False):
    if isinstance(format, str):
        format = [format]

    os.makedirs(output_folder, exist_ok=True)

    summary = result.summary2().tables[1]
    summary["Odds Ratio"] = np.exp(summary["Coef."])
    summary["CI Lower"] = np.exp(summary["Coef."] - 1.96 * summary["Std.Err."])
    summary["CI Upper"] = np.exp(summary["Coef."] + 1.96 * summary["Std.Err."])
    summary["Significant"] = summary["P>|z|"] < 0.05

    exclude_vars = ['const']
    if not show_cycle:
        exclude_vars += ['cycle', 'cycle_centered',
                         'cycle_cat_4-6', 'cycle_cat_7-9', 'cycle_cat_10-12', 'cycle_cat_13+']
    for val in exclude_vars:
        summary = drop_from_summary(summary, val)

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

    # Set up y-axis labels and colors
    ax.set_yticks(range(len(summary)))
    ax.set_yticklabels(summary.index)
    for tick, name in zip(ax.get_yticklabels(), summary.index):
        tick.set_color("red" if summary.loc[name, "Significant"] else "#0072B2")

    # Final plot styling
    ax.axvline(1, color="gray", linestyle="--")
    ax.set_xlabel("Odds Ratio" + (" (log scale)" if log_scale else ""))
    if log_scale:
        ax.set_xscale("log")
    ax.set_title(title)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
    ax.invert_yaxis()

    # Add legend
    legend_elements = [
        Line2D([0], [0], marker='o', color='red', label='Significant (p < 0.05)', linestyle=''),
        Line2D([0], [0], marker='o', color='#0072B2', label='Not significant', linestyle='')
    ]
    ax.legend(handles=legend_elements, loc='lower left')

    plt.tight_layout()
    for ext in format:
        path = os.path.join(output_folder, f"{plot_name}.{ext}")
        plt.savefig(path, format=ext, dpi=300)

    plt.show()
    
    
def prepare_cox_model_matrix(df,
                             duration_col='duration',
                             event_col='event',
                             categorical_baselines={},
                             numerical_features=[],
                             drop_first=True):
    """
    Prepares model matrix for Cox proportional hazards modeling.

    Parameters:
        df (pd.DataFrame): Original DataFrame
        duration_col (str): Column name for duration/time
        event_col (str): Column name for event indicator (1 = event occurred)
        categorical_baselines (dict): {col: baseline_level}
        numerical_features (list): List of numeric variable names to scale
        drop_first (bool): Whether to drop the first level of categorical vars

    Returns:
        pd.DataFrame: Transformed features with duration and event columns
    """
    df = df.copy()

    # Automatically convert all boolean columns to strings
    bool_cols = df.select_dtypes(include=bool).columns
    df[bool_cols] = df[bool_cols].astype(str)

    # All features to consider
    categorical_features = list(categorical_baselines.keys())
    all_features = numerical_features + categorical_features

    # Drop rows with missing values
    df = df.dropna(subset=all_features + [duration_col, event_col])

    # Ensure proper category ordering and track categories explicitly
    category_levels = []
    for col, baseline in categorical_baselines.items():
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in DataFrame")
        
        # Convert column and baseline to string to avoid numeric or bool issues
        df[col] = df[col].astype(str)
        baseline = str(baseline)

        categories = df[col].dropna().unique().tolist()
        if baseline not in categories:
            raise ValueError(f"Baseline '{baseline}' not found in column '{col}'")
        reordered = [baseline] + [c for c in categories if c != baseline]
        df[col] = pd.Categorical(df[col], categories=reordered, ordered=False)
        category_levels.append(reordered)

    # Preprocessing pipeline
    encoder = OneHotEncoder(
        categories=category_levels,
        drop='first' if drop_first else None,
        handle_unknown='ignore'
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', encoder, categorical_features)
        ]
    )

    # Fit and transform
    X = preprocessor.fit_transform(df[all_features])

    # Get proper column names
    def get_feature_names(transformer):
        feature_names = []
        for name, trans, cols in transformer.transformers_:
            if name == 'num':
                feature_names.extend(cols)
            elif name == 'cat':
                encoder = trans
                if hasattr(encoder, 'get_feature_names_out'):
                    feature_names.extend(encoder.get_feature_names_out(cols))
                else:
                    feature_names.extend(encoder.get_feature_names(cols))
        return feature_names

    feature_names = get_feature_names(preprocessor)

    # Create DataFrame with features
    X_df = pd.DataFrame(X.toarray() if hasattr(X, "toarray") else X, columns=feature_names, index=df.index)

    # Add survival columns
    X_df[duration_col] = df[duration_col]
    X_df[event_col] = df[event_col]

    return X_df


def plot_hazard_ratios(cph_result, output_folder="../plots", plot_name="HR",
                       figsize=(8, None), title="Hazard Ratios with 95% CI",
                       format="jpg", log_scale=False, exclude_vars=None):
    if isinstance(format, str):
        format = [format]

    os.makedirs(output_folder, exist_ok=True)

    summary = cph_result.summary.copy()
    summary["HR"] = np.exp(summary["coef"])
    summary["CI Lower"] = np.exp(summary["coef lower 95%"])
    summary["CI Upper"] = np.exp(summary["coef upper 95%"])
    summary["Significant"] = summary["p"] < 0.05

    if exclude_vars is not None:
        summary = summary[~summary.index.isin(exclude_vars)]

    summary = summary.sort_values("HR")
    fig_height = figsize[1] if figsize[1] is not None else 0.4 * len(summary)
    fig, ax = plt.subplots(figsize=(figsize[0], fig_height))

    for i, (name, row) in enumerate(summary.iterrows()):
        color = "red" if row["Significant"] else "#0072B2"
        ax.errorbar(
            x=row["HR"],
            y=i,
            xerr=[[row["HR"] - row["CI Lower"]], [row["CI Upper"] - row["HR"]]],
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
    ax.set_xlabel("Hazard Ratio" + (" (log scale)" if log_scale else ""))
    if log_scale:
        ax.set_xscale("log")
    ax.set_title(title)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
    ax.invert_yaxis()

    legend_elements = [
        Line2D([0], [0], marker='o', color='red', label='Significant (p < 0.05)', linestyle=''),
        Line2D([0], [0], marker='o', color='#0072B2', label='Not significant', linestyle='')
    ]
    ax.legend(handles=legend_elements, loc='lower left')

    plt.tight_layout()
    for ext in format:
        path = os.path.join(output_folder, f"{plot_name}.{ext}")
        plt.savefig(path, format=ext, dpi=300)

    plt.show()
