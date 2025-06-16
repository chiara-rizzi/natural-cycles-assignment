import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from lifelines import CoxPHFitter
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from grouping_config import baseline_categories

PLOT_DIR = "../plots"

def drop_from_summary(summary, col):
    """Drop a variable from the summary table if it exists."""
    if col in summary.index:
        summary.drop(col, inplace=True)
    return summary

def plot_hazard_ratios(cph_result, output_folder="../plots", plot_name="HR",
                       figsize=(8, None), title="Hazard Ratios with 95% CI",
                       format="jpg", log_scale=False, exclude_vars=None):
    """Plot hazard ratios with 95% confidence intervals from a Cox model result."""
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

def prepare_cox_model_matrix(df, duration_col='duration', event_col='event',
                             categorical_baselines={}, numerical_features=[], drop_first=True):
    """Prepare design matrix for Cox model with one-hot encoding and scaling."""
    df = df.copy()

    # Handle boolean columns by converting them to string
    bool_cols = df.select_dtypes(include=bool).columns
    df[bool_cols] = df[bool_cols].astype(str)

    categorical_features = list(categorical_baselines.keys())
    all_features = numerical_features + categorical_features

    # Drop rows with missing data in relevant columns
    df = df.dropna(subset=all_features + [duration_col, event_col])

    # Relevel categorical variables
    category_levels = []
    for col, baseline in categorical_baselines.items():
        df[col] = df[col].astype(str)
        baseline = str(baseline)
        categories = df[col].dropna().unique().tolist()
        if baseline not in categories:
            raise ValueError(f"Baseline '{baseline}' not found in column '{col}'")
        reordered = [baseline] + [c for c in categories if c != baseline]
        df[col] = pd.Categorical(df[col], categories=reordered, ordered=False)
        category_levels.append(reordered)

    # Set up transformers
    encoder = OneHotEncoder(categories=category_levels,
                            drop='first' if drop_first else None,
                            handle_unknown='ignore')
    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), numerical_features),
        ('cat', encoder, categorical_features)
    ])

    # Fit and transform
    X = preprocessor.fit_transform(df[all_features])

    # Extract feature names
    def get_feature_names(transformer):
        feature_names = []
        for name, trans, cols in transformer.transformers_:
            if name == 'num':
                feature_names.extend(cols)
            elif name == 'cat':
                if hasattr(trans, 'get_feature_names_out'):
                    feature_names.extend(trans.get_feature_names_out(cols))
                else:
                    feature_names.extend(trans.get_feature_names(cols))
        return feature_names

    feature_names = get_feature_names(preprocessor)
    X_df = pd.DataFrame(X.toarray() if hasattr(X, "toarray") else X, columns=feature_names, index=df.index)
    X_df[duration_col] = df[duration_col]
    X_df[event_col] = df[event_col]
    return X_df

def run_cox_model(data):
    """
    Fit a Cox proportional hazards model to analyze time-to-pregnancy.

    Parameters:
        data (str or pd.DataFrame): Path to input data or DataFrame.

    Returns:
        cph: Fitted CoxPHFitter model object
    """
    # Load data
    if isinstance(data, str):
        df = pd.read_csv(data)
    else:
        df = data.copy()

    print("Input data loaded, shape:", df.shape)

    # Define Cox input columns
    df['event'] = df['outcome_pregnant']
    df['duration'] = df['n_cycles_trying']

    # Define numeric and categorical variables
    numerical_vars = []  # Add numerical features if needed

    # Prepare model matrix
    print("Preparing Cox model matrix...")
    # remove cycle_cat, used for logistic regression
    baseline_categories.pop('cycle_cat', None) 
    X_df = prepare_cox_model_matrix(df, duration_col='duration', event_col='event',
                                    categorical_baselines=baseline_categories,
                                    numerical_features=numerical_vars)

    print("Model matrix shape:", X_df.shape)

    # Fit Cox model
    cph = CoxPHFitter()
    cph.fit(X_df, duration_col='duration', event_col='event')

    print("Cox model fit complete.")
    print(cph.summary)

    # Plot HR
    plot_hazard_ratios(cph, output_folder=PLOT_DIR, plot_name="cox_hr_plot")

    return cph
