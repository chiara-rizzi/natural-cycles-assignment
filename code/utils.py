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
    print(f"Saved: {output_path}") 

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
    print(f"Saved: {output_path}")

    
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
    print(f"Saved: {output_path}")
    
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

    print(f"Saved: {plot_path}")
