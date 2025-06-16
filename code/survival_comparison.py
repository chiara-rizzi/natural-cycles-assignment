import pandas as pd
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter
import os

from grouping_config import baseline_categories

PLOT_DIR = "../plots"

# Line and CI fill color schemes
COLORS = [
    "#277CB7",  # blue line
    "#F28E2B",  # orange line
    "#59A14F",  # green line
    "#E15759",  # red/pink line
    "#B07AA1",  # purple
    "#76B7B2",  # teal
    "#FF9DA7",  # soft pink
    "#9C755F",  # brown
    "#EDC948",  # yellow
]

CI_COLORS = [
    "#C6DCEC",  # light blue fill
    "#FDD9B5",  # light orange fill
    "#CDEACA",  # light green fill
    "#F5C5C6",  # light red fill
    "#E2CCE5",  # light purple fill
    "#D4EBEA",  # light teal fill
    "#FBD9DE",  # lighter pink fill
    "#E4D3C8",  # light brown fill
    "#FFF4C2",  # light yellow fill
]

# Remove 'cycle_cat' from baseline definition
BASELINE = {k: v for k, v in baseline_categories.items() if k != "cycle_cat"}


def plot_survival_comparison(data, by_category=None, group_queries=None, group_labels=None, n_cycles=13, show_plot=True):
    """
    Plot multiple Kaplan-Meier survival curves for comparison with confidence intervals.
    """
    if isinstance(data, str):
        df = pd.read_csv(data)
    else:
        df = data.copy()

    fig, ax = plt.subplots(figsize=(10, 5))
    kmf = KaplanMeierFitter()

    if by_category:
        categories = df[by_category].dropna().unique()
        for i, category in enumerate(sorted(categories)):
            mask = df[by_category] == category
            durations = df.loc[mask, 'n_cycles_trying']
            events = df.loc[mask, 'outcome_pregnant']
            label = str(category)

            kmf.fit(durations, events, label=label)
            kmf.plot_survival_function(ci_show=False, linewidth=2, color=COLORS[i % len(COLORS)], ax=ax)

            ci_df = kmf.confidence_interval_
            times = ci_df.index
            lower = ci_df.iloc[:, 0]
            upper = ci_df.iloc[:, 1]
            ax.fill_between(
                times, lower, upper,
                color=CI_COLORS[i % len(CI_COLORS)],
                alpha=0.3,
                step="post"
            )
    elif group_queries and group_labels:
        for i, (query, label) in enumerate(zip(group_queries, group_labels)):
            subset = df.query(query)
            durations = subset['n_cycles_trying']
            events = subset['outcome_pregnant']

            kmf.fit(durations, events, label=label)
            kmf.plot_survival_function(ci_show=False, linewidth=2, color=COLORS[i % len(COLORS)], ax=ax)

            ci_df = kmf.confidence_interval_
            times = ci_df.index
            lower = ci_df.iloc[:, 0]
            upper = ci_df.iloc[:, 1]
            ax.fill_between(
                times, lower, upper,
                color=CI_COLORS[i % len(CI_COLORS)],
                alpha=0.3,
                step="post"
            )
    else:
        raise ValueError("You must specify either `by_category` or both `group_queries` and `group_labels`.")

    title = "Kaplan-Meier Survival Comparison"
    if by_category:
        title += f"\nby '{by_category}'"
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Number of Cycles Trying")
    ax.set_ylabel("Probability of NOT Being Pregnant")
    ax.set_xlim(0, n_cycles + 2)
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(title=by_category if by_category else "Groups")

    plt.tight_layout()
    os.makedirs(PLOT_DIR, exist_ok=True)

    filename = f"survival_comparison_{by_category}.jpg" if by_category else "survival_comparison_custom.jpg"
    output_path = os.path.join(PLOT_DIR, filename)
    plt.savefig(output_path, format='jpg')
    if show_plot:
        plt.show()
    plt.close()
    print(f"Saved: {output_path}")


def run_survival_comparison(data, n_cycles=13, show_plot=True):
    """
    Plot survival comparison for categorical/boolean columns in baseline_categories (excluding 'cycle_cat').
    """
    if isinstance(data, str):
        df = pd.read_csv(data)
    else:
        df = data.copy()

    cols_to_check = [col for col in BASELINE if col in df.columns]
    for col in cols_to_check:
        if df[col].nunique(dropna=True) > 1:
            #print(f"Plotting survival comparison for: {col}")
            plot_survival_comparison(df, by_category=col, n_cycles=n_cycles, show_plot=show_plot)

