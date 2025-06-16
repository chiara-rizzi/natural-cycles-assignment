import pandas as pd
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter
import os

PLOT_DIR = "../plots"


def run_survival_analysis(data, n_cycles=13):
    """
    Run Kaplan-Meier survival analysis on the given dataset and plot results.

    Parameters:
        data_path (str): Path to the dataset.
        n_cycles (int): Number of cycles to compute cumulative probability.

    Returns:
        kmf (KaplanMeierFitter): Fitted Kaplan-Meier model.
    """
    if isinstance(data, str):
        df = pd.read_csv(data)
    else:
        df = data.copy()

    time = df['n_cycles_trying']
    event = df['outcome_pregnant']

    kmf = KaplanMeierFitter()
    kmf.fit(durations=time, event_observed=event)

    # Handle n_cycles as a list or single value
    if isinstance(n_cycles, (list, tuple)):
        ref_cycle = n_cycles[0]
        all_cycles = n_cycles
    else:
        ref_cycle = n_cycles
        all_cycles = [n_cycles]

    for cycle in all_cycles:
        surv_prob = kmf.survival_function_at_times(cycle).values[0]
        pregnancy_prob = 1 - surv_prob
        if cycle in kmf.confidence_interval_.index:
            ci = kmf.confidence_interval_.loc[cycle]
        else:
            ci = kmf.confidence_interval_.interpolate().loc[cycle]
        lower = 1 - ci.iloc[1]
        upper = 1 - ci.iloc[0]

        print(f"Probability of pregnancy within {cycle} cycles: {pregnancy_prob:.2%} ({round(pregnancy_prob*100)}%)")
        print(f"95% CI: [{lower:.2%} - {upper:.2%}] ({round(lower*100)}% - {round(upper*100)}%)")

    # Plot survival function
    os.makedirs(PLOT_DIR, exist_ok=True)
    plt.figure(figsize=(7, 5))
    kmf.plot_survival_function(ci_show=True, linewidth=2)
    #plt.axvline(ref_cycle, color='red', linestyle='--', label=f'{ref_cycle} cycles')
    plt.title("Kaplan-Meier Survival Curve with 95% Confidence Interval")
    plt.xlabel("Number of Cycles Trying")
    plt.ylabel("Probability of NOT Being Pregnant")
    plt.xlim(0, ref_cycle + 2)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    output_path=os.path.join(PLOT_DIR, "Q1_survival_curve.jpg")
    plt.savefig(output_path)
    print(f"Saved: {output_path}")
    plt.close()

    # Plot cumulative density
    plt.figure(figsize=(7, 5))
    kmf.plot_cumulative_density(ci_show=True, linewidth=2)
    #plt.axvline(ref_cycle, color='red', linestyle='--', label=f'{ref_cycle} cycles')
    plt.title("Cumulative Probability of Pregnancy with 95% Confidence Interval")
    plt.xlabel("Number of Cycles Trying")
    plt.ylabel("Probability of Being Pregnant")
    plt.xlim(0, ref_cycle + 2)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    output_path=os.path.join(PLOT_DIR, "Q1_pregnancy_probability.jpg")
    plt.savefig(output_path)
    print(f"Saved: {output_path}")
    plt.close()

    # Median time to pregnancy
    median_time = kmf.median_survival_time_
    print(f"Median number of cycles to pregnancy: {median_time:.1f}")

    return kmf
