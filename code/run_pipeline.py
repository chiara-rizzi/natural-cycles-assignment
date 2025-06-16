from eda import run_eda
from imputation import run_imputation
from survival_analysis import run_survival_analysis
from regression import run_logistic_regression
from cox_model import run_cox_model


def run_pipeline(data_path="../data/ncdatachallenge-2021-v1.csv"):
    print("\n=== Step 1: Exploratory Data Analysis ===")
    df_full = run_eda(data_path)

    print("\n=== Step 2: Imputation ===")
    imputed_df = run_imputation(
        numeric_path="../data/modified_numeric.csv",
        output_path_small="../data/modified_small_imputed.csv",
        output_path_full="../data/modified_full_imputed.csv"
    )

    print("\n=== Step 3: Survival Analysis ===")
    run_survival_analysis(imputed_df, n_cycles=13)

    print("\n=== Step 4: Logistic Regression ===")
    regression_results = run_logistic_regression(imputed_df)

    print("\n=== Step 5: Cox Proportional Hazards Model ===")
    cox_results = run_cox_model(imputed_df)

    print("\nPipeline complete.")
    return {
        "regression": regression_results,
        "cox": cox_results
    }


if __name__ == "__main__":
    import argparse
    import matplotlib
    import warnings

    # Use a non-interactive backend
    matplotlib.use("Agg")
    warnings.filterwarnings(
        "ignore",
        message="FigureCanvasAgg is non-interactive, and thus cannot be shown"
    )

    parser = argparse.ArgumentParser(description="Run analysis pipeline or specific step.")
    parser.add_argument("--step", type=str, choices=["eda", "impute", "survival", "regression", "cox"], help="Step to run.")
    parser.add_argument("--csv", type=str, help="Path to CSV file if running a single step.")
    args = parser.parse_args()

    if not args.step:
        csv_path = args.csv if args.csv else "../data/ncdatachallenge-2021-v1.csv"
        print(f"Running full pipeline using: {csv_path}")
        run_pipeline(data_path=csv_path)
    else:
        if not args.csv:
            raise ValueError("--csv argument is required when running a specific step.")

        print(f"Running step: {args.step} on file: {args.csv}")

        if args.step == "eda":
            run_eda(args.csv)
        elif args.step == "impute":
            run_imputation(
                numeric_path=args.csv,
                output_path_small="../data/modified_small_imputed.csv",
                output_path_full="../data/modified_full_imputed.csv"
            )
        elif args.step == "survival":
            run_survival_analysis(args.csv, n_cycles=[13, 6, 12])
        elif args.step == "regression":
            run_logistic_regression(args.csv)
        elif args.step == "cox":
            run_cox_model(args.csv)
