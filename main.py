# MAIN PROGRAM
# Your code starts here. This is the file that will be run.
# =====================================================================
# Assignment 3: Optimizing monoclonal antibody production.
#
# Workflow
# --------
# 1. Choose the 30 batches to "download" (space-filling design) -> selected_batches.csv
# 2. Load only those 30 batch files from provided_files/
# 3. Compare VAR / polynomial NVAR / k-NN NVAR models with LOBOCV
# 4. Fit the best model on all 30 batches
# 5. Forecast every initial condition (7 x 7 x 7 levels) and find the combination
#    of initial glucose, glutamine, VCD and harvest time with the highest
#    space-time profit
# 6. Quantify uncertainty of the optimum with 1000 noisy simulations

import os

import numpy as np
import pandas as pd

import src.functions as sf  # access functions with sf.function_name()
import src.constants as c
from src.classes import FeatureScaler


def main():
    rng = np.random.default_rng(c.SEED)
    os.makedirs(c.FIGURES_DIR, exist_ok=True)
    os.makedirs(c.TABLES_DIR, exist_ok=True)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 20)

    # ------------------------------------------------------------------
    # 1. Select the 30 batches (only done once so the budget is respected)
    # ------------------------------------------------------------------
    ic = sf.load_initial_conditions()
    if c.USE_DOWNLOADED_FILES:
        selected = sf.selection_from_files(ic)
        print(f"Using the {len(selected)} batch files found in '{c.PROVIDED_DIR}'")
        if len(selected) > c.N_BATCHES:
            raise SystemExit(f"{len(selected)} files found but the budget is {c.N_BATCHES}.")
    elif os.path.exists(c.SELECTED_FILE):
        selected = pd.read_csv(c.SELECTED_FILE)
        print(f"Using the existing selection in {c.SELECTED_FILE}")
    else:
        selected = sf.design_batches(ic, c.N_BATCHES, rng, c.N_DESIGNS)
        selected.to_csv(c.SELECTED_FILE, index=False)
        print(f"New design saved to {c.SELECTED_FILE}")
    print(f"\nSelected {len(selected)} batches:")
    print(selected.to_string(index=False))

    missing = sf.check_files(selected)
    if missing:
        raise SystemExit(
            f"\n{len(missing)} of the selected batch files are not in '{c.PROVIDED_DIR}':\n"
            f"{missing}\nDownload them, or set USE_DOWNLOADED_FILES = True in "
            "src/constants.py to use the files you already have.")

    # ------------------------------------------------------------------
    # 2. Load the data of the selected batches only
    # ------------------------------------------------------------------
    batches = sf.load_batches(selected)
    series = [b.data for b in batches]
    time_h = batches[0].time_h
    batches[0].display_info()
    sf.save_table(selected, "selected_batches")
    sf.plot_training_data(batches, "training_data.png")
    sf.save_table(sf.observed_table(series, selected, time_h), "observed_batches")
    print(f"\nLoaded {len(series)} batches with {len(time_h)} time points "
          f"({time_h[0]:g} - {time_h[-1]:g} h)")

    # ------------------------------------------------------------------
    # 3. Model selection with leave-one-batch-out cross-validation
    # ------------------------------------------------------------------
    print("\nLOBOCV model comparison (full-batch forecasts):")
    table = sf.compare_models(series, rng)
    print("\nRanking:")
    print(table.drop(columns="cfg").to_string(index=False, float_format="%.2f"))
    sf.save_table(table.drop(columns="cfg"), "model_comparison")

    best_cfg = table.loc[0, "cfg"]
    p = best_cfg["p"]
    print(f"\nBest model: {sf.config_label(best_cfg)}")
    sf.plot_cv_forecast(series, time_h, best_cfg, rng, "cv_forecasts.png")

    # ------------------------------------------------------------------
    # 4. Fit the best model on all 30 batches
    # ------------------------------------------------------------------
    scaler = FeatureScaler().fit(series)
    x_lag, x_cur = sf.build_lagged(scaler.transform_list(series), p)
    model = sf.build_model(best_cfg).fit(x_lag, x_cur)
    residuals = x_cur - model.predict(x_lag)  # white noise source

    # ------------------------------------------------------------------
    # 5. Search the whole initial-condition grid
    # ------------------------------------------------------------------
    conditions = sf.candidate_conditions(ic, selected)
    print(f"\nForecasting {len(conditions)} candidate initial conditions "
          "(inside the range covered by the training batches) ...")
    ranking = sf.search_optimum(model, p, scaler, residuals, conditions, time_h, rng)
    sf.save_table(ranking.head(20), "optimization_ranking_top20")
    sf.save_table(sf.sensitivity_table(ranking), "initial_condition_effects")
    sf.plot_condition_effects(ranking, "initial_condition_effects.png")
    sf.plot_heatmaps(ranking, "profit_heatmaps.png")
    print(f"\nTop {c.N_TOP} conditions (median forecast):")
    print(ranking.head(c.N_TOP).to_string(index=False, float_format="%.4g"))

    # ------------------------------------------------------------------
    # 6. Uncertainty of the optimum
    # ------------------------------------------------------------------
    best = ranking.iloc[0]
    glc, gln, vcd = best[c.COL_GLC], best[c.COL_GLN], best[c.COL_VCD]
    sims = sf.forecast_with_uncertainty(model, p, scaler, residuals, glc, gln, vcd, time_h, rng)
    result, t_best = sf.recommended_table(sims, glc, gln, vcd, time_h)
    sf.save_table(result, "recommended_conditions")
    sf.plot_optimum(time_h, sims, glc, gln, vcd, "optimum_forecast.png")

    obs = sf.observed_best(series, selected, time_h)

    print("\n" + "=" * 70)
    print("RECOMMENDED OPERATING CONDITIONS")
    print("=" * 70)
    print(f"Model: {sf.config_label(best_cfg)}\n")
    print(sf.to_markdown(result))
    print("-" * 70)
    print("Best batch actually measured among the training runs:")
    print(f"  Batch {obs['Batch ID']}: glucose {obs[c.COL_GLC]:g} mM, glutamine "
          f"{obs[c.COL_GLN]:g} mM, VCD {obs[c.COL_VCD]:.4g} cells/mL, "
          f"harvest {obs['Harvest time [h]']:g} h, STP {obs['stp']:.4g} $/(L d)")
    print(f"\nFigures saved in '{c.FIGURES_DIR}/', tables in '{c.TABLES_DIR}/'.")


if __name__ == "__main__":
    main()
