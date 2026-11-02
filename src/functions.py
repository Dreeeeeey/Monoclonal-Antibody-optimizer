# FUNCTIONS
# Define reusable helper functions in this file.

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures

import src.constants as c
from src.classes import Batch, FeatureScaler


# ============================================================================
# 1. Data loading and batch selection
# ============================================================================
def load_initial_conditions():
    """Read initial_conditions.csv (1029 rows, one per batch)."""
    return pd.read_csv(c.INITIAL_CONDITIONS_FILE)


def design_batches(ic, n_batches, rng, n_designs):
    """Choose which batches to run (space-filling design).

    Each initial condition has 7 levels. We want the 30 runs to
      (1) cover every level of every variable about equally (Latin-hypercube
          style: each level is used 4-5 times), and
      (2) be as spread out as possible in the 3-D space.
    We generate many random Latin-hypercube-style designs and keep the one with
    the largest minimum distance between any two runs (maximin).
    """
    levels = [np.sort(ic[col].unique()) for col in c.IC_COLS]
    n_levels = [len(lev) for lev in levels]

    best_idx, best_dist = None, -1.0
    for _ in range(n_designs):
        idx = np.column_stack(
            [rng.permutation(np.arange(n_batches) % n) for n in n_levels]
        )
        scaled = idx / (np.array(n_levels) - 1)
        diff = scaled[:, None, :] - scaled[None, :, :]
        dist = np.sqrt((diff ** 2).sum(axis=2))
        np.fill_diagonal(dist, np.inf)
        if dist.min() > best_dist:
            best_dist, best_idx = dist.min(), idx

    # Convert each design point (level indices) to an actual batch ID.
    # There are 3 replicates per combination, so pick one at random.
    chosen = []
    for row in best_idx:
        mask = np.ones(len(ic), dtype=bool)
        for j, col in enumerate(c.IC_COLS):
            mask &= np.isclose(ic[col].to_numpy(), levels[j][row[j]])
        options = ic[mask]
        chosen.append(options.iloc[rng.integers(len(options))])
    return pd.DataFrame(chosen).reset_index(drop=True)


def index_batch_files(root=None):
    """Find every batch_####.csv under `root`, including all subfolders.

    Returns a dict {file name: full path}. The batch files do not have to be in
    one folder.
    """
    root = c.PROVIDED_DIR if root is None else root
    found = {}
    for folder, _, files in os.walk(root):
        for name in files:
            if name.startswith("batch_") and name.endswith(".csv"):
                found[name] = os.path.join(folder, name)
    return found


_BATCH_FILES = None  # filled the first time a batch is loaded


def selection_from_files(ic):
    """Use the batch files that are already in PROVIDED_DIR.

    Returns the rows of initial_conditions.csv whose batch file exists.
    """
    files = index_batch_files()
    ids = [int(name[6:-4]) for name in files]  # 'batch_0436.csv' -> 436
    selected = ic[ic[c.COL_BATCH].astype(int).isin(ids)]
    return selected.sort_values(c.COL_BATCH).reset_index(drop=True)


def check_files(selected):
    """Return the batch IDs of `selected` whose files cannot be found."""
    files = index_batch_files()
    return [int(b) for b in selected[c.COL_BATCH]
            if f"batch_{int(b):04d}.csv" not in files]


def load_batch(batch_id):
    """Read one batch_####.csv file as a DataFrame."""
    global _BATCH_FILES
    if _BATCH_FILES is None:
        _BATCH_FILES = index_batch_files()
    name = f"batch_{int(batch_id):04d}.csv"
    if name not in _BATCH_FILES:
        raise FileNotFoundError(
            f"{name} not found anywhere under '{c.PROVIDED_DIR}' "
            f"({len(_BATCH_FILES)} batch files were found). "
            "Check PROVIDED_DIR in src/constants.py.")
    return pd.read_csv(_BATCH_FILES[name])


def load_batches(selected):
    """Load the selected batches and return a list of Batch objects."""
    batches = []
    for _, row in selected.iterrows():
        df = load_batch(row[c.COL_BATCH])
        batches.append(Batch(
            batch_id=int(row[c.COL_BATCH]),
            glucose_0=row[c.COL_GLC],
            glutamine_0=row[c.COL_GLN],
            vcd_0=row[c.COL_VCD],
            time_h=df[c.COL_TIME].to_numpy(),
            data=df[c.FEATURES].to_numpy(dtype=float),
        ))
    return batches


# ============================================================================
# 2. Lagged vectors and models
# ============================================================================
def build_lagged(series_list, p):
    """Build x_lag and x_cur for a VAR/NVAR(p) model.

    Every batch is handled separately (the first p rows of a batch are skipped
    since they have no full set of lags). The lag blocks are tiled:
        x_lag = [ x(t-1) | x(t-2) | ... | x(t-p) ]   (each block = all features)
    """
    x_lag_all, x_cur_all = [], []
    for x_b in series_list:
        n = len(x_b)
        x_cur_all.append(x_b[p:])
        x_lag_all.append(np.hstack([x_b[p - i:n - i] for i in range(1, p + 1)]))
    return np.vstack(x_lag_all), np.vstack(x_cur_all)


def build_model(cfg):
    """Create the model f in x(t) = f(x(t-1), ..., x(t-p)) + noise.

    cfg = {"type": "poly", "degree": d, "p": p}  -> polynomial NVAR (d=1: VAR)
    cfg = {"type": "knn",  "k": k,       "p": p} -> k-NN NVAR
    The data are already standardized, so no scaler is needed in the pipeline.
    """
    if cfg["type"] == "poly":
        return make_pipeline(
            PolynomialFeatures(degree=cfg["degree"], include_bias=False),
            LinearRegression(),
        )
    if cfg["type"] == "knn":
        return KNeighborsRegressor(n_neighbors=cfg["k"], weights="uniform")
    raise ValueError(f"Unknown model type: {cfg['type']}")


def config_label(cfg):
    if cfg["type"] == "poly":
        name = "VAR" if cfg["degree"] == 1 else f"Poly(deg={cfg['degree']}) NVAR"
        return f"{name}({cfg['p']})"
    return f"kNN(k={cfg['k']}) NVAR({cfg['p']})"


def candidate_configs():
    configs = []
    for p in c.LAGS:
        for d in c.POLY_DEGREES:
            configs.append({"type": "poly", "degree": d, "p": p})
        for k in c.KNN_NEIGHBOURS:
            configs.append({"type": "knn", "k": k, "p": p})
    return configs


# ============================================================================
# 3. Forecasting with white noise
# ============================================================================
def simulate(model, x_ic, p, n_steps, residuals, n_sims, lower, upper, rng, noise=True):
    """Forecast whole batches starting from initial conditions.

    Parameters
    ----------
    x_ic      : (m, F) standardized initial state of m conditions
    residuals : (n, F) standardized 1-step training residuals
    Returns   : (m, n_sims, n_steps + 1, F) standardized trajectories
    """
    m, n_feat = x_ic.shape
    traj = np.empty((m, n_sims, n_steps + 1, n_feat))
    traj[:, :, 0, :] = x_ic[:, None, :]

    # All p lag blocks start as the initial conditions (x_ic, x_ic, ...).
    lags = np.tile(np.repeat(x_ic[:, None, :], n_sims, axis=1), (1, 1, p))
    lags = lags.reshape(m * n_sims, p * n_feat)

    for t in range(1, n_steps + 1):
        pred = model.predict(lags)
        if noise:
            # Each feature gets noise from ITS OWN residuals.
            idx = rng.integers(0, len(residuals), size=pred.shape)
            pred = pred + np.take_along_axis(residuals, idx, axis=0)
        pred = np.clip(pred, lower, upper)  # no negative values, no divergence
        traj[:, :, t, :] = pred.reshape(m, n_sims, n_feat)
        # Shift the lag blocks: newest prediction first, drop the oldest block.
        lags = np.hstack([pred, lags[:, :(p - 1) * n_feat]])
    return traj


def initial_state(glc, gln, vcd):
    """Raw initial state vectors, shape (m, F).

    Lactate, ammonia and mAb start at 0 and TCD = VCD (known initial state of the culture).
    """
    glc, gln, vcd = (np.atleast_1d(np.asarray(v, dtype=float)) for v in (glc, gln, vcd))
    x = np.zeros((len(glc), c.N_FEATURES))
    x[:, c.IDX_GLC] = glc
    x[:, c.IDX_GLN] = gln
    x[:, c.IDX_VCD] = vcd
    x[:, c.IDX_TCD] = vcd
    return x


# ============================================================================
# 4. Model evaluation: leave-one-batch-out cross-validation
# ============================================================================
def lobocv(cfg, series, n_sims, rng):
    """LOBOCV of the full-batch forecast for one model configuration.

    For each batch: train on the other batches, forecast the held-out batch from
    its initial conditions, and compare with the real data.
    Returns the per-batch RMSE of mAb [mg/L] and of all (standardized) features.
    """
    p = cfg["p"]
    rmse_mab, rmse_all = [], []
    for i in range(len(series)):
        train = [s for j, s in enumerate(series) if j != i]
        scaler = FeatureScaler().fit(train)  # fit on training batches only
        x_lag, x_cur = build_lagged(scaler.transform_list(train), p)
        model = build_model(cfg).fit(x_lag, x_cur)
        residuals = x_cur - model.predict(x_lag)

        test = scaler.transform(series[i])
        traj = simulate(model, test[:1], p, len(test) - 1, residuals,
                        n_sims, scaler.lower_, scaler.upper_, rng)
        forecast = np.median(traj[0], axis=0)  # (time, F), median of sims

        true_raw = series[i]
        pred_raw = scaler.inverse_transform(forecast)
        rmse_mab.append(np.sqrt(np.mean((pred_raw[:, c.IDX_MAB] - true_raw[:, c.IDX_MAB]) ** 2)))
        rmse_all.append(np.sqrt(np.mean((forecast - test) ** 2)))
    return np.array(rmse_mab), np.array(rmse_all)


def compare_models(series, rng):
    """Run LOBOCV for all candidate models and return a sorted summary table."""
    rows = []
    for cfg in candidate_configs():
        rmse_mab, rmse_all = lobocv(cfg, series, c.N_SIMS_CV, rng)
        rows.append({
            "Model": config_label(cfg),
            "mAb RMSE mean [mg/L]": rmse_mab.mean(),
            "mAb RMSE SD [mg/L]": rmse_mab.std(),
            "All-feature RMSE (std.)": rmse_all.mean(),
            "cfg": cfg,
        })
        print(f"  {rows[-1]['Model']:<24} mAb RMSE = "
              f"{rmse_mab.mean():8.2f} +/- {rmse_mab.std():7.2f} mg/L")
    table = pd.DataFrame(rows).sort_values("mAb RMSE mean [mg/L]").reset_index(drop=True)
    return table


# ============================================================================
# 5. Economics
# ============================================================================
def economics(glc, gln, vcd, mab, time_h):
    """Compute the economic metrics.

    glc, gln, vcd : initial conditions, shape (m, 1) or scalars
    mab           : mAb concentration [mg/L], shape (m, T)
    time_h        : harvest times [h], shape (T,)
    Returns dict with cost, value, profit [$/L], and space-time profit
    [$/(L d)] and space-time yield [mg/(L d)] for every condition and time.
    """
    cost = c.COST_GLC * glc + c.COST_GLN * gln + c.COST_VCD * vcd
    value = c.VALUE_MAB * mab
    profit = value - cost
    cycle_d = np.asarray(time_h) / 24.0 + c.DOWNTIME_D
    return {
        "cost": cost,
        "value": value,
        "profit": profit,
        "stp": profit / cycle_d,
        "sty": mab / cycle_d,
    }


# ============================================================================
# 6. Optimization
# ============================================================================
def candidate_conditions(ic, training):
    """All 7 x 7 x 7 initial-condition combinations that lie INSIDE the range
    covered by the training batches (no extrapolation: model validity).

    ic       : initial_conditions DataFrame (gives the possible levels)
    training : DataFrame of the batches used to train the model
    """
    levels = []
    for col in c.IC_COLS:
        lev = np.sort(ic[col].unique())
        lo, hi = training[col].min(), training[col].max()
        levels.append(lev[(lev >= lo) & (lev <= hi)])
    glc, gln, vcd = np.meshgrid(*levels, indexing="ij")
    return np.column_stack([glc.ravel(), gln.ravel(), vcd.ravel()])


def search_optimum(model, p, scaler, residuals, conditions, time_h, rng):
    """Forecast every candidate condition and find the best (conditions, harvest time).

    The median mAb trajectory over N_SIMS_SEARCH simulations is used.
    Returns a DataFrame with the best harvest time and metrics for each condition.
    """
    n_steps = len(time_h) - 1
    results = []
    for start in range(0, len(conditions), c.SEARCH_CHUNK):
        cond = conditions[start:start + c.SEARCH_CHUNK]
        x_ic = scaler.transform(initial_state(cond[:, 0], cond[:, 1], cond[:, 2]))
        traj = simulate(model, x_ic, p, n_steps, residuals,
                        c.N_SIMS_SEARCH, scaler.lower_, scaler.upper_, rng)
        mab = scaler.inverse_transform(traj)[..., c.IDX_MAB]  # (m, sims, T)
        mab_mid = np.median(mab, axis=1)  # (m, T)

        eco = economics(cond[:, [0]], cond[:, [1]], cond[:, [2]], mab_mid, time_h)
        best_t = eco["stp"].argmax(axis=1)
        rows = np.arange(len(cond))
        results.append(pd.DataFrame({
            c.COL_GLC: cond[:, 0], c.COL_GLN: cond[:, 1], c.COL_VCD: cond[:, 2],
            "Harvest time [h]": time_h[best_t],
            "mAb [mg/L]": mab_mid[rows, best_t],
            "Profit [$/L]": eco["profit"][rows, best_t],
            "Space-time profit [$/(L d)]": eco["stp"][rows, best_t],
            "Space-time yield [mg/(L d)]": eco["sty"][rows, best_t],
        }))
    return pd.concat(results, ignore_index=True).sort_values(
        "Space-time profit [$/(L d)]", ascending=False).reset_index(drop=True)


def forecast_with_uncertainty(model, p, scaler, residuals, glc, gln, vcd, time_h, rng):
    """Many noisy simulations of one condition. Returns raw (sims, T, F) array."""
    x_ic = scaler.transform(initial_state(glc, gln, vcd))
    traj = simulate(model, x_ic, p, len(time_h) - 1, residuals,
                    c.N_SIMS_FINAL, scaler.lower_, scaler.upper_, rng)
    return scaler.inverse_transform(traj[0])


def observed_best(series, selected, time_h):
    """Best space-time profit actually observed in the 30 measured batches."""
    best = None
    for s, (_, row) in zip(series, selected.iterrows()):
        eco = economics(row[c.COL_GLC], row[c.COL_GLN], row[c.COL_VCD],
                        s[None, :, c.IDX_MAB], time_h[:len(s)])
        t = eco["stp"][0].argmax()
        if best is None or eco["stp"][0, t] > best["stp"]:
            best = {"Batch ID": int(row[c.COL_BATCH]), "Harvest time [h]": time_h[t],
                    "stp": eco["stp"][0, t], c.COL_GLC: row[c.COL_GLC],
                    c.COL_GLN: row[c.COL_GLN], c.COL_VCD: row[c.COL_VCD]}
    return best


# ============================================================================
# 7. Plots
# ============================================================================
def plot_cv_forecast(series, time_h, cfg, rng, path_name):
    """Plot LOBOCV mAb forecasts (median and 95 % PI) for the first 12 batches."""
    p = cfg["p"]
    n_plot = min(12, len(series))
    fig, axes = plt.subplots(3, 4, figsize=(14, 8), sharex=True)
    for i, ax in zip(range(n_plot), axes.ravel()):
        train = [s for j, s in enumerate(series) if j != i]
        scaler = FeatureScaler().fit(train)
        x_lag, x_cur = build_lagged(scaler.transform_list(train), p)
        model = build_model(cfg).fit(x_lag, x_cur)
        residuals = x_cur - model.predict(x_lag)
        test = scaler.transform(series[i])
        traj = simulate(model, test[:1], p, len(test) - 1, residuals,
                        200, scaler.lower_, scaler.upper_, rng)
        mab = scaler.inverse_transform(traj[0])[..., c.IDX_MAB]
        lo, mid, hi = np.quantile(mab, [0.025, 0.5, 0.975], axis=0)
        ax.fill_between(time_h[:len(lo)], lo, hi, alpha=0.3, label="95% PI")
        ax.plot(time_h[:len(mid)], mid, "k", label="Forecast")
        ax.scatter(time_h[:len(mid)], series[i][:, c.IDX_MAB], s=8, c="tab:orange", label="Test")
        ax.set_title(f"Held-out batch {i + 1}", fontsize=9)
    axes[0, 0].legend(fontsize=7)
    fig.supxlabel("Time [h]")
    fig.supylabel("mAb [mg/L]")
    fig.suptitle(f"Leave-one-batch-out forecasts: {config_label(cfg)}")
    save_figure(fig, path_name)


def plot_optimum(time_h, sims, glc, gln, vcd, path_name):
    """Plot mAb forecast and space-time profit vs harvest time for the optimum."""
    mab = sims[..., c.IDX_MAB]
    lo, mid, hi = np.quantile(mab, [0.025, 0.5, 0.975], axis=0)
    eco_mid = economics(glc, gln, vcd, mid[None, :], time_h)
    eco_lo = economics(glc, gln, vcd, lo[None, :], time_h)
    eco_hi = economics(glc, gln, vcd, hi[None, :], time_h)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].fill_between(time_h, lo, hi, alpha=0.3, label="95% PI")
    axes[0].plot(time_h, mid, "k", label="Median forecast")
    axes[0].set_xlabel("Time [h]")
    axes[0].set_ylabel("mAb [mg/L]")
    axes[0].legend()

    axes[1].fill_between(time_h, eco_lo["stp"][0], eco_hi["stp"][0], alpha=0.3, label="95% PI")
    axes[1].plot(time_h, eco_mid["stp"][0], "k", label="Median")
    t_best = eco_mid["stp"][0].argmax()
    axes[1].axvline(time_h[t_best], color="r", ls="--", label=f"Harvest: {time_h[t_best]:g} h")
    axes[1].set_xlabel("Harvest time [h]")
    axes[1].set_ylabel("Space-time profit [$/(L d)]")
    axes[1].legend()
    fig.suptitle(f"Recommended: glucose {glc:g} mM, glutamine {gln:g} mM, VCD {vcd:.3g} cells/mL")
    save_figure(fig, path_name)


# ============================================================================
# 8. Tables (CSV) and extra figures
# ============================================================================
def to_markdown(df):
    """Convert a DataFrame to a Markdown table (no extra packages needed)."""

    def fmt(v):
        if isinstance(v, (float, np.floating)):
            return f"{v:,.0f}" if abs(v) >= 1e4 else f"{v:.4g}"
        return str(v)

    header = "| " + " | ".join(str(col) for col in df.columns) + " |"
    divider = "| " + " | ".join("---" for _ in df.columns) + " |"
    rows = ["| " + " | ".join(fmt(v) for v in row) + " |" for row in df.to_numpy()]
    return "\n".join([header, divider] + rows) + "\n"


def save_table(df, name):
    """Save a table as tables/<name>.csv."""
    os.makedirs(c.TABLES_DIR, exist_ok=True)
    df.to_csv(os.path.join(c.TABLES_DIR, f"{name}.csv"), index=False)


def save_figure(fig, name):
    os.makedirs(c.FIGURES_DIR, exist_ok=True)
    fig.tight_layout()
    fig.savefig(os.path.join(c.FIGURES_DIR, name), dpi=150)
    plt.close(fig)


def recommended_table(sims, glc, gln, vcd, time_h):
    """Summary of the recommended operating point with 95 % prediction intervals.

    sims : (n_sims, T, F) raw simulated trajectories of the recommended condition
    Returns (table, index of the best harvest time).
    """
    mab = sims[..., c.IDX_MAB]
    eco = economics(glc, gln, vcd, mab, time_h)
    t_best = int(np.argmax(np.median(eco["stp"], axis=0)))

    def interval(values):
        lo, mid, hi = np.quantile(values[:, t_best], [0.025, 0.5, 0.975])
        return mid, lo, hi

    cost = float(eco["cost"])
    rows = [
        ("Initial glucose", glc, None, None, "mM"),
        ("Initial glutamine", gln, None, None, "mM"),
        ("Initial VCD", vcd, None, None, "cells/mL"),
        ("Harvest time", time_h[t_best] / 24.0, None, None, "d"),
        ("Cell culture cost", cost, None, None, "$/L"),
        ("mAb at harvest", *interval(mab), "mg/L"),
        ("Cell culture value at harvest", *interval(eco["value"]), "$/L"),
        ("Profit", *interval(eco["profit"]), "$/L"),
        ("Space-time profit", *interval(eco["stp"]), "$/(L d)"),
        ("Space-time yield", *interval(eco["sty"]), "mg/(L d)"),
    ]
    table = pd.DataFrame(rows, columns=["Quantity", "Estimate", "95% PI lower",
                                        "95% PI upper", "Unit"])
    table[["95% PI lower", "95% PI upper"]] = table[["95% PI lower", "95% PI upper"]].fillna("-")
    return table, t_best


def observed_table(series, selected, time_h):
    """For each measured batch: its best harvest time and economics."""
    rows = []
    for s, (_, row) in zip(series, selected.iterrows()):
        t = time_h[:len(s)]
        eco = economics(row[c.COL_GLC], row[c.COL_GLN], row[c.COL_VCD],
                        s[None, :, c.IDX_MAB], t)
        i = int(eco["stp"][0].argmax())
        rows.append({
            "Batch ID": int(row[c.COL_BATCH]),
            c.COL_GLC: row[c.COL_GLC], c.COL_GLN: row[c.COL_GLN], c.COL_VCD: row[c.COL_VCD],
            "Best harvest [d]": t[i] / 24.0,
            "mAb [mg/L]": s[i, c.IDX_MAB],
            "Space-time profit [$/(L d)]": eco["stp"][0, i],
            "Space-time yield [mg/(L d)]": eco["sty"][0, i],
        })
    return pd.DataFrame(rows).sort_values("Space-time profit [$/(L d)]",
                                          ascending=False).reset_index(drop=True)


def sensitivity_table(ranking):
    """Effect of each initial condition on the space-time profit and yield.

    For every level of one variable, summarize all model-predicted conditions
    that share that level (the other two variables vary freely).
    """
    stp, sty = "Space-time profit [$/(L d)]", "Space-time yield [mg/(L d)]"
    parts = []
    for col in c.IC_COLS:
        g = ranking.groupby(col)
        part = pd.DataFrame({
            "Variable": col,
            "Level": g.size().index,
            "Mean STP": g[stp].mean().to_numpy(),
            "Max STP": g[stp].max().to_numpy(),
            "Mean STY": g[sty].mean().to_numpy(),
            "Max STY": g[sty].max().to_numpy(),
            "Mean best harvest [d]": g["Harvest time [h]"].mean().to_numpy() / 24.0,
        })
        parts.append(part)
    return pd.concat(parts, ignore_index=True)


def plot_training_data(batches, path_name):
    """mAb, VCD, glucose and glutamine time series of all training batches."""
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    for ax, idx in zip(axes.ravel(), [c.IDX_MAB, c.IDX_VCD, c.IDX_GLC, c.IDX_GLN]):
        for b in batches:
            ax.plot(b.time_h / 24.0, b.data[:, idx], lw=0.9, alpha=0.7)
        ax.set_ylabel(c.FEATURES[idx])
    axes[1, 0].set_xlabel("Time [d]")
    axes[1, 1].set_xlabel("Time [d]")
    fig.suptitle(f"Training data: {len(batches)} batch runs")
    save_figure(fig, path_name)


def plot_condition_effects(ranking, path_name):
    """Space-time profit and yield vs. each initial condition (mean and best)."""
    stp, sty = "Space-time profit [$/(L d)]", "Space-time yield [mg/(L d)]"
    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    for row, metric in enumerate([stp, sty]):
        for col_i, col in enumerate(c.IC_COLS):
            g = ranking.groupby(col)[metric]
            ax = axes[row, col_i]
            ax.plot(g.mean().index, g.mean().to_numpy(), "o-", label="Mean over others")
            ax.plot(g.max().index, g.max().to_numpy(), "s--", label="Best over others")
            ax.set_xlabel(col)
            ax.set_ylabel(metric)
    axes[0, 0].legend()
    save_figure(fig, path_name)


def plot_heatmaps(ranking, path_name):
    """Best predicted space-time profit for each pair of initial conditions."""
    stp = "Space-time profit [$/(L d)]"
    pairs = [(0, 1), (0, 2), (1, 2)]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, (i, j) in zip(axes, pairs):
        xi, xj = c.IC_COLS[i], c.IC_COLS[j]
        piv = ranking.pivot_table(index=xj, columns=xi, values=stp, aggfunc="max")
        im = ax.imshow(piv.to_numpy(), origin="lower", aspect="auto", cmap="viridis")
        ax.set_xticks(range(len(piv.columns)))
        ax.set_xticklabels([f"{v:.3g}" for v in piv.columns], rotation=45)
        ax.set_yticks(range(len(piv.index)))
        ax.set_yticklabels([f"{v:.3g}" for v in piv.index])
        ax.set_xlabel(xi)
        ax.set_ylabel(xj)
        fig.colorbar(im, ax=ax, label="Max STP [$/(L d)]")
    save_figure(fig, path_name)
