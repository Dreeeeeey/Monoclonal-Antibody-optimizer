# CONSTANTS
# Define constant values that do not change throughout your program.
# =======================================================================

import os

# Paths
# ----------------------------------------------------------------------------
PROVIDED_DIR = "datasets"
INITIAL_CONDITIONS_FILE = os.path.join(PROVIDED_DIR, "initial_conditions.csv")
SELECTED_FILE = "selected_batches.csv"  # the 30 batches we "download"
FIGURES_DIR = "figures"  # PNG figures
TABLES_DIR = "tables"  # CSV and Markdown tables

# ----------------------------------------------------------------------------
# Column names
# ----------------------------------------------------------------------------
COL_BATCH = "Batch ID"
COL_TIME = "Time [h]"
COL_GLC = "Glucose [mM]"
COL_GLN = "Glutamine [mM]"
COL_LAC = "Lactate [mM]"
COL_AMM = "Ammonia [mM]"
COL_VCD = "VCD [cells/mL]"
COL_TCD = "TCD [cells/mL]"
COL_MAB = "mAb [mg/L]"

# Initial-condition columns (initial_conditions.csv)
IC_COLS = [COL_GLC, COL_GLN, COL_VCD]

# Time series features (batch_####.csv). The ORDER matters: it defines the
# column order of every lagged-vector block.
FEATURES = [COL_GLC, COL_GLN, COL_LAC, COL_AMM, COL_VCD, COL_TCD, COL_MAB]
N_FEATURES = len(FEATURES)
IDX_GLC = FEATURES.index(COL_GLC)
IDX_GLN = FEATURES.index(COL_GLN)
IDX_VCD = FEATURES.index(COL_VCD)
IDX_TCD = FEATURES.index(COL_TCD)
IDX_MAB = FEATURES.index(COL_MAB)

# ----------------------------------------------------------------------------
# Experimental budget and economics
# ----------------------------------------------------------------------------
N_BATCHES = 30  # number of batch runs we are allowed to use

# True : skip the design and use whichever batch_####.csv files are already
#        present in PROVIDED_DIR (any number of folders, at most N_BATCHES).
USE_DOWNLOADED_FILES = True
DOWNTIME_D = 1.0  # [d] reactor downtime between batches

COST_GLC = 10.0  # [$/L per mM]
COST_GLN = 20.0  # [$/L per mM]
COST_VCD = 0.0005  # [$/L per (cells/mL)]
VALUE_MAB = 1000.0  # [$/L per (mg/L)]

# ----------------------------------------------------------------------------
# Modelling settings
# ----------------------------------------------------------------------------
SEED = 42
N_DESIGNS = 3000  # random candidate designs for the batch selection

# Candidate models for LOBOCV: polynomial degree 1 is just a VAR model.
POLY_DEGREES = [1, 2]
KNN_NEIGHBOURS = [2, 3, 4, 5]
LAGS = [1, 2, 3]

N_SIMS_CV = 20  # simulations per test batch during model selection
N_SIMS_SEARCH = 30  # simulations per candidate during the optimization
N_SIMS_FINAL = 1000  # simulations for the final optimum (95 % PI)
SEARCH_CHUNK = 50  # candidate conditions simulated at once (memory)
N_TOP = 5  # number of runner-up conditions to report
