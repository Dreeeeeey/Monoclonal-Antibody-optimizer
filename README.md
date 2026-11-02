# Monoclonal antibody production optimizer

This project uses time series forecasting to optimize monoclonal antibody (mAb) production in a 1L batch bioreactor. From 30 measured batch runs, this project learns how nutrients, metabolites and cell densities evolve, simulates unseen batches, and searches for the initial glucose, glutamine and viable cell density, plus the harvest time, that maximize space-time profit while avoiding expensive trial-and-error experiments.

## Highlights
- Built a multivariate time series pipeline that forecasts seven interacting process variables from initial conditions alone.
- Evaluated 18 vector autoregression variants (linear, polynomial and k-nearest neighbours) with leave-one-batch-out cross-validation.
- Quantified forecast uncertainty with 1000 white-noise simulations to report 95% prediction intervals on value, profit and yield.
- Optimized initial glucose, glutamine, cell density and harvest time against a cost model for space-time profit.
- Identified the limits of the recommendation by checking how well the training runs cover the initial-condition space.

## Environment
- Python: 3.14.7
- matplotlib: 3.11.2
- numpy: 2.5.3
- pandas: 3.0.6
- scikit-learn: 1.9.1

## Code Design
Running `main.py` executes a single pipeline from raw files to a recommended operating point. Each stage exists because the next one depends on its output.

### Batch selection and loading
The experimental budget is 30 batch runs out of 1029 available, so the choice of runs strongly limits what any model can learn. This stage reads the table of initial conditions, selects the 30 batches to use (by default the batch files present in the data folder; the code can also generate a space-filling design), and loads only those files into `Batch` objects that hold the initial conditions and the measured time series. The loader searches every subfolder for the matching `batch_####.csv` files and stops with a clear message if one is missing. This keeps the project honest about the budget and guarantees that every later stage sees exactly the same 30 batches.

### Preprocessing
Glucose is measured in millimolar while cell densities are of order 10⁶ cells/mL, so unscaled data would let the cell densities dominate every distance and every error. A scaler standardizes each of the seven variables independently, using one mean and one standard deviation per variable across all time points of the training batches. Scaling happens before lagged columns are built, so every lag of a variable shares identical scaling. The scaler also stores physical bounds: forecasts may not fall below zero or exceed twice the largest value seen in training. These bounds stop unstable models from diverging when their own predictions are reused as inputs.

### Lagged feature construction
A vector autoregression predicts the state of the culture at time $t$ from its state at the previous $p$ time points. This stage converts every batch into rows of inputs (the tiled lag blocks of all seven variables) and outputs (the current state of all seven variables). Batches are processed one at a time so that a row never mixes the end of one batch with the start of another. The result is a conventional feature matrix and target matrix that any scikit-learn regressor can learn from.

### Model comparison
Several model families could describe the culture dynamics, and a model that fits the training rows well can still forecast whole batches badly. This stage compares linear vector autoregression, a degree-2 polynomial model and k-nearest-neighbours regression with 2 to 5 neighbours, each with 1 to 3 lags. Every candidate is scored with leave-one-batch-out cross-validation: the model is retrained without one batch, forecasts that batch from its initial conditions only, and is judged on the error of the mAb concentration. Scaling is refitted inside each fold so that no information leaks from the held-out batch. The lowest cross-validated error selects the final architecture.

### Recursive forecasting with noise
The optimization needs forecasts for initial conditions that were never run, and it needs to know how uncertain those forecasts are. This stage starts from an initial state (lactate, ammonia and mAb are zero and total cell density equals viable cell density), predicts one step ahead, feeds the prediction back in as a lag, and repeats until the end of the batch. At each step, white noise is added to each variable by resampling that variable's own training residuals, which reproduces the scatter seen in the data. All conditions and simulations are advanced together in vectorized arrays, so a full search needs only one model call per time step.

### Economic evaluation and optimization
The goal is profit per reactor volume and time, not mAb concentration alone. For every candidate combination of initial glucose, glutamine and viable cell density, this stage simulates the batch, takes the median mAb trajectory, and converts it into cost, value, profit, space-time profit and space-time yield at every possible harvest time, including one day of reactor downtime per cycle. The search is limited to values of each variable that lie inside the range covered by the training batches, so no single variable is extrapolated. The combination and harvest time with the highest space-time profit become the recommendation.

### Uncertainty and reporting
A single best number is not enough to act on. The recommended condition is re-simulated 1000 times, and the 2.5th and 97.5th percentiles of each quantity form its 95% prediction interval. The program then writes figures to `figures/` (training data, cross-validation forecasts, the effect of each initial condition, profit heatmaps and the recommended forecast) and CSV tables to `tables/` (selected batches, model comparison, measured batches, effects of initial conditions, best conditions and the final recommendation). These files are the evidence used in the analysis below.

## Analysis
 
### Choice of the 30 batch runs
The 30 batches are the files available in the data folder: batches 436–464 (except 439) plus batches 780 and 831 (Table 1). They were not chosen by a space-filling design, and their coverage shows it. Batch IDs follow the full factorial grid in order, so consecutive files sweep the initial VCD at a fixed glucose and glutamine setting. As a result, 21 of the 30 runs share 20 mM glucose and 1 mM glutamine and cover all seven VCD levels (0.2 to 1.4 × 10⁶ cells/mL) in triplicate. Five runs share 15 mM glucose, 7 mM glutamine and 1.2–1.4 × 10⁶ cells/mL, two have 2 mM glutamine, and batches 780 and 831 add single runs at 3 mM and 5 mM glutamine, both at 30 mM glucose.
 
This choice has one clear advantage: triplicates at every VCD level give a clean estimate of batch-to-batch noise and a well-resolved effect of seeding density, which Figure 1 shows is large (mAb curves, top left). The weaknesses are serious, though. Glucose and glutamine are barely varied, and the 15 mM runs are all also high in glutamine and VCD, so their effects cannot be separated. There are no runs at 25 mM glucose, none at 4 or 6 mM glutamine, and none that combine a low VCD with a high glutamine level. Table 3 counts the runs behind each level. With a free choice I would use the space-filling design that the code supports (random Latin-hypercube-style designs scored by their maximin distance), which spreads 30 runs evenly across all three variables.
 
**Table 1. The 30 batches used for training**
 
| Batch ID | Glucose [mM] | Glutamine [mM] | VCD [cells/mL] |
|---|---|---|---|
| 436 | 15 | 7 | 1,200,000 |
| 437 | 15 | 7 | 1,200,000 |
| 438 | 15 | 7 | 1,200,000 |
| 440 | 15 | 7 | 1,400,000 |
| 441 | 15 | 7 | 1,400,000 |
| 442 | 20 | 1 | 200,000 |
| 443 | 20 | 1 | 200,000 |
| 444 | 20 | 1 | 200,000 |
| 445 | 20 | 1 | 400,000 |
| 446 | 20 | 1 | 400,000 |
| 447 | 20 | 1 | 400,000 |
| 448 | 20 | 1 | 600,000 |
| 449 | 20 | 1 | 600,000 |
| 450 | 20 | 1 | 600,000 |
| 451 | 20 | 1 | 800,000 |
| 452 | 20 | 1 | 800,000 |
| 453 | 20 | 1 | 800,000 |
| 454 | 20 | 1 | 1,000,000 |
| 455 | 20 | 1 | 1,000,000 |
| 456 | 20 | 1 | 1,000,000 |
| 457 | 20 | 1 | 1,200,000 |
| 458 | 20 | 1 | 1,200,000 |
| 459 | 20 | 1 | 1,200,000 |
| 460 | 20 | 1 | 1,400,000 |
| 461 | 20 | 1 | 1,400,000 |
| 462 | 20 | 1 | 1,400,000 |
| 463 | 20 | 2 | 200,000 |
| 464 | 20 | 2 | 200,000 |
| 780 | 30 | 3 | 200,000 |
| 831 | 30 | 5 | 800,000 |
 
![Figure 1: mAb, VCD, glucose and glutamine time series of the 30 training batches](figures/training_data.png)
 
*Figure 1. Training data. The mAb curves (top left) plateau between about 0.75 and 1.4 mg/L, VCD (top right) fans out from the seeding densities, and the five 15 mM glucose runs are exhausted of glucose after about a week (bottom left).*
 
### Model architecture and hyperparameters
The chosen model is k-nearest-neighbours NVAR with $k = 2$ and $p = 2$ lags (`kNN(k=2) NVAR(2)`). A vector (multivariate) structure is essential because growth depends on nutrients, nutrient consumption depends on cell density, and lactate and ammonia inhibit growth. Non-linearity matters because cultures show sigmoidal growth that stops abruptly when a nutrient runs out, a shape linear models cannot represent. The number of lags and the model hyperparameters were not set by hand. Eighteen candidates were scored by leave-one-batch-out cross-validation on whole-batch forecasts, not one-step predictions, because the application requires forecasting an entire run from its initial conditions. The selected configuration had the lowest mAb error, 0.0716 ± 0.0589 mg/L (Table 2).
 
The lag order and the number of neighbours control a bias-variance trade-off. More lags give the model a longer memory, but they multiply the inputs while the number of independent batches stays at 30, so extra lags tend to add noise. In k-nearest-neighbours regression, a small $k$ follows the data closely but is sensitive to noise, while a large $k$ blurs the sharp transition at the end of exponential growth. The search range was kept small deliberately, since a wide search on 30 batches would itself overfit the cross-validation score. Uniform neighbour weights were used because there is too little data to justify distance weighting. The margin between the best candidates is small: the five best models lie between 0.072 and 0.077 mg/L, far less than the batch-to-batch standard deviation of about 0.06 mg/L. The exact values $k = 2$ and $p = 2$ should therefore not be over-interpreted, and $k = 2$ is the lowest value tested, so a smaller neighbourhood might do equally well.
 
### Comparison with alternatives
**Table 2. Leave-one-batch-out forecast error of all 18 candidate models (sorted by mAb RMSE)**
 
| Model | mAb RMSE mean [mg/L] | mAb RMSE SD [mg/L] | All-feature RMSE (std.) |
|---|---|---|---|
| kNN(k=2) NVAR(2) | 0.0716 | 0.0589 | 0.233 |
| kNN(k=2) NVAR(1) | 0.0744 | 0.0635 | 0.253 |
| kNN(k=3) NVAR(2) | 0.0762 | 0.0615 | 0.258 |
| kNN(k=2) NVAR(3) | 0.0766 | 0.0567 | 0.248 |
| kNN(k=4) NVAR(2) | 0.0772 | 0.0604 | 0.262 |
| kNN(k=3) NVAR(1) | 0.0776 | 0.0580 | 0.258 |
| kNN(k=4) NVAR(1) | 0.0814 | 0.0604 | 0.277 |
| kNN(k=3) NVAR(3) | 0.0824 | 0.0581 | 0.270 |
| kNN(k=5) NVAR(1) | 0.0846 | 0.0595 | 0.281 |
| kNN(k=5) NVAR(2) | 0.0851 | 0.0634 | 0.279 |
| kNN(k=4) NVAR(3) | 0.0905 | 0.0629 | 0.287 |
| kNN(k=5) NVAR(3) | 0.1012 | 0.0720 | 0.303 |
| VAR(3) | 0.1266 | 0.0447 | 0.357 |
| VAR(2) | 0.1477 | 0.0527 | 0.419 |
| Poly(deg=2) NVAR(2) | 0.1634 | 0.2653 | 0.785 |
| Poly(deg=2) NVAR(1) | 0.1663 | 0.2011 | 0.933 |
| Poly(deg=2) NVAR(3) | 0.1728 | 0.2746 | 0.881 |
| VAR(1) | 0.1808 | 0.0591 | 0.527 |
 
Table 2 shows a clear separation between model families: all twelve kNN models (0.072–0.101 mg/L) beat every linear and polynomial model (0.127–0.181 mg/L). The best linear alternative, `VAR(3)`, reached 0.1266 mg/L, compared with 0.0716 mg/L for the selected model. Non-parametric kNN models performed best because their predictions are averages of observed values, so they can never leave the range of the training data and cannot run away when fed their own outputs. Polynomial models were the worst and also the most variable (SD up to 0.27 mg/L), because quadratic terms extrapolate wildly when predictions are fed back over a long forecast horizon.
 
![Figure 2: leave-one-batch-out mAb forecasts of the selected model](figures/cv_forecasts.png)
 
*Figure 2. Leave-one-batch-out mAb forecasts of the selected model for the first twelve batches (orange: measured, black: median forecast, blue: 95% prediction interval).*
 
In Figure 2, the model follows the fast batches in panels 1–5, which reach about 1.05 mg/L within roughly four days, and the slow batches seeded at 2 × 10⁵ cells/mL in panels 6–8, which take more than ten days to approach 0.8 mg/L. The 95% bands contain most of the measured points.
 
### Advantages and limitations
The main advantage is that the model needs no mechanistic growth model, forecasts all seven variables consistently, and produces prediction intervals through simulation.
 
The limitations are real. With 30 batches, each model is trained on few independent trajectories, so the cross-validation error is noisy. It is also optimistic: the replicates of a held-out batch stay in the training set, so Figure 2 shows the forecast of conditions the model has effectively already seen, and the error for a new combination will be larger. The prediction intervals reflect residual noise only; they do not include uncertainty in the model structure or in regions without data, so they are too narrow there. A kNN model cannot predict values beyond those seen in training, and the search covers only the discrete levels inside the data range, so a better optimum may exist between levels or outside it. Finally, because mAb is worth little per litre in this process (Figure 1), a small error in predicted mAb has a large relative effect on profit, so the ranking of conditions is more reliable than the absolute dollar values.
 
### Effect of initial conditions on space-time profit and yield
Space-time profit is the profit per litre divided by the cycle time (harvest time plus one day of downtime), and space-time yield is the same ratio for mAb concentration. The mAb concentrations in this process are small, about 0.75–1.4 mg/L (Figure 1), so culture value is only about \$750–1,400 per litre. The medium and cells cost between \$270 and \$1,140 per litre across the search range, so cost is comparable in size to value, and cheap starting conditions matter as much as high mAb. Table 3 and Figure 3 summarize the model's predictions; the "Training runs" column shows how much data supports each level.
 
**Table 3. Predicted space-time profit (STP, \$/(L d)) and yield (STY, mg/(L d)) for each level of each initial condition, across all combinations of the other two variables**
 
| Variable | Level | Training runs | Mean STP | Max STP | Mean STY | Max STY | Mean best harvest [d] |
|---|---|---|---|---|---|---|---|
| Glucose [mM] | 15 | 5 | 64.0 | 202.5 | 0.190 | 0.488 | 5.2 |
| Glucose [mM] | 20 | 23 | 59.1 | 173.4 | 0.187 | 0.351 | 5.4 |
| Glucose [mM] | 25 | 0 | 55.0 | 152.8 | 0.181 | 0.324 | 5.7 |
| Glucose [mM] | 30 | 2 | 45.9 | 137.7 | 0.162 | 0.295 | 7.2 |
| Glutamine [mM] | 1 | 21 | 50.7 | 85.1 | 0.153 | 0.242 | 5.6 |
| Glutamine [mM] | 2 | 2 | 39.2 | 70.3 | 0.128 | 0.219 | 7.1 |
| Glutamine [mM] | 3 | 1 | 27.5 | 57.0 | 0.101 | 0.205 | 9.8 |
| Glutamine [mM] | 4 | 0 | 47.0 | 105.5 | 0.165 | 0.256 | 5.9 |
| Glutamine [mM] | 5 | 1 | 84.6 | 182.7 | 0.246 | 0.335 | 3.6 |
| Glutamine [mM] | 6 | 0 | 76.3 | 202.5 | 0.239 | 0.388 | 3.8 |
| Glutamine [mM] | 7 | 5 | 66.8 | 195.0 | 0.229 | 0.488 | 5.2 |
| VCD [cells/mL] | 200,000 | 6 | 96.9 | 202.5 | 0.198 | 0.488 | 4.4 |
| VCD [cells/mL] | 400,000 | 3 | 75.4 | 165.9 | 0.187 | 0.388 | 4.7 |
| VCD [cells/mL] | 600,000 | 3 | 64.8 | 119.3 | 0.184 | 0.327 | 5.1 |
| VCD [cells/mL] | 800,000 | 4 | 55.0 | 93.2 | 0.185 | 0.330 | 5.7 |
| VCD [cells/mL] | 1,000,000 | 3 | 40.0 | 71.5 | 0.178 | 0.252 | 7.2 |
| VCD [cells/mL] | 1,200,000 | 6 | 35.1 | 62.9 | 0.176 | 0.254 | 6.2 |
| VCD [cells/mL] | 1,400,000 | 5 | 24.8 | 50.2 | 0.154 | 0.256 | 7.7 |
 
![Figure 3: effect of each initial condition on space-time profit and yield](figures/initial_condition_effects.png)
 
*Figure 3. Predicted space-time profit (top row) and yield (bottom row) against each initial condition. Blue: mean over all combinations of the other two variables; orange: best combination.*
 
**Glucose.** Predicted profit falls steadily as glucose increases: the best achievable value drops from 202.5 \$/(L d) at 15 mM to 137.7 \$/(L d) at 30 mM, and the mean from 64.0 to 45.9 (Figure 3, left column). Every extra 10 mM costs \$100 per litre, and it does not appear to buy early mAb: the five 15 mM runs are out of glucose after about a week (Figure 1, bottom left), yet their mAb had plateaued by day four (Figure 2, panels 1–5). This trend cannot be separated from the other variables, however, because the 15 mM runs are also the high-glutamine, high-VCD runs, and 25 mM has no run at all.
 
**Glutamine.** The best-case profit falls from 85.1 at 1 mM to 57.0 \$/(L d) at 3 mM and then rises to 202.5 at 6 mM (Figure 3, middle column). I do not interpret this as a biological effect. Table 3 shows that 3 mM and 5 mM are each represented by a single run, and 4 mM and 6 mM by none. The model fills these levels with the nearest runs, mainly the five fast-growing 7 mM batches, so the apparent glutamine optimum mostly reflects sparse data.
 
**Seeding density.** Mean profit decreases from 96.9 at 2 × 10⁵ to 24.8 \$/(L d) at 1.4 × 10⁶ cells/mL (Figure 3, right column). Part of this is arithmetic: raising the VCD by 1.2 × 10⁶ cells/mL adds \$600 per litre of cost, while the mAb plateau in Figure 1 rises by only about 0.6 mg/L, worth roughly \$600. The measured runs disagree with the model's trend, though. Their best space-time profit occurs at 6 × 10⁵–10⁶ cells/mL (Table 6), not at the lowest density.
 
**Yield and harvest time.** Space-time yield follows the same ordering (Figure 3, bottom row), but its highest value, 0.488 mg/(L d), is reached with a harvest after only 8 h, because yield ignores cost and rewards the earliest high rate. The mean best harvest time in Table 3 rises from 4.4 d at 2 × 10⁵ to 7.7 d at 1.4 × 10⁶ cells/mL, probably because an expensive, dense culture must accumulate more mAb before its extra cost is repaid.
 
![Figure 4: heatmaps of the maximum predicted space-time profit for each pair of initial conditions](figures/profit_heatmaps.png)
 
*Figure 4. Best predicted space-time profit for each pair of initial conditions, maximized over the third variable.*
 
Figure 4 shows where the model sees the highest profit: the corner with low glucose, 5–7 mM glutamine and 2 × 10⁵ cells/mL (bright cells in all three panels). No training run lies in this corner.
 
### Recommended operating conditions
The search ranked every combination of initial conditions inside the training range. Table 4 gives the best one, with 95% prediction intervals from 1000 simulations.
 
**Table 4. Recommended operating conditions**
 
| Quantity | Estimate | 95% PI lower | 95% PI upper | Unit |
|---|---|---|---|---|
| Initial glucose | 15 | - | - | mM |
| Initial glutamine | 6 | - | - | mM |
| Initial VCD | 200,000 | - | - | cells/mL |
| Harvest time | 1.0 | - | - | d |
| Cell culture cost | 370 | - | - | \$/L |
| mAb at harvest | 0.765 | 0.695 | 0.840 | mg/L |
| Cell culture value at harvest | 765 | 695 | 840 | \$/L |
| Profit | 395 | 325 | 470 | \$/L |
| Space-time profit | 197.5 | 162.5 | 235.1 | \$/(L d) |
| Space-time yield | 0.383 | 0.348 | 0.420 | mg/(L d) |
 
In words, the model recommends starting the culture with 15 mM glucose, 6 mM glutamine and 200,000 cells/mL, and harvesting after 1 day (24 h). This is predicted to cost \$370 per litre and return a culture value of \$765 per litre at harvest (95% PI: \$695 to \$840), for a space-time profit of 197.5 \$/(L d) (95% PI: 162.5 to 235.1) and a space-time yield of 0.383 mg/(L d) (95% PI: 0.348 to 0.420). The intervals are the 2.5th and 97.5th percentiles of 1000 simulated batches; cost is known exactly, so it has none.
 
![Figure 5: forecast and space-time profit at the recommended conditions](figures/optimum_forecast.png)
 
*Figure 5. Left: median mAb forecast with 95% prediction interval at the recommended conditions. Right: space-time profit against harvest time; the dashed line marks the chosen harvest.*
 
The right panel of Figure 5 explains the harvest time. Profit is negative at time zero (only cost, no mAb), jumps to about 195 \$/(L d) after 8 h as mAb reaches 0.63 mg/L (left panel), peaks at 197.5 at 24 h, and then declines steadily to about 25 \$/(L d) at 28 days, because mAb has plateaued while the cycle time keeps growing. The curve is nearly flat between 8 and 24 h, and the prediction band there spans roughly 140 to 250 \$/(L d), so any harvest in this window is statistically indistinguishable; 24 h simply has the highest median.
 
**Table 5. The ten best conditions by median predicted space-time profit** (the full top 20 is in `tables/optimization_ranking_top20.csv`)
 
| Glucose [mM] | Glutamine [mM] | VCD [cells/mL] | Harvest [h] | mAb [mg/L] | Profit [\$/L] | STP [\$/(L d)] | STY [mg/(L d)] |
|---|---|---|---|---|---|---|---|
| 15 | 6 | 200,000 | 24 | 0.775 | 405 | 202.5 | 0.388 |
| 15 | 7 | 200,000 | 8 | 0.650 | 260 | 195.0 | 0.488 |
| 15 | 5 | 200,000 | 64 | 1.020 | 670 | 182.7 | 0.278 |
| 20 | 5 | 200,000 | 40 | 0.863 | 463 | 173.4 | 0.323 |
| 20 | 6 | 200,000 | 40 | 0.865 | 445 | 166.9 | 0.324 |
| 15 | 5 | 400,000 | 40 | 0.893 | 443 | 165.9 | 0.335 |
| 20 | 7 | 200,000 | 32 | 0.820 | 380 | 162.9 | 0.351 |
| 25 | 5 | 200,000 | 40 | 0.858 | 408 | 152.8 | 0.322 |
| 25 | 6 | 200,000 | 64 | 1.020 | 550 | 150.0 | 0.278 |
| 15 | 6 | 400,000 | 40 | 0.858 | 388 | 145.3 | 0.322 |
 
The runners-up in Table 5 lie in the same region and the top three fall within the prediction interval of the recommendation, so the profit surface is flat near its top.
 
### Reliability of the recommendation
**Table 6. The five measured batches with the highest space-time profit**
 
| Batch ID | Glucose [mM] | Glutamine [mM] | VCD [cells/mL] | Best harvest [d] | mAb [mg/L] | STP [\$/(L d)] | STY [mg/(L d)] |
|---|---|---|---|---|---|---|---|
| 452 | 20 | 1 | 800,000 | 3.0 | 0.95 | 82.5 | 0.238 |
| 451 | 20 | 1 | 800,000 | 5.0 | 1.08 | 76.7 | 0.180 |
| 453 | 20 | 1 | 800,000 | 5.0 | 1.06 | 73.3 | 0.177 |
| 456 | 20 | 1 | 1,000,000 | 4.3 | 1.11 | 73.1 | 0.208 |
| 450 | 20 | 1 | 600,000 | 5.3 | 0.98 | 72.6 | 0.155 |
 
The model predicts 197.5 \$/(L d), against 82.5 for the best measured run (batch 452, Table 6). That gap should be treated as a hypothesis, not a result. The recommended glucose and VCD are the lowest values in the training range, so the true optimum may lie beyond them. More importantly, each variable lies inside its own range but the combination does not: no training run pairs a low VCD with a high glutamine level. The predicted first day (0.63 mg/L after 8 h and 0.77 mg/L after 24 h) closely resembles the first day of the 15 mM glucose, 7 mM glutamine runs (Figure 2, panels 1–5), which were seeded at six to seven times the VCD. In contrast, the measured runs seeded at 2 × 10⁵ cells/mL accumulate mAb slowly (Figure 2, panels 6–8). The model has probably combined the low cost of the low-density culture with the fast start of a high-density one. The prediction intervals do not capture this, because they only include residual noise.
 
The recommendation is therefore a direction to test, not a validated optimum. The measured runs support a more conservative fallback: 20 mM glucose, 1 mM glutamine and 8 × 10⁵ cells/mL, harvested after about 3 days (82.5 \$/(L d)). The next step is a verification experiment at the recommended conditions, plus new runs from a space-filling design that fill the gaps in Table 3, evaluated with cross-validation that holds out all replicates of a condition.
 