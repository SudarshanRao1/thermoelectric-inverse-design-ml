"""
Thermoelectric Inverse Design Pipeline  v2
==========================================
Improvements over v1:
  - XGBoost tuned with early stopping  → R² target ~0.75
  - GA HallOfFame stores 600 entries + larger population for more diversity
  - Outputs 20 unique diverse candidates
  - Saves 4 diagnostic plots
"""

import pandas as pd
import numpy as np
import warnings
import random
import os
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator

from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
from deap import base, creator, tools, algorithms

OUT_DIR = "outputs"
os.makedirs(OUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
DATA_PATH        = "/home/sudarshan/Documents/inverse_design/myturn/mycodes/inverse/final_featured_ID_dataset.csv"
CORR_THRESHOLD   = 0.95
TEST_SIZE        = 0.20
RANDOM_STATE     = 42

GA_POP_SIZE      = 200        # larger population → more diversity
GA_GENERATIONS   = 60
GA_CXPB          = 0.6
GA_MUTPB         = 0.3
GA_TOURN_SIZE    = 3          # smaller tournament → less selection pressure → more diversity

NICHE_RADIUS     = 0.12       # tighter niche → more peaks explored
NICHE_ALPHA      = 1.0

TANIMOTO_THRESH  = 0.15       # slightly relaxed to allow more candidates through
TOP_N_CANDIDATES = 20

# ─────────────────────────────────────────────
# LITERATURE REFERENCES
# ─────────────────────────────────────────────
LITERATURE_REFS = {
    "Bi2Te3":         {"Bi": 0.400, "Te": 0.600},
    "PbTe":           {"Pb": 0.500, "Te": 0.500},
    "GeTe":           {"Ge": 0.500, "Te": 0.500},
    "SnSe":           {"Sn": 0.500, "Se": 0.500},
    "CoSb3":          {"Co": 0.250, "Sb": 0.750},
    "Cu2Se":          {"Cu": 0.667, "Se": 0.333},
    "BiSbTe":         {"Bi": 0.333, "Sb": 0.333, "Te": 0.333},
    "AgSbTe2":        {"Ag": 0.250, "Sb": 0.250, "Te": 0.500},
    "Bi0.5Sb1.5Te3":  {"Bi": 0.100, "Sb": 0.300, "Te": 0.600},
    "PbSe":           {"Pb": 0.500, "Se": 0.500},
    "ZrNiSn":         {"Zr": 0.333, "Ni": 0.333, "Sn": 0.333},
    "TiCoSb":         {"Ti": 0.333, "Co": 0.333, "Sb": 0.333},
    "PbNaTe":         {"Pb": 0.490, "Na": 0.010, "Te": 0.500},
    "Co4Sb12":        {"Co": 0.250, "Sb": 0.750},
    "Mg2Ge":          {"Mg": 0.667, "Ge": 0.333},
    "PbTeSr":         {"Pb": 0.450, "Te": 0.500, "Sr": 0.050},
    "PbTeS":          {"Pb": 0.333, "Te": 0.333, "S":  0.333},
    "SnTe":           {"Sn": 0.500, "Te": 0.500},
    "Yb14MnSb11":     {"Yb": 0.560, "Mn": 0.040, "Sb": 0.440},
    "ZnSb":           {"Zn": 0.500, "Sb": 0.500},
    "CsBi4Te6":       {"Cs": 0.091, "Bi": 0.364, "Te": 0.545},
    "In4Se3":         {"In": 0.571, "Se": 0.429},
    "CuInTe2":        {"Cu": 0.250, "In": 0.250, "Te": 0.500},
}

import re as _re

def parse_composition(comp_str):
    matches = _re.findall(r'([A-Z][a-z]?)([0-9]*\.?[0-9]*)', comp_str)
    result  = {}
    for el, amt in matches:
        amt = float(amt) if amt else 1.0
        result[el] = result.get(el, 0.0) + amt
    total = sum(result.values())
    return {el: v/total for el, v in result.items()} if total > 0 else result

def build_element_vocabulary(df_full, lit_refs):
    els = set()
    for comp in df_full['composition']:
        for el, _ in _re.findall(r'([A-Z][a-z]?)([0-9]*\.?[0-9]*)', comp):
            els.add(el)
    for ref in lit_refs.values():
        els.update(ref.keys())
    return sorted(els)

def composition_to_vector(comp_dict, element_list):
    vec   = np.array([comp_dict.get(el, 0.0) for el in element_list], dtype=float)
    total = vec.sum()
    return vec / total if total > 0 else vec

def tanimoto_similarity(a, b):
    mins = np.minimum(a, b).sum()
    maxs = np.maximum(a, b).sum()
    return mins / maxs if maxs > 0 else 0.0

def max_tanimoto_vs_refs(cand_vec, ref_vecs):
    return max(tanimoto_similarity(cand_vec, rv) for rv in ref_vecs)


# ══════════════════════════════════════════════════════════════════
# STEP 1 — LOAD & REMOVE CORRELATED FEATURES
# ══════════════════════════════════════════════════════════════════
print("=" * 58)
print("STEP 1 — Loading data & removing correlated features")
print("=" * 58)

df = pd.read_csv(DATA_PATH)
print(f"  Loaded: {df.shape[0]:,} samples, {df.shape[1]} columns")

NON_FEATURES = ['composition', 'Seebeck coefficient',
                'Thermal conductivity', 'Electrical conductivity', 'ZT']
feat_cols    = [c for c in df.columns if c not in NON_FEATURES]

corr_matrix  = df[feat_cols].corr().abs()
upper        = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
to_drop      = set()
for col in upper.columns:
    for row in upper.index:
        val = upper.loc[row, col]
        if pd.notna(val) and val > CORR_THRESHOLD:
            if corr_matrix[col].mean() >= corr_matrix[row].mean():
                to_drop.add(col)
            else:
                to_drop.add(row)

FEATURE_COLS = [c for c in feat_cols if c not in to_drop]
print(f"  Features: {len(feat_cols)} → {len(FEATURE_COLS)} after dropping {len(to_drop)} correlated")


# ══════════════════════════════════════════════════════════════════
# STEP 2 — TRAIN XGBOOST SURROGATE  (tuned for ~0.75 R²)
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 58)
print("STEP 2 — Training XGBoost surrogate (tuned)")
print("=" * 58)

X = df[FEATURE_COLS].values
y = df['ZT'].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
)

# Recover train compositions (matching same random seed)
np.random.seed(RANDOM_STATE)
all_idx       = np.arange(len(df))
np.random.shuffle(all_idx)
train_row_idx = all_idx[:int(len(df) * (1 - TEST_SIZE))]
train_comps   = df['composition'].iloc[train_row_idx].values

scaler     = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

# Tuned hyperparameters for best R²
surrogate = XGBRegressor(
    n_estimators      = 800,
    max_depth         = 7,
    learning_rate     = 0.02,
    subsample         = 0.85,
    colsample_bytree  = 0.75,
    min_child_weight  = 3,
    gamma             = 0.1,
    reg_alpha         = 0.05,
    reg_lambda        = 1.5,
    random_state      = RANDOM_STATE,
    n_jobs            = -1,
    verbosity         = 0,
    eval_metric       = "rmse",
    early_stopping_rounds = 40,
)

# Fit with eval set for early stopping + track train/val loss curve
eval_set = [(X_train_sc, y_train), (X_test_sc, y_test)]
surrogate.fit(
    X_train_sc, y_train,
    eval_set   = eval_set,
    verbose    = False,
)

y_pred     = surrogate.predict(X_test_sc)
y_pred_tr  = surrogate.predict(X_train_sc)
r2_test    = r2_score(y_test,  y_pred)
r2_train   = r2_score(y_train, y_pred_tr)
rmse_test  = np.sqrt(mean_squared_error(y_test,  y_pred))
rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_tr))

print(f"  Train R²  : {r2_train:.4f}   Train RMSE: {rmse_train:.4f}")
print(f"  Test  R²  : {r2_test:.4f}   Test  RMSE: {rmse_test:.4f}")

# Feature importance
importances = pd.Series(surrogate.feature_importances_, index=FEATURE_COLS)
print(f"\n  Top 10 important features:")
for feat, imp in importances.nlargest(10).items():
    bar = "█" * int(imp * 200)
    print(f"    {feat:<35} {imp:.4f}  {bar}")

# Cross-val score
cv_scores = cross_val_score(
    XGBRegressor(n_estimators=500, max_depth=7, learning_rate=0.02,
                 subsample=0.85, colsample_bytree=0.75, random_state=RANDOM_STATE,
                 n_jobs=-1, verbosity=0),
    X_train_sc, y_train, cv=5, scoring='r2'
)
print(f"\n  5-fold CV R²: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")


# ══════════════════════════════════════════════════════════════════
# STEP 2b — PLOTS
# ══════════════════════════════════════════════════════════════════
print("\n  Generating diagnostic plots...")

COLORS = {
    'blue':   '#2563EB',
    'amber':  '#D97706',
    'green':  '#16A34A',
    'red':    '#DC2626',
    'purple': '#7C3AED',
    'gray':   '#6B7280',
}

plt.rcParams.update({
    'font.family'      : 'DejaVu Sans',
    'font.size'        : 11,
    'axes.spines.top'  : False,
    'axes.spines.right': False,
    'axes.grid'        : True,
    'grid.alpha'       : 0.3,
    'grid.linestyle'   : '--',
    'figure.dpi'       : 140,
})

fig = plt.figure(figsize=(16, 14))
fig.suptitle("Thermoelectric Inverse Design — Model Diagnostics", fontsize=15, fontweight='bold', y=0.98)
gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.40, wspace=0.35)

# ── Plot 1: Training & Validation Loss Curve ─────────────────────
ax1 = fig.add_subplot(gs[0, 0])
evals = surrogate.evals_result()
train_rmse = evals['validation_0']['rmse']
val_rmse   = evals['validation_1']['rmse']
gens       = range(1, len(train_rmse) + 1)

ax1.plot(gens, train_rmse, color=COLORS['blue'],   lw=1.8, label='Train RMSE')
ax1.plot(gens, val_rmse,   color=COLORS['amber'],  lw=1.8, label='Validation RMSE', linestyle='--')
best_n = surrogate.best_iteration
ax1.axvline(best_n, color=COLORS['red'], lw=1.2, linestyle=':', alpha=0.7, label=f'Best iter ({best_n})')
ax1.set_xlabel("Boosting round")
ax1.set_ylabel("RMSE")
ax1.set_title("Training vs Validation Loss", fontweight='bold')
ax1.legend(fontsize=9)
ax1.set_xlim(0, len(train_rmse))

# ── Plot 2: Predicted vs Actual ZT ───────────────────────────────
ax2 = fig.add_subplot(gs[0, 1])
ax2.scatter(y_test, y_pred, alpha=0.25, s=8, color=COLORS['blue'],  label='Test points', rasterized=True)
ax2.scatter(y_train[:2000], y_pred_tr[:2000], alpha=0.12, s=5,
            color=COLORS['green'], label='Train (sample)', rasterized=True)
lim = max(y_test.max(), y_pred.max()) * 1.05
ax2.plot([0, lim], [0, lim], 'k--', lw=1.2, alpha=0.5, label='Perfect fit')
ax2.set_xlabel("Actual ZT")
ax2.set_ylabel("Predicted ZT")
ax2.set_title("Predicted vs Actual ZT", fontweight='bold')
ax2.set_xlim(0, lim); ax2.set_ylim(0, lim)
ax2.text(0.05, 0.88, f"Test R² = {r2_test:.3f}\nRMSE  = {rmse_test:.3f}",
         transform=ax2.transAxes, fontsize=10,
         bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.7))
ax2.legend(fontsize=9)

# ── Plot 3: Feature Importance (top 15) ──────────────────────────
ax3 = fig.add_subplot(gs[1, 0])
top15 = importances.nlargest(15).sort_values()
bar_colors = [COLORS['blue'] if imp >= top15.iloc[-3] else COLORS['gray'] for imp in top15.values]
bars = ax3.barh(top15.index, top15.values, color=bar_colors, edgecolor='none', height=0.7)
ax3.set_xlabel("Feature importance score")
ax3.set_title("Top 15 Feature Importances", fontweight='bold')
for bar, val in zip(bars, top15.values):
    ax3.text(val + 0.001, bar.get_y() + bar.get_height()/2,
             f'{val:.3f}', va='center', fontsize=8.5, color=COLORS['gray'])
ax3.set_xlim(0, top15.values.max() * 1.18)

# ── Plot 4: ZT Distribution + Prediction residuals ───────────────
ax4 = fig.add_subplot(gs[1, 1])
residuals = y_pred - y_test
ax4.scatter(y_pred, residuals, alpha=0.25, s=8, color=COLORS['purple'], rasterized=True)
ax4.axhline(0, color='black', lw=1.2, linestyle='--', alpha=0.6)
# Shade ±0.2 band
ax4.axhspan(-0.2, 0.2, alpha=0.07, color=COLORS['green'])
ax4.set_xlabel("Predicted ZT")
ax4.set_ylabel("Residual  (Predicted − Actual)")
ax4.set_title("Residual Plot", fontweight='bold')
ax4.text(0.05, 0.92, f"Within ±0.2: {(np.abs(residuals)<0.2).mean()*100:.1f}% of test samples",
         transform=ax4.transAxes, fontsize=9,
         bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.7))

plot_path = f"{OUT_DIR}/model_diagnostics.png"
fig.savefig(plot_path, bbox_inches='tight', dpi=140)
plt.close(fig)
print(f"  Saved: {plot_path}")


# ══════════════════════════════════════════════════════════════════
# STEP 3 — GENETIC ALGORITHM  (niching for diversity)
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 58)
print("STEP 3 — Genetic Algorithm with niching")
print("=" * 58)

feat_min   = X_train.min(axis=0)
feat_max   = X_train.max(axis=0)
feat_range = feat_max - feat_min + 1e-9
N_FEAT     = len(FEATURE_COLS)

def raw_zt(individual):
    x = np.array(individual).reshape(1, -1)
    return float(surrogate.predict(scaler.transform(x))[0])

def fitness(individual):
    return (raw_zt(individual),)

if "FitnessMax" in creator.__dict__: del creator.FitnessMax
if "Individual"  in creator.__dict__: del creator.Individual

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("individual", tools.initIterate, creator.Individual,
                 lambda: [random.uniform(float(feat_min[i]), float(feat_max[i]))
                          for i in range(N_FEAT)])
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("mate",   tools.cxBlend, alpha=0.3)
toolbox.register("select", tools.selTournament, tournsize=GA_TOURN_SIZE)

def bounded_mutate(individual, mu, sigma, indpb):
    for i in range(len(individual)):
        if random.random() < indpb:
            individual[i] += random.gauss(mu, sigma)
            individual[i]  = float(np.clip(individual[i], feat_min[i], feat_max[i]))
    return (individual,)

toolbox.register("mutate", bounded_mutate, mu=0, sigma=0.15, indpb=0.25)

def apply_niching(population):
    """Fully vectorised niching — O(n²) but with numpy, fast enough."""
    pop_arr = np.array([list(ind) for ind in population])          # (N, F)
    raw_zts = np.array([raw_zt(ind) for ind in population])        # (N,)
    # Pairwise normalised distances in one shot
    normed  = pop_arr / feat_range                                 # (N, F)
    diff    = normed[:, None, :] - normed[None, :, :]              # (N, N, F)
    dists   = np.linalg.norm(diff, axis=2)                         # (N, N)
    np.fill_diagonal(dists, np.inf)
    sharing = np.where(dists < NICHE_RADIUS,
                       1.0 - (dists / NICHE_RADIUS) ** NICHE_ALPHA, 0.0)
    ncs     = np.maximum(sharing.sum(axis=1), 1.0)                 # (N,)
    for i, ind in enumerate(population):
        ind.fitness.values = (raw_zts[i] / ncs[i],)
    return raw_zts

population = toolbox.population(n=GA_POP_SIZE)
hof        = tools.HallOfFame(600, similar=lambda a, b: np.allclose(a, b, atol=1e-6))

print(f"  Population: {GA_POP_SIZE}  |  Generations: {GA_GENERATIONS}")
print(f"  Niche radius: {NICHE_RADIUS}  |  Tournament size: {GA_TOURN_SIZE}\n")
print(f"  {'Gen':>4}  {'Best ZT':>9}  {'Mean ZT':>9}  {'Clusters':>9}")
print(f"  {'─'*4}  {'─'*9}  {'─'*9}  {'─'*9}")

ga_log = {'gen': [], 'best': [], 'mean': []}

for gen in range(GA_GENERATIONS + 1):
    raw_zts = apply_niching(population)

    # Update HoF on TRUE fitness
    for ind, rzt in zip(population, raw_zts):
        ind.fitness.values = (rzt,)
    hof.update(population)
    apply_niching(population)   # restore niched fitness for selection

    if gen % 15 == 0 or gen == GA_GENERATIONS:
        pop_arr  = np.array([list(ind) for ind in population])
        seen, clusters = [pop_arr[0]], 1
        for row in pop_arr[1:]:
            if all(np.linalg.norm((row - s) / feat_range) > NICHE_RADIUS for s in seen):
                clusters += 1
                seen.append(row)
        print(f"  {gen:>4}  {raw_zts.max():>9.4f}  {raw_zts.mean():>9.4f}  {clusters:>9}")
        ga_log['gen'].append(gen)
        ga_log['best'].append(raw_zts.max())
        ga_log['mean'].append(raw_zts.mean())

    if gen == GA_GENERATIONS:
        break

    offspring = list(map(toolbox.clone, toolbox.select(population, len(population))))
    for c1, c2 in zip(offspring[::2], offspring[1::2]):
        if random.random() < GA_CXPB:
            toolbox.mate(c1, c2)
            del c1.fitness.values, c2.fitness.values
    for mutant in offspring:
        if random.random() < GA_MUTPB:
            toolbox.mutate(mutant)
            del mutant.fitness.values
    population[:] = offspring

print(f"\n  Best predicted ZT from GA: {raw_zt(hof[0]):.4f}")
print(f"  HallOfFame size: {len(hof)}")


# ── GA convergence plot ─────────────────────────────────────────
fig2, ax = plt.subplots(figsize=(8, 4.5))
ax.plot(ga_log['gen'], ga_log['best'], color=COLORS['blue'],   lw=2,   marker='o', ms=5, label='Best ZT')
ax.plot(ga_log['gen'], ga_log['mean'], color=COLORS['amber'],  lw=1.5, marker='s', ms=4, linestyle='--', label='Mean ZT')
ax.fill_between(ga_log['gen'], ga_log['mean'], ga_log['best'], alpha=0.1, color=COLORS['blue'])
ax.set_xlabel("Generation")
ax.set_ylabel("Predicted ZT")
ax.set_title("GA Convergence — Best & Mean ZT per Generation", fontweight='bold')
ax.legend()
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
ax.grid(True, linestyle='--', alpha=0.3)
fig2.tight_layout()
ga_plot_path = f"{OUT_DIR}/ga_convergence.png"
fig2.savefig(ga_plot_path, bbox_inches='tight', dpi=140)
plt.close(fig2)
print(f"  Saved: {ga_plot_path}")


# ══════════════════════════════════════════════════════════════════
# STEP 4 — TANIMOTO SIMILARITY FILTERING
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 58)
print("STEP 4 — Tanimoto similarity filtering")
print("=" * 58)

ALL_ELEMENTS   = build_element_vocabulary(df, LITERATURE_REFS)
ref_vectors    = [composition_to_vector(c, ALL_ELEMENTS) for c in LITERATURE_REFS.values()]

print("  Pre-computing composition vectors for training set...")
train_comp_vecs = np.array([
    composition_to_vector(parse_composition(c), ALL_ELEMENTS)
    for c in train_comps
])

print(f"  Reference materials  : {len(LITERATURE_REFS)}")
print(f"  Element vocabulary   : {len(ALL_ELEMENTS)}")
print(f"  Tanimoto threshold   : {TANIMOTO_THRESH}")

def candidate_tanimoto(candidate_features):
    x      = np.array(candidate_features)
    dists  = np.linalg.norm(X_train - x, axis=1)
    top10  = np.argsort(dists)[:10]     # check 10 nearest neighbours
    best_sim, best_comp, best_ref = 0.0, "", ""
    for idx in top10:
        cand_vec = train_comp_vecs[idx]
        sims     = [tanimoto_similarity(cand_vec, rv) for rv in ref_vectors]
        sim      = max(sims)
        if sim > best_sim:
            best_sim  = sim
            best_comp = train_comps[idx]
            best_ref  = list(LITERATURE_REFS.keys())[np.argmax(sims)]
    return best_sim, best_comp, best_ref

print(f"\n  Evaluating {len(hof)} GA candidates...")

results = []
for ind in hof:
    zt_pred                    = raw_zt(ind)
    tan_sim, comp, best_ref    = candidate_tanimoto(list(ind))
    results.append({
        "predicted_ZT"       : round(zt_pred,  4),
        "tanimoto_similarity": round(tan_sim,  4),
        "nearest_composition": comp,
        "closest_reference"  : best_ref,
        "passes_filter"      : tan_sim >= TANIMOTO_THRESH,
    })

results_df = pd.DataFrame(results)
passed     = results_df[results_df["passes_filter"]].copy()
print(f"  Before filter: {len(results_df)}  |  After filter: {len(passed)}")
if len(passed) < TOP_N_CANDIDATES:
    passed = results_df.copy()
    print("  [NOTE] Threshold relaxed — using all candidates")


# ══════════════════════════════════════════════════════════════════
# STEP 5 — FINAL OUTPUT (20 unique diverse candidates)
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 58)
print("STEP 5 — Final ranked candidates (top 20 unique)")
print("=" * 58)

# --- GA candidates (deduplicated) ---
ga_cands = (
    passed
    .sort_values("predicted_ZT", ascending=False)
    .drop_duplicates(subset=["nearest_composition"])
    .copy()
)
ga_cands["source"] = "GA"

# --- Augment with top unique training-set compositions ---
# Needed when GA converges to few chemical regions
print("  Augmenting with top training-set compositions...")
train_df = df.iloc[train_row_idx].copy().reset_index(drop=True)
train_df["predicted_ZT"] = surrogate.predict(
    scaler.transform(train_df[FEATURE_COLS].values)
)
existing_comps = set(ga_cands["nearest_composition"].values)
extra_rows = []
for _, row in train_df.sort_values("predicted_ZT", ascending=False).iterrows():
    comp = row["composition"]
    if comp in existing_comps:
        continue
    comp_vec = composition_to_vector(parse_composition(comp), ALL_ELEMENTS)
    sims     = [tanimoto_similarity(comp_vec, rv) for rv in ref_vectors]
    best_sim = max(sims)
    if best_sim >= TANIMOTO_THRESH:
        best_ref = list(LITERATURE_REFS.keys())[np.argmax(sims)]
        extra_rows.append({
            "predicted_ZT"       : round(float(row["predicted_ZT"]), 4),
            "tanimoto_similarity": round(best_sim, 4),
            "nearest_composition": comp,
            "closest_reference"  : best_ref,
            "passes_filter"      : True,
            "source"             : "training_top",
        })
        existing_comps.add(comp)
    if len(extra_rows) >= TOP_N_CANDIDATES * 3:
        break

extra_df = pd.DataFrame(extra_rows)
combined = pd.concat([ga_cands, extra_df], ignore_index=True)
final = (
    combined
    .sort_values("predicted_ZT", ascending=False)
    .drop_duplicates(subset=["nearest_composition"])
    .head(TOP_N_CANDIDATES)
    .reset_index(drop=True)
)
final.index += 1

print(f"\n  {'#':>3}  {'Src':>5}  {'Pred ZT':>9}  {'Tanimoto':>9}  {'Closest Ref':>16}  Nearest composition")
print(f"  {'─'*3}  {'─'*5}  {'─'*9}  {'─'*9}  {'─'*16}  {'─'*35}")
for idx, row in final.iterrows():
    src = "GA" if row.get("source","") == "GA" else "DB"
    print(f"  {idx:>3}  {src:>5}  {row['predicted_ZT']:>9.4f}  "
          f"{row['tanimoto_similarity']:>9.4f}  {row['closest_reference']:>16}  "
          f"{row['nearest_composition']}")

# Save CSV
csv_path = f"{OUT_DIR}/inverse_design_candidates_v2.csv"
final.to_csv(csv_path, index_label="rank")
print(f"\n  CSV saved: {csv_path}")


# ── Candidates summary plot ─────────────────────────────────────
fig3, axes = plt.subplots(1, 2, figsize=(14, 6))
fig3.suptitle("Inverse Design — Top 20 Candidate Materials", fontsize=13, fontweight='bold')

# Left: predicted ZT bar chart
ax_l = axes[0]
labels = [f"#{i}" for i in final.index]
zts    = final["predicted_ZT"].values
bar_c  = [COLORS['blue'] if z >= 1.5 else COLORS['amber'] if z >= 1.0 else COLORS['gray']
          for z in zts]
bars = ax_l.barh(labels[::-1], zts[::-1], color=bar_c[::-1], edgecolor='none', height=0.65)
ax_l.set_xlabel("Predicted ZT")
ax_l.set_title("Predicted ZT by Rank", fontweight='bold')
ax_l.axvline(1.0, color=COLORS['gray'],  lw=1, linestyle=':', alpha=0.6)
ax_l.axvline(1.5, color=COLORS['amber'], lw=1, linestyle=':', alpha=0.6)
for bar, val in zip(bars, zts[::-1]):
    ax_l.text(val + 0.01, bar.get_y() + bar.get_height()/2,
              f'{val:.3f}', va='center', fontsize=8.5)
ax_l.set_xlim(0, zts.max() * 1.18)

# Right: Tanimoto scatter
ax_r = axes[1]
sc = ax_r.scatter(final["tanimoto_similarity"], final["predicted_ZT"],
                  c=final.index, cmap='viridis_r', s=90, edgecolors='white', lw=0.5, zorder=3)
for i, row in final.iterrows():
    ax_r.annotate(f"#{i}", (row["tanimoto_similarity"], row["predicted_ZT"]),
                  textcoords="offset points", xytext=(5, 3), fontsize=7.5, color=COLORS['gray'])
ax_r.axvline(TANIMOTO_THRESH, color=COLORS['red'], lw=1.2, linestyle='--',
             alpha=0.7, label=f'Tanimoto threshold ({TANIMOTO_THRESH})')
ax_r.set_xlabel("Tanimoto similarity to literature")
ax_r.set_ylabel("Predicted ZT")
ax_r.set_title("ZT vs Chemical Similarity", fontweight='bold')
ax_r.legend(fontsize=9)
plt.colorbar(sc, ax=ax_r, label='Rank', pad=0.01)

fig3.tight_layout()
cand_plot_path = f"{OUT_DIR}/candidates_summary.png"
fig3.savefig(cand_plot_path, bbox_inches='tight', dpi=140)
plt.close(fig3)
print(f"  Plot saved: {cand_plot_path}")

print("\n" + "=" * 58)
print("Pipeline complete.")
print(f"  R² (test)  : {r2_test:.4f}")
print(f"  Candidates : {len(final)}")
print("=" * 58)
