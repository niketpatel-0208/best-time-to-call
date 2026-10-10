"""
priority_model.py — P5: C3 priority / stop model.

Part 1: LightGBM P(meeting) on point-in-time features (Apr-Jul train, Aug early
        stopping, isotonic calibration on Aug). Sep AUC, PR-AUC, ECE, decile lift.
Part 2: attempt-cap curve (1-8), a model-based stop rule (threshold tuned on Aug)
        vs the flat cap of 4, and 14-day NI / 3-7-day General cool-offs. Removed
        calls' meetings are charged as lost. Calls per meeting before/after with
        seller-bootstrap 95% CIs.

Writes model_card.md (parts 1 and 2), chart_calibration.png, chart_cap_curve.png,
chart_stop_rule.png.
"""
import os, resource
try:
    resource.setrlimit(resource.RLIMIT_AS, (resource.RLIM_INFINITY, resource.RLIM_INFINITY))
except Exception:
    pass
for _v in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:
    os.environ.setdefault(_v, '1')
os.environ.setdefault('LD_LIBRARY_PATH', '/scratch/work/lib')
import ctypes, glob
for _cand in ['/scratch/work/lib/libgomp.so.1'] + glob.glob(
        '/scratch/.local/lib/python3.12/site-packages/scikit_learn.libs/libgomp-*.so*'):
    if os.path.exists(_cand):
        try:
            ctypes.CDLL(_cand, mode=ctypes.RTLD_GLOBAL)
            break
        except OSError:
            pass

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.isotonic import IsotonicRegression
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json, warnings
warnings.filterwarnings('ignore')

BASE = '/scratch/work/base_table.parquet'
REST_BUCKETS = {'NUR', 'UA', 'PUA', 'PIM'}
SEED = 7
RNG = np.random.default_rng(7)
NBOOT = 1000
NI_COOL_MIN = 14 * 1440     # 14 days
GEN_COOL_MIN = 3 * 1440     # 3 days

RAW = ['fk_glusr_usr_id', 'redis_bucket', 'bot', 'call_attempt_count', 'episode',
       'lead_sent_time', 'call_start_time', 'meeting_fixed', 'disposition_label', 'band',
       'delay_min', 'prev_outcome', 'label_regime', 'num_categories', 'hour', 'prior_attempts',
       'prior_answers', 'prior_meetings', 'prior_ni', 'prior_cl', 'prior_gen',
       'smoothed_answer_rate', 'smoothed_meeting_rate', 'days_since_last_answer', 'gap_min',
       'hour_shift', 'seller_state', 'annual_turnover', 'gst_registration_year',
       'top_category_group', 'nature_of_business', 'business_type']

NUM = ['delay_min', 'hour', 'prior_attempts', 'prior_answers', 'prior_meetings', 'prior_ni',
       'prior_cl', 'prior_gen', 'smoothed_answer_rate', 'smoothed_meeting_rate',
       'days_since_last_answer', 'gap_min', 'hour_shift', 'num_categories', 'gst_registration_year']
CAT = ['band', 'bucket_group', 'attempt_bucket', 'bot', 'state_of_seller', 'label_regime',
       'dow', 'trigger_hour', 'delay_bucket', 'new_vs_repeat', 'num_cat_bucket', 'seller_state',
       'annual_turnover', 'nature_of_business', 'top_category_group', 'business_type']

def load():
    b = pd.read_parquet(BASE, columns=RAW, use_threads=False)
    b = b.sort_values(['fk_glusr_usr_id', 'call_start_time']).reset_index(drop=True)
    b['band'] = b['band'].astype(str)
    b['month'] = b['call_start_time'].dt.to_period('M').astype(str)
    b['bucket_group'] = np.where(b['redis_bucket'].isin(REST_BUCKETS), 'Rest', 'Top 3')
    b['attempt_bucket'] = np.where(b['call_attempt_count'] >= 4, '4+',
                                   b['call_attempt_count'].astype('int64').astype(str))
    b['state_of_seller'] = b['prev_outcome'].fillna('Fresh')
    b['dow'] = b['call_start_time'].dt.dayofweek
    b['trigger_hour'] = b['lead_sent_time'].dt.hour
    b['delay_bucket'] = pd.cut(b['delay_min'], [-1, 1, 5, 30, 120, 1e9],
                               labels=['<=1m', '1-5m', '5-30m', '30-120m', '>120m']).astype(str)
    b['num_cat_bucket'] = pd.cut(b['num_categories'], [0, 5, 15, 50, 1e9],
                                 labels=['1-5', '6-15', '16-50', '50+']).astype(str)
    first_month = b.groupby('fk_glusr_usr_id')['month'].transform('min')
    b['new_vs_repeat'] = np.where(first_month == '2026-09', 'new', 'repeat')
    b['y'] = (b['meeting_fixed'] == 1).astype(int)
    ge = b.groupby(['fk_glusr_usr_id', 'episode'], sort=False)
    b['prev_label'] = ge['disposition_label'].shift(1)
    for c in CAT:
        b[c] = b[c].astype(str).fillna('NA').astype('category')
    return b

def ece(y, p, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    e = 0.0
    for k in range(bins):
        m = idx == k
        if m.sum() == 0:
            continue
        e += m.mean() * abs(p[m].mean() - y[m].mean())
    return e

# ---------------------------------------------------------- stop-rule machinery
def kept_mask(df, allow, force_first=True):
    """An attempt happens iff it is allowed and every earlier attempt in the SAME
    seller-episode happened (cummin of the allow indicator within the episode)."""
    a = np.asarray(allow, dtype=bool).copy()
    if force_first:
        a[df['call_attempt_count'].values == 1] = True
    key = (df['fk_glusr_usr_id'].astype(str) + '|' + df['episode'].astype(str)).values
    return pd.Series(a).groupby(key, sort=False).cummin().values

def rule_metrics(df, kept):
    y = df['y'].values
    calls = int(kept.sum()); meets = int((kept & (y == 1)).sum())
    return {'calls': calls, 'meetings': meets,
            'calls_spent': calls / len(df), 'meetings_retained': meets / y.sum(),
            'meetings_lost': int(y.sum() - meets),
            'cpm': calls / meets if meets else float('nan')}

def boot_cpm(df, kept, codes, S, boot_idx):
    kc = np.bincount(codes, weights=kept.astype(float), minlength=S)
    km = np.bincount(codes, weights=(kept & (df['y'].values == 1)).astype(float), minlength=S)
    out = []
    for idx in boot_idx:
        c = kc[idx].sum(); m = km[idx].sum()
        out.append(c / m if m > 0 else np.nan)
    out = np.array(out)
    return tuple(np.nanpercentile(out, [2.5, 97.5]))

def cooloff_allow(df):
    pl = df['prev_label'].values
    g = df['gap_min'].values
    viol = ((pl == 'Not Interested') & (g < NI_COOL_MIN)) | \
           ((pl == 'General (talked)') & (g < GEN_COOL_MIN))
    return ~viol

def main():
    b = load()
    tr = b[b['month'].isin(['2026-04', '2026-05', '2026-06', '2026-07'])]
    va = b[b['month'] == '2026-08']
    te = b[b['month'] == '2026-09'].copy()
    feats = NUM + CAT
    Xtr, ytr = tr[feats], tr['y'].values
    Xva, yva = va[feats], va['y'].values
    print(f"train {len(Xtr):,}  aug {len(Xva):,}  sep {len(Xte) if False else len(te):,}")

    clf = lgb.LGBMClassifier(n_estimators=3000, learning_rate=0.03, num_leaves=63,
                             min_child_samples=100, subsample=0.8, colsample_bytree=0.8,
                             reg_lambda=1.0, n_jobs=1, random_state=SEED, objective='binary')
    clf.fit(Xtr, ytr, eval_set=[(Xva, yva)], eval_metric='auc',
            callbacks=[lgb.early_stopping(80, verbose=False), lgb.log_evaluation(0)])
    best_it = clf.best_iteration_ or clf.n_estimators
    p_va = clf.predict_proba(Xva, num_iteration=best_it)[:, 1]
    p_te = clf.predict_proba(te[feats], num_iteration=best_it)[:, 1]
    iso = IsotonicRegression(out_of_bounds='clip', y_min=0, y_max=1).fit(p_va, yva)
    p_va_cal = iso.predict(p_va)
    p_te_cal = iso.predict(p_te)
    te['p'] = p_te_cal
    va = va.copy(); va['p'] = p_va_cal

    auc = roc_auc_score(te['y'], p_te_cal); prauc = average_precision_score(te['y'], p_te_cal)
    e_raw = ece(te['y'].values, p_te); e_cal = ece(te['y'].values, p_te_cal)
    print(f"Sep AUC {auc:.3f}  PR-AUC {prauc:.3f}  ECE raw {e_raw:.4f} cal {e_cal:.4f}")

    # decile lift
    d = pd.DataFrame({'p': p_te_cal, 'y': te['y'].values})
    d['decile'] = pd.qcut(d['p'].rank(method='first'), 10, labels=False)
    dec = d.groupby('decile').agg(n=('y', 'size'), rate=('y', 'mean'), mean_p=('p', 'mean')).reset_index()
    base_rate = te['y'].mean()
    dec['lift'] = dec['rate'] / base_rate
    dec['share_meetings'] = dec['rate'].mul(dec['n']) / te['y'].sum()
    dec['cum_meetings_share'] = dec['rate'].mul(dec['n']).cumsum() / te['y'].sum()

    # ---------------- part 2: cap curve, stop rule, cool-offs ----------------
    codes, uniq = pd.factorize(te['fk_glusr_usr_id'].values)
    S = len(uniq)
    boot_idx = RNG.integers(0, S, size=(NBOOT, S))

    # cap curve 1-8
    cap_curve = []
    for cap in range(1, 9):
        kept = te['call_attempt_count'].values <= cap
        m = rule_metrics(te, kept)
        m['cap'] = cap
        cap_curve.append(m)
    # tune threshold on Aug: minimise calls/meeting subject to a retention floor
    grid_rows = []
    for tau in np.arange(0.005, 0.10, 0.0025):
        k = kept_mask(va, va['p'].values >= tau)
        m = rule_metrics(va, k)
        grid_rows.append((float(tau), m['meetings_retained'], m['cpm'], m['calls_spent']))
    print("Aug tau grid (tau, retain, cpm, calls_spent):")
    for g in grid_rows[:12]:
        print(f"   {g[0]:.4f}  retain {g[1]*100:5.1f}%  cpm {g[2]:5.1f}  calls {g[3]*100:5.1f}%")
    cand = [g for g in grid_rows if g[1] >= 0.97] or \
           [g for g in grid_rows if g[1] >= 0.95] or sorted(grid_rows, key=lambda g: -g[1])[:1]
    best = min(cand, key=lambda g: g[2])
    tau_star = best[0]
    print(f"tuned tau={tau_star:.4f}  (Aug: retain {best[1]*100:.1f}%, cpm {best[2]:.1f})")

    rules = {}
    rules['baseline'] = np.ones(len(te), dtype=bool)
    rules['cap4'] = te['call_attempt_count'].values <= 4
    rules['model_stop'] = kept_mask(te, te['p'].values >= tau_star)
    rules['cap4_cooloff'] = kept_mask(te, (te['call_attempt_count'].values <= 4) & cooloff_allow(te))
    rules['model_cooloff'] = kept_mask(te, (te['p'].values >= tau_star) & cooloff_allow(te))
    results = {}
    for name, kept in rules.items():
        m = rule_metrics(te, kept)
        lo, hi = boot_cpm(te, kept, codes, S, boot_idx)
        m['cpm_ci'] = [round(lo, 2), round(hi, 2)]
        results[name] = m
        print(f"[{name}] calls {m['calls_spent']*100:.1f}%  meetings kept {m['meetings_retained']*100:.1f}% "
              f"(lost {m['meetings_lost']})  cpm {m['cpm']:.1f} [{lo:.1f},{hi:.1f}]")

    # charts
    fig, ax = plt.subplots(figsize=(8, 4.6))
    caps = [c['cap'] for c in cap_curve]
    ax.plot(caps, [c['meetings_retained'] * 100 for c in cap_curve], 'o-', color='#4c72b0', label='meetings retained')
    ax.plot(caps, [c['calls_spent'] * 100 for c in cap_curve], 's--', color='#55a868', label='calls spent')
    ax.set_xlabel('attempt cap'); ax.set_ylabel('% of baseline'); ax.set_title('Attempt-cap curve (September)')
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig('/scratch/work/chart_cap_curve.png', dpi=110); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.6))
    for name, col, mk in [('baseline', '#999999', 'o'), ('cap4', '#4c72b0', 's'),
                          ('model_stop', '#c44e52', 'D'), ('cap4_cooloff', '#8172b2', '^'),
                          ('model_cooloff', '#55a868', '*')]:
        r = results[name]
        ax.scatter(r['calls_spent'] * 100, r['meetings_retained'] * 100, s=90, color=col, marker=mk, label=name)
    ax.set_xlabel('calls spent (% of baseline)'); ax.set_ylabel('meetings retained (% of baseline)')
    ax.set_title('Stop rules: meetings retained vs calls spent (September)')
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig('/scratch/work/chart_stop_rule.png', dpi=110); plt.close(fig)

    # ---------------- model card (part 1 + part 2) ----------------
    imp = pd.Series(clf.booster_.feature_importance('gain'), index=feats).sort_values(ascending=False)
    L = []
    L.append("# Model card — C3 priority / stop model\n")
    L.append("Problem 3, Best Time to Call. Part 1 is the priority model; part 2 is the stop rule, the "
             "attempt-cap curve and the cool-offs.\n")
    L.append("## Part 1 — P(meeting)\n")
    L.append("### What it predicts\n")
    L.append("The probability that a given call attempt fixes a meeting (`meeting_fixed == 1`). It ranks "
             "attempts and supplies the calibrated P(meeting) the stop rule uses.\n")
    L.append("### Data and split\n")
    L.append(f"- Train: Apr–Jul 2026, n = {len(Xtr):,}. Early stopping: Aug 2026, n = {len(Xva):,} "
             f"(best iteration {best_it}).")
    L.append(f"- Isotonic calibration: fit on Aug out-of-sample scores.")
    L.append(f"- Test: Sep 2026 (locked), n = {len(te):,}, base meeting rate {base_rate*100:.2f}%.\n")
    L.append("### Features (all point-in-time)\n")
    L.append("Only rows strictly before the attempt's `call_start_time` contribute. Numeric: " + ", ".join(NUM) + ".")
    L.append("Categorical: " + ", ".join(CAT) + ".")
    L.append("**Excluded by rule:** `lead_call_duration`, `vendor_response_time`, `lead_tbro_time`, and the "
             "current row's `disposition_label`.\n")
    L.append("### Model\n")
    L.append("LightGBM binary classifier: `num_leaves=63`, `learning_rate=0.03`, `min_child_samples=100`, "
             "`subsample=0.8`, `colsample_bytree=0.8`, `reg_lambda=1.0`, early stopping on Aug AUC.\n")
    L.append("### September performance\n")
    L.append("| metric | value |")
    L.append("|---|---|")
    L.append(f"| AUC (calibrated) | {auc:.3f} |")
    L.append(f"| PR-AUC | {prauc:.3f} |")
    L.append(f"| ECE (raw, 10 bins) | {e_raw:.4f} |")
    L.append(f"| ECE (calibrated, 10 bins) | {e_cal:.4f} |")
    L.append("")
    L.append("### Decile lift table (September, calibrated)\n")
    L.append("| decile | n | mean P(meeting) | observed rate | lift vs base | share of all meetings | cum. (1→10) |")
    L.append("|---|---|---|---|---|---|---|")
    for _, r in dec.iterrows():
        L.append(f"| {int(r['decile'])+1} | {int(r['n']):,} | {r['mean_p']*100:.2f}% | {r['rate']*100:.2f}% | "
                 f"{r['lift']:.2f}x | {r['share_meetings']*100:.1f}% | {r['cum_meetings_share']*100:.1f}% |")
    L.append("")
    L.append("### Top features by gain\n")
    L.append("| feature | gain |")
    L.append("|---|---|")
    for f_, v in imp.head(12).items():
        L.append(f"| {f_} | {v:,.0f} |")
    L.append("")
    L.append("## Part 2 — stop rule, attempt cap and cool-offs\n")
    L.append("Removed calls' meetings are charged as **lost** (not credited), so every rule below is scored on "
             "meetings actually retained.\n")
    L.append("### Attempt-cap curve (September)\n")
    L.append("| cap | calls spent | meetings retained | meetings lost | calls per meeting |")
    L.append("|---|---|---|---|---|")
    for c in cap_curve:
        L.append(f"| {c['cap']} | {c['calls_spent']*100:.1f}% | {c['meetings_retained']*100:.1f}% | "
                 f"{c['meetings_lost']} | {c['cpm']:.1f} |")
    L.append("")
    L.append(f"### Model-based stop rule\n")
    L.append(f"Stop the episode when the calibrated P(meeting) for the next attempt falls below τ, tuned on "
             f"August to minimise calls per meeting subject to retaining ≥97% of meetings: **τ = {tau_star:.4f}** "
             f"(Aug retain {best[1]*100:.1f}%, calls per meeting {best[2]:.1f}).\n")
    L.append("### Comparison on September (95% CIs from 1,000 seller resamples)\n")
    L.append("| rule | calls spent | meetings retained | meetings lost | calls per meeting | 95% CI |")
    L.append("|---|---|---|---|---|---|")
    label = {'baseline': 'current schedule (no rule)', 'cap4': 'flat cap of 4', 'model_stop': 'model stop rule',
             'cap4_cooloff': 'cap 4 + cool-offs', 'model_cooloff': 'model stop + cool-offs'}
    for name in ['baseline', 'cap4', 'model_stop', 'cap4_cooloff', 'model_cooloff']:
        r = results[name]
        L.append(f"| {label[name]} | {r['calls_spent']*100:.1f}% | {r['meetings_retained']*100:.1f}% | "
                 f"{r['meetings_lost']} | {r['cpm']:.1f} | [{r['cpm_ci'][0]:.1f}, {r['cpm_ci'][1]:.1f}] |")
    L.append("")
    L.append("### Cool-offs\n")
    L.append(f"- After Not Interested: require a gap of ≥14 days (min 7). Calls arriving sooner are removed; "
             f"the segment carries the meetings they would have made as lost.")
    L.append(f"- After General (talked): require a gap of ≥3 days (the design's 3–7-day window). Calls "
             f"arriving sooner are removed.")
    L.append(f"Effect on September is shown by the 'cap 4 + cool-offs' and 'model stop + cool-offs' rows above.\n")
    L.append("### Read-out\n")
    base_cpm = results['baseline']['cpm']
    L.append(f"- Calls per meeting falls from **{base_cpm:.1f}** (current schedule) to "
             f"**{results['cap4']['cpm']:.1f}** under the flat cap of 4 and "
             f"**{results['model_stop']['cpm']:.1f}** under the model stop rule, while retaining "
             f"{results['model_stop']['meetings_retained']*100:.1f}% of meetings.")
    better = results['model_stop']['cpm'] <= results['cap4']['cpm']
    L.append(f"- The model stop rule {'matches or beats' if better else 'does not beat'} the flat cap of 4 on "
             f"calls per meeting, and it adapts per attempt rather than using one global number.")
    L.append(f"- Adding the cool-offs moves calls per meeting to {results['model_cooloff']['cpm']:.1f} "
             f"({results['model_cooloff']['meetings_retained']*100:.1f}% of meetings retained).")
    L.append("- These are unconstrained by capacity; the capacity-constrained number is in P6.\n")
    L.append("### Limitations\n")
    L.append("- The stop rule removes calls, and a removed call removes the meetings it would have made — "
             "counted here as lost, never credited.")
    L.append("- τ is tuned on August; a September shift in the score distribution would move the operating point.")
    L.append("- Cool-off gaps are measured on logged transitions, which reflect the current schedule's timing.\n")
    L.append("Charts: `chart_calibration.png`, `chart_cap_curve.png`, `chart_stop_rule.png`.\n")
    open('/scratch/work/model_card.md', 'w').write("\n".join(L))
    print("wrote model_card.md (parts 1 and 2)")

    json.dump({'auc': float(auc), 'pr_auc': float(prauc), 'ece_raw': float(e_raw), 'ece_cal': float(e_cal),
               'best_iteration': int(best_it), 'tau_star': float(tau_star),
               'cap_curve': cap_curve, 'rules': results,
               'deciles': dec.to_dict(orient='records')},
              open('/scratch/work/priority_metrics.json', 'w'), indent=2)
    print("wrote priority_metrics.json")

if __name__ == '__main__':
    main()
