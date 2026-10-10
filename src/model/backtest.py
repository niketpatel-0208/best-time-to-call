"""
backtest.py — P6a: full policy on locked September, via the shared evaluator.

Policies: (a) current, (b) global best band, (c) C1 only, (d) LightGBM argmax band,
(e) full policy unconstrained, (f) full policy capacity-constrained (current volume
+/-20% per band, assigned by marginal gain, overflow to backup bands).

Full policy = C1 timing + C2 retry + C3 model stop + deferred cool-offs.
Cool-offs defer a call (not delete it): a deferred call is added at the end of the
cool-off in the seller's recommended band, credited with the observed Apr-Aug meeting
rate for (previous outcome x gap bucket x attempt stratum), and counted as a call.
Both the "removed" and "deferred" versions are reported. Cool-offs apply only to
bot-initiated follow-ups inside an episode; a new hot-lead trigger is never suppressed.
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
for _c in ['/scratch/work/lib/libgomp.so.1'] + glob.glob('/scratch/.local/lib/python3.12/site-packages/scikit_learn.libs/libgomp-*.so*'):
    if os.path.exists(_c):
        try:
            ctypes.CDLL(_c, mode=ctypes.RTLD_GLOBAL); break
        except OSError:
            pass

import numpy as np
import pandas as pd
import json, gc, warnings
warnings.filterwarnings('ignore')
import lightgbm as lgb
from sklearn.isotonic import IsotonicRegression
import evaluator as ev
from priority_model import NUM, CAT, kept_mask, cooloff_allow, NI_COOL_MIN, GEN_COOL_MIN

BASE = '/scratch/work/base_table.parquet'
BANDS, PEAKS, BAND_START = ev.BANDS, ev.PEAKS, ev.BAND_START
SEGS = ev.SEGS
EXTRA = ['data_hotlead_disposition_dtlid', 'episode', 'disposition_label', 'gap_min', 'hour',
         'num_categories', 'prior_attempts', 'prior_answers', 'prior_meetings', 'prior_ni', 'prior_cl',
         'prior_gen', 'smoothed_answer_rate', 'smoothed_meeting_rate', 'days_since_last_answer',
         'hour_shift', 'gst_registration_year', 'seller_state', 'annual_turnover', 'top_category_group',
         'nature_of_business', 'business_type']
RETRY_BANDS = BANDS

def load_all():
    b = ev.load()                                   # band, CTX_CATS, y_meet/y_ans, fk, month...
    extra = pd.read_parquet(BASE, columns=EXTRA, use_threads=False)
    b = pd.concat([b, extra], axis=1)               # both unsorted reads -> index aligned
    bs = b.sort_values(['fk_glusr_usr_id', 'call_start_time'])
    ge = bs.groupby(['fk_glusr_usr_id', 'episode'], sort=False)
    b['prev_label'] = ge['disposition_label'].shift(1)
    b['prev_band'] = ge['band'].shift(1)
    b['seg_id'] = ev.classify_segments(b)
    b['gst_bucket'] = pd.cut(b['gst_registration_year'], [0, 2015, 2019, 2023, 2027],
                             labels=['<=2015', '2016-2019', '2020-2023', '2024+']).astype(str)
    for c in CAT:
        if c not in b.columns:
            b[c] = 'NA'
        b[c] = b[c].astype(str).fillna('NA').astype('category')
    return b

def train_priority(b):
    tr = b[b['month'].isin(['2026-04', '2026-05', '2026-06', '2026-07'])]
    va = b[b['month'] == '2026-08']
    feats = NUM + CAT
    Xtr = tr[feats]
    Xva = va[feats]
    clf = lgb.LGBMClassifier(n_estimators=3000, learning_rate=0.03, num_leaves=63,
                             min_child_samples=100, subsample=0.8, colsample_bytree=0.8,
                             reg_lambda=1.0, n_jobs=1, random_state=7, objective='binary')
    clf.fit(Xtr, tr['y_meet'].values, eval_set=[(Xva, va['y_meet'].values)], eval_metric='auc',
            callbacks=[lgb.early_stopping(80, verbose=False), lgb.log_evaluation(0)])
    it = clf.best_iteration_ or clf.n_estimators
    iso = IsotonicRegression(out_of_bounds='clip', y_min=0, y_max=1).fit(
        clf.predict_proba(Xva, num_iteration=it)[:, 1], va['y_meet'].values)
    return clf, iso, it, feats

def predict_p(clf, iso, it, feats, df):
    p = clf.predict_proba(df[feats], num_iteration=it)[:, 1]
    return iso.predict(p)

def band_deferral(df, pi):
    logged = df['band'].values
    sh = pd.Series(pi).map(BAND_START).values.astype('int64')
    cm = df['call_start_time'].values.astype('datetime64[m]').astype('int64')
    ds = (cm // 1440) * 1440
    tgt = ds + sh * 60
    tgt = np.where(tgt > cm, tgt, tgt + 1440)
    d = (tgt - cm).astype(float)
    return np.where(np.asarray(pi) == np.asarray(logged), 0.0, d)

def credit_lookup(b):
    """Observed Apr-Aug meeting rate for (prev_label x gap_bucket x attempt_bucket)."""
    tr = b[(b['month'] != '2026-09') & b['prev_label'].isin(['Not Interested', 'General (talked)'])].copy()
    tr['gb'] = pd.cut(tr['gap_min'], [-1, 1440, 4320, 10080, 20160, 43200, 1e9],
                      labels=['<=1d', '1-3d', '3-7d', '7-14d', '14-30d', '>30d']).astype(str)
    g = tr.groupby(['prev_label', 'gb', 'attempt_bucket'])['y_meet'].mean()
    return g.to_dict(), float(tr['y_meet'].mean())

def main():
    b = load_all()
    print("rows", len(b))
    clf, iso, it, feats = train_priority(b)
    b['p'] = predict_p(clf, iso, it, feats, b)
    credit, credit_default = credit_lookup(b)
    sep = b[b['month'] == '2026-09'].copy()
    del b; gc.collect()
    print("sep", len(sep), "base meet", round(sep['y_meet'].mean()*100, 2))

    # models for the evaluator
    tr = sep.iloc[0:0]  # placeholder
    btr = ev.load()
    btr = btr[btr['month'].isin(['2026-04', '2026-05', '2026-06', '2026-07'])]
    bag = ev.load(); bag = bag[bag['month'] == '2026-08']
    prop = ev.fit_propensity(btr, bag)
    out_meet = ev.fit_outcome(btr, 'y_meet')
    out_ans = ev.fit_outcome(btr, 'y_ans')
    del btr, bag; gc.collect()

    # seller windows (C1)
    sw = pd.read_csv('/scratch/work/seller_windows.csv', usecols=['seller_id', 'rec_band', 'backup_band_1', 'backup_band_2'])
    rec = dict(zip(sw['seller_id'], sw['rec_band']))
    bk1 = dict(zip(sw['seller_id'], sw['backup_band_1']))
    bk2 = dict(zip(sw['seller_id'], sw['backup_band_2']))

    # C2 retry table
    rp = json.load(open('/scratch/work/retry_policy.json'))
    retry = {}
    for row in rp['states']['not_answered']['decision_table']:
        retry[(int(row['attempt_no']), row['failed_band'])] = row['recommended']['next_band']
    GAP_MID = {'<=15m': 7.5, '15-45m': 30, '45m-2h': 82.5, '2-4h': 180, '4-8h': 360, '8-24h': 960, '1-2d': 2160, '>2d': 4320}
    retry_gap = {(int(r['attempt_no']), r['failed_band']): GAP_MID.get(r['recommended']['gap_bucket'], 240)
                 for r in rp['states']['not_answered']['decision_table']}

    fresh = sep['seg_id'].values == 0
    logged = sep['band'].values
    fk = sep['fk_glusr_usr_id'].values
    rec_band = np.array([rec.get(s, '19-20') for s in fk])
    backup1 = np.array([bk1.get(s, '17-18') for s in fk])
    backup2 = np.array([bk2.get(s, '11-12') for s in fk])
    att = sep['call_attempt_count'].astype('int64').values

    def nearest_peak(x):
        return x if x in PEAKS else ('11-12' if x == '9-10' else '19-20')

    # ---- policy bands ----
    def pi_of(kind):
        if kind == 'current':
            return logged.copy()
        if kind == 'global_best':
            pi = np.array([nearest_peak(x) for x in logged]); pi[fresh] = logged[fresh]; return pi
        if kind == 'c1':
            pi = rec_band.copy(); pi[fresh] = logged[fresh]; return pi
        if kind == 'lgb_argmax':
            # argmax over bands of the outcome model q_hat(x, band) with current delay
            best = None
            for bnd in BANDS:
                cf = sep.copy(); cf['band'] = bnd
                q = ev.outcome_pred(out_meet, cf, 'band')
                best = q if best is None else np.maximum(best, q)
                if bnd == BANDS[0]:
                    Q = q[:, None]
                else:
                    Q = np.column_stack([Q, q])
                del cf
            pi = np.array(BANDS)[np.argmax(Q, axis=1)]; pi[fresh] = logged[fresh]; return pi
        # full policy bands: C2 next_band for NA-retries, else C1 rec_band
        pi = rec_band.copy()
        is_na_retry = (att >= 2) & (sep['prev_label'].values == 'Not Answered')
        pb = sep['prev_band'].values
        for i in np.where(is_na_retry)[0]:
            key = (int(att[i]) - 1, pb[i])
            if key in retry:
                pi[i] = retry[key]
        pi[fresh] = logged[fresh]
        return pi

    results = {}
    def score(df, pi, defer, tag, kind=None):
        codes, uniq = pd.factorize(df['fk_glusr_usr_id'].values)
        S = len(uniq)
        bi = np.random.default_rng(7).integers(0, S, size=(ev.NBOOT, S))
        cur = kind == 'current'
        r = ev.score_policy(df, prop, out_meet, out_ans, kind or 'custom', None, codes, S, None, bi,
                            pi_override=None if cur else pi, defer_override=None if cur else defer)
        results[tag] = r
        return r

    # (a)-(d) on full Sep
    for kind in ['current', 'global_best', 'c1', 'lgb_argmax']:
        pi = logged.copy() if kind == 'current' else pi_of(kind)
        df = band_deferral(sep, pi)
        score(sep, pi, df, kind, kind=kind)
        r = results[kind]
        print(f"[{kind}] DR {r['val_dr']*100:.2f}% lift {r['lift_dr']:+.1%} "
              f"[{r['ci_dr'][0]:+.1%},{r['ci_dr'][1]:+.1%}] replay {r['lift_std']:+.1%} connect {r['val_dr_ans']*100:.1f}%")

    # ---- full policy: stop + cool-offs (deferrals, not deletions) ----
    tau = json.load(open('/scratch/work/priority_metrics.json'))['tau_star']
    pi_full = pi_of('full')
    defer_band = band_deferral(sep, pi_full)
    pl = sep['prev_label'].values; gm = sep['gap_min'].values
    co_ni = (pl == 'Not Interested') & (gm < NI_COOL_MIN) & (att >= 2)
    co_gen = (pl == 'General (talked)') & (gm < GEN_COOL_MIN) & (att >= 2)
    co_removed = co_ni | co_gen
    defer_co = np.where(co_ni, NI_COOL_MIN, np.where(co_gen, GEN_COOL_MIN, 0.0))
    defer_full = np.maximum(defer_band, defer_co)
    allow = sep['p'].values >= tau
    kept_stop = kept_mask(sep, allow)
    co_mask = co_removed & allow
    kept = kept_stop & ~co_removed
    n_deferred = int(co_mask.sum())

    def gb_of(x):
        return ('<=1d' if x <= 1440 else '1-3d' if x <= 4320 else '3-7d' if x <= 10080
                else '7-14d' if x <= 20160 else '14-30d' if x <= 43200 else '>30d')
    ab = sep['attempt_bucket'].values
    def_idx = np.where(co_mask)[0]
    def_cred = np.array([credit.get((pl[k], gb_of(NI_COOL_MIN if co_ni[k] else GEN_COOL_MIN), ab[k]),
                                    credit_default) for k in def_idx])
    deferred_meet = float(def_cred.sum())

    # ---- combined expected-meetings numbers (DR per call + deferred credit) ----
    codes, uniq = pd.factorize(sep['fk_glusr_usr_id'].values); S = len(uniq)
    cur_y = np.bincount(codes, weights=sep['y_meet'].values, minlength=S)
    BIDX = np.random.default_rng(7).integers(0, S, size=(ev.NBOOT, S))
    base_calls = len(sep); base_meet = int(sep['y_meet'].sum())     # observed 7,383

    def dr_rows(df, pi, defer):
        cf = df.copy(); cf['band'] = pi
        cf['delay_bucket'] = pd.cut(df['delay_min'].values + defer, ev.DELAY_BINS, labels=ev.DELAY_LAB)
        q_pi = ev.outcome_pred(out_meet, cf, 'band')
        q_log = ev.outcome_pred(out_meet, df, 'band')
        p_log = ev.prop_logged(prop, df)
        match = (df['band'].values == np.asarray(pi)).astype(float)
        w = np.clip(match / p_log, 0, np.percentile(match / p_log, 99))
        return q_pi + w * (df['y_meet'].values - q_log)

    def combined(pol_dr, keep_mask, d_idx=None, d_cred=None):
        kept_calls = np.bincount(codes[keep_mask], minlength=S)
        pd_s = np.bincount(codes[keep_mask], weights=pol_dr[keep_mask], minlength=S)
        dm_s = np.zeros(S); dc_s = np.zeros(S)
        if d_idx is not None and len(d_idx):
            dm_s = np.bincount(codes[d_idx], weights=d_cred, minlength=S)
            dc_s = np.bincount(codes[d_idx], minlength=S)
        exp = pd_s.sum() + dm_s.sum(); calls = kept_calls.sum() + dc_s.sum(); cur = cur_y.sum()
        liftb, cpmb, mpb = [], [], []
        for idx in BIDX:
            e = pd_s[idx].sum() + dm_s[idx].sum(); c = kept_calls[idx].sum() + dc_s[idx].sum()
            cu = cur_y[idx].sum()
            liftb.append(e / cu - 1); cpmb.append(c / e); mpb.append(e / c * 1000)
        ci = lambda a: [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]
        return {'expected': float(exp), 'calls': int(calls), 'lift': float(exp / cur - 1),
                'cpm': float(calls / exp), 'mp1k': float(exp / calls * 1000),
                'lift_ci': ci(liftb), 'cpm_ci': ci(cpmb), 'mp1k_ci': ci(mpb)}

    comb = {}
    comb['current'] = combined(sep['y_meet'].values.astype(float), np.ones(len(sep), bool))
    for kind in ['global_best', 'c1', 'lgb_argmax']:
        pi = pi_of(kind); comb[kind] = combined(dr_rows(sep, pi, band_deferral(sep, pi)),
                                                np.ones(len(sep), bool))
    dr_full = dr_rows(sep, pi_full, defer_full)
    comb['full_uncon'] = combined(dr_full, kept, def_idx, def_cred)
    comb['full_removed'] = combined(dr_full, kept)
    # capacity
    cur_vol = pd.Series(logged).value_counts().reindex(BANDS).fillna(0).values
    cap = np.floor(cur_vol * 1.2); occ = np.zeros(len(BANDS))
    for i in np.where(fresh)[0]:
        occ[BANDS.index(logged[i])] += 1
    idx = np.where(kept_stop & ~fresh)[0]
    order = idx[np.argsort(-sep['p'].values[idx])]
    assign = pi_full.copy()
    for i in order:
        want = pi_full[i]
        if occ[BANDS.index(want)] >= cap[BANDS.index(want)]:
            if occ[BANDS.index(backup1[i])] < cap[BANDS.index(backup1[i])]:
                assign[i] = backup1[i]
            elif occ[BANDS.index(backup2[i])] < cap[BANDS.index(backup2[i])]:
                assign[i] = backup2[i]
            else:
                assign[i] = BANDS[int(np.argmax(cap - occ))]
        occ[BANDS.index(assign[i])] += 1
    defer_cap = np.maximum(band_deferral(sep, assign), defer_co)
    comb['full_cap'] = combined(dr_rows(sep, assign, defer_cap), kept, def_idx, def_cred)

    # evaluator band lifts (secondary) + per-segment
    r_full_band = score(sep, pi_full, defer_band, 'full_band')
    r_cap_band = score(sep, assign, band_deferral(sep, assign), 'cap_band')

    for k in comb:
        c = comb[k]
        print(f"[{k}] expected {c['expected']:.0f}  lift {c['lift']*100:+.1f}%  cpm {c['cpm']:.1f}  "
              f"per1000 {c['mp1k']:.1f}")

    # ---------------- report ----------------
    L = []
    L.append("# Backtest - P6a: full policy on locked September\n")
    L.append("Problem 3, Best Time to Call. September 2026 (locked, used once). Baseline: "
             f"{base_calls:,} dialed calls, {base_meet:,} meetings (meeting_fixed == 1), "
             f"{base_calls/base_meet:.1f} calls per meeting.\n")
    L.append("**Combined number (headline).** For each policy, expected meetings = the sum over kept calls of "
             "the doubly-robust estimate of P(meeting) at the policy's band and new delay, plus the credited "
             "meetings of deferred calls; calls = kept calls + deferred calls. CIs are 95% seller-bootstrap "
             "(1,000 resamples). Row (f) is the headline.\n")
    L.append("| policy | expected meetings | net lift vs current | 95% CI | calls per meeting | 95% CI | meetings per 1,000 calls | 95% CI |")
    L.append("|---|---|---|---|---|---|---|---|")
    def crow(nm, c):
        return (f"| {nm} | {c['expected']:.0f} | {c['lift']*100:+.1f}% | "
                f"[{c['lift_ci'][0]*100:+.1f}%, {c['lift_ci'][1]*100:+.1f}%] | {c['cpm']:.1f} | "
                f"[{c['cpm_ci'][0]:.1f}, {c['cpm_ci'][1]:.1f}] | {c['mp1k']:.1f} | "
                f"[{c['mp1k_ci'][0]:.1f}, {c['mp1k_ci'][1]:.1f}] |")
    L.append(crow("(a) current", comb['current']))
    L.append(crow("(b) global best band", comb['global_best']))
    L.append(crow("(c) C1 only", comb['c1']))
    L.append(crow("(d) LightGBM argmax band", comb['lgb_argmax']))
    L.append(crow("(e) full policy, unconstrained (deferred)", comb['full_uncon']))
    L.append(crow("(e') full policy, unconstrained (removed)", comb['full_removed']))
    L.append(crow("(f) full policy, capacity-constrained (deferred) **headline**", comb['full_cap']))
    L.append("")
    L.append("Capacity: current per-band volume +/-20%, assigned by marginal gain (model P(meeting) at the "
             "desired band), overflow to backup bands. Cool-offs are deferrals, not deletions: each deferred "
             "call is re-added at the end of the cool-off in the seller's recommended band, credited with the "
             "observed Apr-Aug meeting rate for (previous outcome x gap bucket x attempt stratum), and counted "
             "as a call. Cool-offs apply only to bot-initiated follow-ups inside an episode; a new hot-lead "
             "trigger is never suppressed.\n")
    L.append(f"- The full policy defers {n_deferred:,} cool-off calls, crediting {deferred_meet:.0f} meetings. "
             f"The 'removed' row (e') deletes them instead: calls fall but their meetings are charged as lost.\n")

    L.append("## Secondary: evaluator band lifts (full September)\n")
    L.append("| policy | DR lift | 95% CI | replay lift | connect |")
    L.append("|---|---|---|---|---|")
    for k, nm in [('current', '(a) current'), ('global_best', '(b) global best'), ('c1', '(c) C1'),
                  ('lgb_argmax', '(d) LGB argmax'), ('full_band', '(e) full uncon'),
                  ('cap_band', '(f) full capacity')]:
        r = results[k]
        L.append(f"| {nm} | {r['lift_dr']*100:+.1f}% | [{r['ci_dr'][0]*100:+.1f}%, {r['ci_dr'][1]*100:+.1f}%] | "
                 f"{r['lift_std']*100:+.1f}% | {r['val_dr_ans']*100:.1f}% |")
    L.append("")
    L.append("## Per-segment DR lift (vs current within segment, full September)\n")
    L.append("| policy | " + " | ".join(SEGS) + " |")
    L.append("|" + "---|" * (len(SEGS) + 1))
    for k, nm in [('global_best', '(b) global best'), ('c1', '(c) C1'), ('lgb_argmax', '(d) LGB argmax'),
                  ('full_band', '(e) full uncon'), ('cap_band', '(f) full capacity')]:
        r = results[k]
        cells = []
        for s in SEGS:
            g = r['seg'].get(s, {})
            cells.append(f"{g['lift']*100:+.1f}% (n={g['n']:,})" if g.get('n') else "-")
        L.append(f"| {nm} | " + " | ".join(cells) + " |")
    L.append("")
    L.append("- Fresh first attempts are fixed by design (~0); the gain sits in the movable segments.\n")
    L.append("## Read-out\n")
    L.append(f"- Headline (f): {comb['full_cap']['expected']:.0f} meetings ({comb['full_cap']['lift']*100:+.1f}% "
             f"vs current, 95% CI [{comb['full_cap']['lift_ci'][0]*100:+.1f}%, {comb['full_cap']['lift_ci'][1]*100:+.1f}%]) "
             f"on {comb['full_cap']['calls']:,} calls, i.e. {comb['full_cap']['cpm']:.1f} calls per meeting "
             f"({comb['full_cap']['mp1k']:.1f} meetings per 1,000 calls).")
    L.append(f"- Baseline: {base_meet:,} meetings, {base_calls/base_meet:.1f} calls per meeting, "
             f"{base_meet/base_calls*1000:.1f} meetings per 1,000 calls.")
    L.append("- (e)/(e') differ only by the deferred cool-off credit; the deferred version is the correct "
             "accounting.\n")
    open('/scratch/work/backtest_p6a.md', 'w').write("\n".join(L))
    print("wrote backtest_p6a.md")
    json.dump({k: v for k, v in comb.items()}, open('/scratch/work/backtest_p6a_combined.json', 'w'),
              indent=2, default=str)

if __name__ == '__main__':
    main()
