"""
folds.py — P6b(2): rolling folds.

For a test month M (jun/jul/aug), train on the prior months (last one as validation):
refit C1 timing (Beta-Binomial), the P(meeting) stop model, and the DR outcome /
propensity models on the prior months, then evaluate on M: current vs the two
operating points (Max meetings = tau 0, cool-offs on; Max efficiency = tau 0.0125,
cool-offs on), with the +/-20% capacity constraint and the combined accounting.

Usage: python folds.py <jun|jul|aug>
Writes folds_<month>.json
"""
import os, resource, sys
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
import backtest as bt
from backtest import load_all, band_deferral, credit_lookup
import timing_model as tm
from priority_model import NUM, CAT, kept_mask, NI_COOL_MIN, GEN_COOL_MIN

BANDS = ev.BANDS
ALL = ['2026-04', '2026-05', '2026-06', '2026-07', '2026-08']
MON = {'jun': '2026-06', 'jul': '2026-07', 'aug': '2026-08'}

def fit_priority(b, train_m, val_m):
    tr = b[b['month'].isin(train_m)]; va = b[b['month'] == val_m]
    feats = NUM + CAT
    clf = lgb.LGBMClassifier(n_estimators=3000, learning_rate=0.03, num_leaves=63,
                             min_child_samples=100, subsample=0.8, colsample_bytree=0.8,
                             reg_lambda=1.0, n_jobs=1, random_state=7, objective='binary')
    clf.fit(tr[feats], tr['y_meet'].values, eval_set=[(va[feats], va['y_meet'].values)],
            eval_metric='auc', callbacks=[lgb.early_stopping(80, verbose=False), lgb.log_evaluation(0)])
    it = clf.best_iteration_ or clf.n_estimators
    iso = IsotonicRegression(out_of_bounds='clip', y_min=0, y_max=1).fit(
        clf.predict_proba(va[feats], num_iteration=it)[:, 1], va['y_meet'].values)
    return clf, iso, it, feats

def c1_rec_band(train_m, val_m):
    tmb = tm.load()
    trf = tmb[tmb['fresh'] & tmb['month'].isin(train_m)]
    vaf = tmb[tmb['month'] == val_m]
    pre = tm.precompute(trf, 'y_meet')
    best = None
    for k in [20, 100]:
        for k2 in [5, 20]:
            m = tm.fit_from(pre, k, k2)
            nll = tm.nll_on(m, vaf, 'y_meet')
            if best is None or nll < best[0]:
                best = (nll, k, k2)
    m = tm.fit_from(pre, best[1], best[2])
    del tmb, trf, vaf, pre; gc.collect()
    s = pd.read_csv('/scratch/input/sellers.csv', usecols=['fk_glusr_usr_id', 'seller_state',
                    'annual_turnover', 'gst_registration_year', 'top_category_group'])
    s = s.rename(columns={'fk_glusr_usr_id': 'seller_id'})
    s['state'] = s['seller_state'].fillna('NA').astype(str)
    s['turnover'] = s['annual_turnover'].fillna('NA').astype(str)
    s['cat'] = s['top_category_group'].fillna('NA').astype(str)
    s['gst_bucket'] = pd.cut(s['gst_registration_year'], [0, 2015, 2019, 2023, 2027],
                             labels=['<=2015', '2016-2019', '2020-2023', '2024+']).astype(str)
    s['bucket'] = 'NA'
    Pm, _ = tm.predict_matrix(m, s, s['seller_id'].values)
    rec = np.array(tm.BANDS)[np.argmax(Pm, axis=1)]
    order = np.argsort(-Pm, axis=1)
    b1 = np.array(tm.BANDS)[order[:, 1]]; b2 = np.array(tm.BANDS)[order[:, 2]]
    return dict(zip(s['seller_id'], rec)), dict(zip(s['seller_id'], b1)), dict(zip(s['seller_id'], b2))

def main():
    key = sys.argv[1] if len(sys.argv) > 1 else 'aug'
    test_m = MON[key]
    prior = [m for m in ALL if m < test_m]
    val_m = prior[-1]; train_m = prior[:-1]
    print(f"fold {key}: test {test_m}, val {val_m}, train {train_m}")

    b = load_all()
    clf, iso, it, feats = fit_priority(b, train_m, val_m)
    b['p'] = iso.predict(clf.predict_proba(b[feats], num_iteration=it)[:, 1])
    credit, credit_default = credit_lookup(b)
    rec, bk1, bk2 = c1_rec_band(train_m, val_m)
    test = b[b['month'] == test_m].copy()
    del b; gc.collect()

    btr = ev.load(); btr = btr[btr['month'].isin(train_m)]
    bag = ev.load(); bag = bag[bag['month'] == val_m]
    prop = ev.fit_propensity(btr, bag)
    out_meet = ev.fit_outcome(btr, 'y_meet')
    del btr, bag; gc.collect()

    rp = json.load(open('/scratch/work/retry_policy.json'))
    retry = {(int(r['attempt_no']), r['failed_band']): r['recommended']['next_band']
             for r in rp['states']['not_answered']['decision_table']}
    fresh = test['seg_id'].values == 0
    logged = test['band'].values; fk = test['fk_glusr_usr_id'].values
    att = test['call_attempt_count'].astype('int64').values
    rec_band = np.array([rec.get(s, '19-20') for s in fk])
    backup1 = np.array([bk1.get(s, '17-18') for s in fk]); backup2 = np.array([bk2.get(s, '11-12') for s in fk])
    pi_full = rec_band.copy()
    is_na = (att >= 2) & (test['prev_label'].values == 'Not Answered'); pb = test['prev_band'].values
    for i in np.where(is_na)[0]:
        k = (int(att[i]) - 1, pb[i])
        if k in retry:
            pi_full[i] = retry[k]
    pi_full[fresh] = logged[fresh]
    pl = test['prev_label'].values; gm = test['gap_min'].values
    co_ni = (pl == 'Not Interested') & (gm < NI_COOL_MIN) & (att >= 2)
    co_gen = (pl == 'General (talked)') & (gm < GEN_COOL_MIN) & (att >= 2)
    defer_co = np.where(co_ni, NI_COOL_MIN, np.where(co_gen, GEN_COOL_MIN, 0.0))
    def gb_of(x):
        return ('<=1d' if x <= 1440 else '1-3d' if x <= 4320 else '3-7d' if x <= 10080
                else '7-14d' if x <= 20160 else '14-30d' if x <= 43200 else '>30d')
    ab = test['attempt_bucket'].values
    cred_row = np.array([credit.get((pl[i], gb_of(NI_COOL_MIN if co_ni[i] else GEN_COOL_MIN), ab[i]), credit_default)
                         for i in range(len(test))])
    cur_vol = pd.Series(logged).value_counts().reindex(BANDS).fillna(0).values
    cap = np.floor(cur_vol * 1.2)
    fresh_occ = np.zeros(len(BANDS))
    for i in np.where(fresh)[0]:
        fresh_occ[BANDS.index(logged[i])] += 1

    def dr_rows(pi, defer):
        cf = test.copy(); cf['band'] = pi
        cf['delay_bucket'] = pd.cut(test['delay_min'].values + defer, ev.DELAY_BINS, labels=ev.DELAY_LAB)
        q_pi = ev.outcome_pred(out_meet, cf, 'band'); q_log = ev.outcome_pred(out_meet, test, 'band')
        p_log = ev.prop_logged(prop, test)
        match = (logged == np.asarray(pi)).astype(float)
        w = np.clip(match / p_log, 0, np.percentile(match / p_log, 99))
        return q_pi + w * (test['y_meet'].values - q_log)

    def assign_capacity(kept):
        occ = fresh_occ.copy(); assign = pi_full.copy()
        idx = np.where(kept & ~fresh)[0]
        for i in idx[np.argsort(-test['p'].values[idx])]:
            want = pi_full[i]
            if occ[BANDS.index(want)] >= cap[BANDS.index(want)]:
                if occ[BANDS.index(backup1[i])] < cap[BANDS.index(backup1[i])]:
                    assign[i] = backup1[i]
                elif occ[BANDS.index(backup2[i])] < cap[BANDS.index(backup2[i])]:
                    assign[i] = backup2[i]
                else:
                    assign[i] = BANDS[int(np.argmax(cap - occ))]
            occ[BANDS.index(assign[i])] += 1
        return assign

    base_calls = len(test); base_meet = int(test['y_meet'].sum())

    def evaluate(tau, cooloff):
        allow = np.ones(len(test), bool) if tau <= 0 else (test['p'].values >= tau)
        co_removed = (co_ni | co_gen) if cooloff else np.zeros(len(test), bool)
        co_mask = co_removed & allow
        kept = kept_mask(test, allow) & ~co_removed
        assign = assign_capacity(kept)
        dcap = np.maximum(band_deferral(test, assign), defer_co if cooloff else 0.0)
        dr = dr_rows(assign, dcap)
        di = np.where(co_mask)[0]
        exp = dr[kept].sum() + cred_row[di].sum(); calls = int(kept.sum()) + len(di)
        return {'expected': float(exp), 'calls': int(calls), 'lift': float(exp / base_meet - 1),
                'cpm': float(calls / exp)}

    out = {'fold': key, 'test_month': test_m, 'train_months': train_m, 'val_month': val_m,
           'base_calls': base_calls, 'base_meet': base_meet, 'base_cpm': base_calls / base_meet,
           'current': {'expected': float(base_meet), 'calls': base_calls, 'lift': 0.0, 'cpm': base_calls / base_meet},
           'max_meetings': evaluate(0.0, True), 'max_efficiency': evaluate(0.0125, True)}
    print(json.dumps(out, indent=2))
    json.dump(out, open(f'/scratch/work/folds_{key}.json', 'w'), indent=2)

if __name__ == '__main__':
    main()
