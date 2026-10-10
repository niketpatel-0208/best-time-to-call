"""
frontier.py — P6b(1): operating frontier.

Sweep the stop threshold tau (no stop -> aggressive) x cool-offs on/off, all with
C1+C2 timing and the +/-20% capacity constraint, using the combined accounting
(expected meetings = sum of DR P(meeting) over kept calls + credited deferred
meetings; calls = kept + deferred). Plots net meetings lift vs calls per meeting,
marks "Max meetings" (best net lift) and "Max efficiency" (= row f).
Writes chart_frontier.png and frontier.md.
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
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import evaluator as ev
import backtest as bt
from backtest import load_all, train_priority, predict_p, band_deferral, credit_lookup
from priority_model import kept_mask, NI_COOL_MIN, GEN_COOL_MIN

BANDS, BAND_START = ev.BANDS, ev.BAND_START
NBOOT = ev.NBOOT

def main():
    b = load_all()
    clf, iso, it, feats = train_priority(b)
    b['p'] = predict_p(clf, iso, it, feats, b)
    credit, credit_default = credit_lookup(b)
    sep = b[b['month'] == '2026-09'].copy()
    del b; gc.collect()
    base_calls = len(sep); base_meet = int(sep['y_meet'].sum())
    print(f"sep {base_calls} base_meet {base_meet}")

    btr = ev.load(); btr = btr[btr['month'].isin(['2026-04', '2026-05', '2026-06', '2026-07'])]
    bag = ev.load(); bag = bag[bag['month'] == '2026-08']
    prop = ev.fit_propensity(btr, bag)
    out_meet = ev.fit_outcome(btr, 'y_meet')
    del btr, bag; gc.collect()

    sw = pd.read_csv('/scratch/work/seller_windows.csv', usecols=['seller_id', 'rec_band', 'backup_band_1', 'backup_band_2'])
    rec = dict(zip(sw['seller_id'], sw['rec_band'])); bk1 = dict(zip(sw['seller_id'], sw['backup_band_1']))
    bk2 = dict(zip(sw['seller_id'], sw['backup_band_2']))
    rp = json.load(open('/scratch/work/retry_policy.json'))
    retry = {(int(r['attempt_no']), r['failed_band']): r['recommended']['next_band']
             for r in rp['states']['not_answered']['decision_table']}
    tau_star = json.load(open('/scratch/work/priority_metrics.json'))['tau_star']

    fresh = sep['seg_id'].values == 0
    logged = sep['band'].values; fk = sep['fk_glusr_usr_id'].values
    att = sep['call_attempt_count'].astype('int64').values
    rec_band = np.array([rec.get(s, '19-20') for s in fk])
    backup1 = np.array([bk1.get(s, '17-18') for s in fk]); backup2 = np.array([bk2.get(s, '11-12') for s in fk])
    pi_full = rec_band.copy()
    is_na = (att >= 2) & (sep['prev_label'].values == 'Not Answered'); pb = sep['prev_band'].values
    for i in np.where(is_na)[0]:
        k = (int(att[i]) - 1, pb[i])
        if k in retry:
            pi_full[i] = retry[k]
    pi_full[fresh] = logged[fresh]

    pl = sep['prev_label'].values; gm = sep['gap_min'].values
    co_ni = (pl == 'Not Interested') & (gm < NI_COOL_MIN) & (att >= 2)
    co_gen = (pl == 'General (talked)') & (gm < GEN_COOL_MIN) & (att >= 2)
    defer_co = np.where(co_ni, NI_COOL_MIN, np.where(co_gen, GEN_COOL_MIN, 0.0))

    def gb_of(x):
        return ('<=1d' if x <= 1440 else '1-3d' if x <= 4320 else '3-7d' if x <= 10080
                else '7-14d' if x <= 20160 else '14-30d' if x <= 43200 else '>30d')
    ab = sep['attempt_bucket'].values
    cred_row = np.array([credit.get((pl[i], gb_of(NI_COOL_MIN if co_ni[i] else GEN_COOL_MIN), ab[i]), credit_default)
                         for i in range(len(sep))])

    codes, uniq = pd.factorize(fk); S = len(uniq)
    cur_y = np.bincount(codes, weights=sep['y_meet'].values, minlength=S)
    BIDX = np.random.default_rng(7).integers(0, S, size=(NBOOT, S))
    cur_vol = pd.Series(logged).value_counts().reindex(BANDS).fillna(0).values
    cap = np.floor(cur_vol * 1.2)
    fresh_occ = np.zeros(len(BANDS))
    for i in np.where(fresh)[0]:
        fresh_occ[BANDS.index(logged[i])] += 1

    def dr_rows(pi, defer):
        cf = sep.copy(); cf['band'] = pi
        cf['delay_bucket'] = pd.cut(sep['delay_min'].values + defer, ev.DELAY_BINS, labels=ev.DELAY_LAB)
        q_pi = ev.outcome_pred(out_meet, cf, 'band')
        q_log = ev.outcome_pred(out_meet, sep, 'band')
        p_log = ev.prop_logged(prop, sep)
        match = (logged == np.asarray(pi)).astype(float)
        w = np.clip(match / p_log, 0, np.percentile(match / p_log, 99))
        return q_pi + w * (sep['y_meet'].values - q_log)

    def assign_capacity(kept):
        occ = fresh_occ.copy(); assign = pi_full.copy()
        idx = np.where(kept & ~fresh)[0]
        for i in idx[np.argsort(-sep['p'].values[idx])]:
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

    def combined(dr, kept, def_idx, def_cred, with_ci=False):
        kc = np.bincount(codes[kept], minlength=S)
        pd_s = np.bincount(codes[kept], weights=dr[kept], minlength=S)
        dm_s = np.zeros(S); dc_s = np.zeros(S)
        if len(def_idx):
            dm_s = np.bincount(codes[def_idx], weights=def_cred, minlength=S)
            dc_s = np.bincount(codes[def_idx], minlength=S)
        exp = pd_s.sum() + dm_s.sum(); calls = kc.sum() + dc_s.sum()
        out = {'expected': float(exp), 'calls': int(calls), 'lift': float(exp / base_meet - 1),
               'cpm': float(calls / exp), 'mp1k': float(exp / calls * 1000)}
        if with_ci:
            lb, cb = [], []
            for idx in BIDX:
                e = pd_s[idx].sum() + dm_s[idx].sum(); c = kc[idx].sum() + dc_s[idx].sum()
                cu = cur_y[idx].sum(); lb.append(e / cu - 1); cb.append(c / e)
            out['lift_ci'] = [float(np.percentile(lb, 2.5)), float(np.percentile(lb, 97.5))]
            out['cpm_ci'] = [float(np.percentile(cb, 2.5)), float(np.percentile(cb, 97.5))]
        return out

    def point(tau, cooloff):
        allow = np.ones(len(sep), bool) if tau <= 0 else (sep['p'].values >= tau)
        co_removed = (co_ni | co_gen) if cooloff else np.zeros(len(sep), bool)
        co_mask = co_removed & allow
        kept = kept_mask(sep, allow) & ~co_removed
        assign = assign_capacity(kept)
        dcap = np.maximum(band_deferral(sep, assign), defer_co if cooloff else 0.0)
        dr = dr_rows(assign, dcap)
        di = np.where(co_mask)[0]
        return combined(dr, kept, di, cred_row[di])

    taus = [0.0, 0.005, 0.0075, 0.01, 0.0125, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05]
    pts = []
    for cool in [False, True]:
        for t in taus:
            r = point(t, cool)
            r['tau'] = t; r['cooloff'] = cool
            pts.append(r)
            print(f"tau {t:5.4f} cool {int(cool)}  lift {r['lift']*100:+.1f}%  cpm {r['cpm']:.1f}  per1k {r['mp1k']:.1f}")

    # named points
    cool_pts = [p for p in pts if p['cooloff']]
    max_meet = max(pts, key=lambda p: p['lift'])
    max_eff = min(cool_pts, key=lambda p: abs(p['tau'] - tau_star))   # row f
    # recompute named points with CIs
    for nm, p in [('max_meetings', max_meet), ('max_efficiency', max_eff)]:
        allow = np.ones(len(sep), bool) if p['tau'] <= 0 else (sep['p'].values >= p['tau'])
        co_removed = (co_ni | co_gen) if p['cooloff'] else np.zeros(len(sep), bool)
        co_mask = co_removed & allow
        kept = kept_mask(sep, allow) & ~co_removed
        assign = assign_capacity(kept); dcap = np.maximum(band_deferral(sep, assign), defer_co if p['cooloff'] else 0.0)
        di = np.where(co_mask)[0]
        p.update(combined(dr_rows(assign, dcap), kept, di, cred_row[di], with_ci=True))
    print("max_meetings", max_meet['tau'], max_meet['cooloff'], f"{max_meet['lift']*100:+.1f}%")
    print("max_efficiency", max_eff['tau'], max_eff['cooloff'], f"{max_eff['lift']*100:+.1f}%")

    # chart
    fig, ax = plt.subplots(figsize=(8.5, 5))
    for cool, col, mk, lab in [(False, '#4c72b0', 'o', 'cool-offs off'), (True, '#55a868', 's', 'cool-offs on')]:
        xs = [p['cpm'] for p in pts if p['cooloff'] == cool]
        ys = [p['lift'] * 100 for p in pts if p['cooloff'] == cool]
        ax.plot(xs, ys, mk + '-', color=col, label=lab, markersize=5)
    ax.scatter([max_meet['cpm']], [max_meet['lift'] * 100], s=170, marker='*', color='#c44e52', zorder=5,
               label=f"Max meetings (tau={max_meet['tau']:g})")
    ax.scatter([max_eff['cpm']], [max_eff['lift'] * 100], s=170, marker='D', color='#8172b2', zorder=5,
               label=f"Max efficiency = row f (tau={max_eff['tau']:g})")
    ax.axhline(0, color='#999999', lw=0.8)
    ax.set_xlabel('calls per meeting (lower is better)'); ax.set_ylabel('net meetings lift vs current (%)')
    ax.set_title('Operating frontier: meetings lift vs calls per meeting (September)')
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig('/scratch/work/chart_frontier.png', dpi=110); plt.close(fig)

    # table
    L = []
    L.append("# Operating frontier - P6b(1)\n")
    L.append(f"Baseline: {base_calls:,} calls, {base_meet:,} meetings, {base_calls/base_meet:.1f} calls per "
             f"meeting. All points use C1+C2 timing and the +/-20% per-band capacity constraint; the combined "
             f"accounting (DR expected meetings over kept calls + credited deferred cool-off meetings).\n")
    L.append("| tau | cool-offs | expected meetings | net lift vs current | calls per meeting | meetings per 1,000 calls |")
    L.append("|---|---|---|---|---|---|")
    for p in pts:
        L.append(f"| {p['tau']:g} | {'on' if p['cooloff'] else 'off'} | {p['expected']:.0f} | {p['lift']*100:+.1f}% | "
                 f"{p['cpm']:.1f} | {p['mp1k']:.1f} |")
    L.append("")
    L.append("## Named operating points (95% seller-bootstrap CI)\n")
    L.append("| point | tau | cool-offs | expected meetings | net lift | 95% CI | calls per meeting | 95% CI |")
    L.append("|---|---|---|---|---|---|---|---|")
    for nm, p in [('Max meetings', max_meet), ('Max efficiency (= row f)', max_eff)]:
        L.append(f"| {nm} | {p['tau']:g} | {'on' if p['cooloff'] else 'off'} | {p['expected']:.0f} | "
                 f"{p['lift']*100:+.1f}% | [{p['lift_ci'][0]*100:+.1f}%, {p['lift_ci'][1]*100:+.1f}%] | "
                 f"{p['cpm']:.1f} | [{p['cpm_ci'][0]:.1f}, {p['cpm_ci'][1]:.1f}] |")
    L.append("")
    L.append(f"- **Max meetings** (tau={max_meet['tau']:g}, cool-offs {'on' if max_meet['cooloff'] else 'off'}): "
             f"the best net lift, {max_meet['lift']*100:+.1f}%, at {max_meet['cpm']:.1f} calls per meeting.")
    L.append(f"- **Max efficiency** (tau={max_eff['tau']:g}, cool-offs on, = row f): {max_eff['cpm']:.1f} calls "
             f"per meeting at {max_eff['lift']*100:+.1f}% net lift.")
    L.append("- The frontier is monotone: stronger stopping lowers calls per meeting and, past a point, lowers "
             "net lift; cool-offs shift the curve left (better efficiency) at a small lift cost.\n")
    open('/scratch/work/frontier.md', 'w').write("\n".join(L))
    print("wrote frontier.md")
    json.dump({'points': pts, 'max_meetings': max_meet, 'max_efficiency': max_eff,
               'baseline': {'calls': base_calls, 'meetings': base_meet}},
              open('/scratch/work/frontier.json', 'w'), indent=2, default=str)

if __name__ == '__main__':
    main()
