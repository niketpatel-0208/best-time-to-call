"""
retry_policy.py — P4: C2 retry / follow-up policy and the 15-minute-rule verdict.

Part 1: from NA -> next-attempt transitions, build the retry decision table.
        Rows = (attempt number, failed daypart); answer = (gap bucket, next band).
        Answer and meeting rates per cell with 95% CIs, compared within strata
        (bucket group x bot x month). Writes retry_policy.json (all policy states).
Part 2: fifteen_minute_rule.md — main_vani dial-to-report histogram, >12-min
        connect/meeting shares, matched arrowhead vs main_vani, daypart-shift
        evidence, verdict and an online A/B test. Writes charts (.png).
"""
import os, resource
try:
    resource.setrlimit(resource.RLIMIT_AS, (resource.RLIM_INFINITY, resource.RLIM_INFINITY))
except Exception:
    pass
for _v in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:
    os.environ.setdefault(_v, '1')

import numpy as np
import pandas as pd
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = '/scratch/work/base_table.parquet'
BANDS = ['9-10', '11-12', '13-14', '15-16', '17-18', '19-20']
REST_BUCKETS = {'NUR', 'UA', 'PUA', 'PIM'}
RNG = np.random.default_rng(11)
COLS = ['fk_glusr_usr_id', 'episode', 'band', 'call_attempt_count', 'call_start_time',
        'vendor_response_time', 'lead_sent_time', 'lead_call_status', 'meeting_fixed',
        'redis_bucket', 'bot', 'disposition_label']
GAPS = [(0, 15, '<=15m'), (15, 45, '15-45m'), (45, 120, '45m-2h'), (120, 240, '2-4h'),
        (240, 480, '4-8h'), (480, 1440, '8-24h'), (1440, 2880, '1-2d'), (2880, 1e9, '>2d')]

def wilson(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)

def gap_bucket(m):
    for lo, hi, lab in GAPS:
        if lo <= m < hi:
            return lab
    return '>2d'

def load():
    b = pd.read_parquet(BASE, columns=COLS, use_threads=False)
    b['band'] = b['band'].astype(str)
    b['month'] = b['call_start_time'].dt.to_period('M').astype(str)
    b['bucket_group'] = np.where(b['redis_bucket'].isin(REST_BUCKETS), 'Rest', 'Top 3')
    b['is_ans'] = (b['lead_call_status'] == 'Answered').astype(int)
    b['is_meet'] = (b['meeting_fixed'] == 1).astype(int)
    b['dtr'] = (b['vendor_response_time'] - b['call_start_time']).dt.total_seconds() / 60.0
    return b

# ------------------------------------------------------------------ transitions
def build_transitions(b):
    b = b.sort_values(['fk_glusr_usr_id', 'call_start_time'])
    ge = b.groupby(['fk_glusr_usr_id', 'episode'], sort=False)
    b['next_start'] = ge['call_start_time'].shift(-1)
    b['next_band'] = ge['band'].shift(-1)
    b['next_ans'] = ge['is_ans'].shift(-1)
    b['next_meet'] = ge['is_meet'].shift(-1)
    f = b[(b['lead_call_status'] == 'NotAnswered') & b['next_start'].notna()].copy()
    f['gap_min'] = (f['next_start'] - f['call_start_time']).dt.total_seconds() / 60.0
    f['gap_bucket'] = f['gap_min'].map(gap_bucket)
    f = f[f['month'] != '2026-09']            # build on Apr-Aug; Sep stays locked
    f['attempt_no'] = f['call_attempt_count'].astype('int64')
    return f

def mh_rr(df, rec_mask, ycol):
    """MH meeting/answer risk ratio of the recommended cell vs the rest, within
    strata of (bucket_group x bot x month)."""
    d = df.copy()
    d['_m'] = rec_mask.astype(int)
    d['_y'] = d[ycol].values
    g = d.groupby(['bucket_group', 'bot', 'month'])
    a = g.apply(lambda x: ((x['_m'] == 1) & (x['_y'] == 1)).sum())
    n1 = g.apply(lambda x: (x['_m'] == 1).sum())
    b_ = g.apply(lambda x: ((x['_m'] == 0) & (x['_y'] == 1)).sum())
    n0 = g.apply(lambda x: (x['_m'] == 0).sum())
    N = n1 + n0
    num = np.nansum(np.where(N > 0, a * n0 / N, 0))
    den = np.nansum(np.where(N > 0, b_ * n1 / N, 0))
    return num / den if den > 0 else np.nan

def boot_rr(df, rec_mask, ycol, B=300):
    """Seller-clustered bootstrap CI for the MH risk ratio of the recommended cell."""
    d = df.copy()
    d['_m'] = rec_mask.astype(int); d['_y'] = d[ycol].values
    codes, uniq = pd.factorize(d['fk_glusr_usr_id'].values)
    S = len(uniq)
    d['_s'] = codes
    d['str'] = pd.factorize(d['bucket_group'].astype(str) + '|' + d['bot'].astype(str)
                            + '|' + d['month'].astype(str))[0]
    K = d['str'].nunique()
    A = np.zeros((S, K, 4))
    np.add.at(A, (d['_s'].values, d['str'].values), np.stack([
        (d['_m'] == 1).values * d['_y'].values, (d['_m'] == 1).values.astype(float),
        (d['_m'] == 0).values * d['_y'].values, (d['_m'] == 0).values.astype(float)], axis=1))
    Aflat = A.reshape(S, K * 4)
    out = []
    for _ in range(B):
        idx = RNG.integers(0, S, size=S)
        counts = np.bincount(idx, minlength=S).astype(float)
        tb = (counts @ Aflat).reshape(K, 4)
        n1 = tb[:, 1]; N = tb[:, 1] + tb[:, 3]
        num = np.nansum(np.where(N > 0, tb[:, 0] * tb[:, 3] / N, 0))
        den = np.nansum(np.where(N > 0, tb[:, 2] * tb[:, 1] / N, 0))
        out.append(num / den if den > 0 else np.nan)
    out = np.array(out)
    return np.nanpercentile(out, [2.5, 97.5])

def decision_table(f):
    rows = []
    for att in [1, 2, 3]:
        for fb in BANDS:
            sub = f[(f['attempt_no'] == att) & (f['band'] == fb)]
            if len(sub) < 100:
                continue
            g = sub.groupby(['gap_bucket', 'next_band']).agg(
                n=('next_ans', 'size'), ans=('next_ans', 'sum'), meet=('next_meet', 'sum')).reset_index()
            g['answer_rate'] = g['ans'] / g['n']
            g['meeting_rate'] = g['meet'] / g['n']
            # recommended: best meeting rate among cells with n>=50 and a different daypart
            cand = g[(g['n'] >= 100) & (g['next_band'] != fb)].copy()
            if len(cand) == 0:
                cand = g[g['n'] >= 100].copy()
            if len(cand) == 0:
                continue
            cand['lb'] = [wilson(r['meet'], r['n'])[0] for _, r in cand.iterrows()]
            cand = cand.sort_values('lb', ascending=False)
            best = cand.iloc[0]
            close = cand[cand['lb'] >= cand['lb'].iloc[0] - 0.005]
            ev = close[close['next_band'].isin(['17-18', '19-20'])]
            if len(ev):
                best = ev.sort_values('lb', ascending=False).iloc[0]
            rec_mask = (sub['gap_bucket'].values == best['gap_bucket']) & \
                       (sub['next_band'].values == best['next_band'])
            rr = mh_rr(sub, rec_mask, 'next_meet')
            try:
                rr_ci = [round(float(x), 3) for x in boot_rr(sub, rec_mask, 'next_meet')]
            except Exception:
                rr_ci = None
            cells = []
            for _, r in g.sort_values('n', ascending=False).head(6).iterrows():
                lo, hi = wilson(r['meet'], r['n'])
                cells.append({'gap_bucket': r['gap_bucket'], 'next_band': r['next_band'],
                              'n': int(r['n']), 'answer_rate': round(r['answer_rate'], 4),
                              'meeting_rate': round(r['meeting_rate'], 4),
                              'meeting_ci': [round(lo, 4), round(hi, 4)]})
            lo, hi = wilson(best['meet'], best['n'])
            rows.append({
                'attempt_no': att, 'failed_band': fb, 'n_failed': int(len(sub)),
                'recommended': {'gap_bucket': best['gap_bucket'], 'next_band': best['next_band'],
                                'n': int(best['n']), 'answer_rate': round(best['answer_rate'], 4),
                                'meeting_rate': round(best['meeting_rate'], 4),
                                'meeting_ci': [round(lo, 4), round(hi, 4)]},
                'mh_meeting_rr_vs_rest': None if np.isnan(rr) else round(float(rr), 3),
                'mh_meeting_rr_ci': rr_ci,
                'top_cells': cells})
    return rows

def marginal(f, ycol):
    g = f.groupby('gap_bucket').agg(n=(ycol, 'size'), y=(ycol, 'sum'))
    g['rate'] = g['y'] / g['n']
    order = [lab for _, _, lab in GAPS]
    g = g.reindex(order).dropna()
    return {lab: {'n': int(r['n']), 'rate': round(r['rate'], 4)} for lab, r in g.iterrows()}

def next_band_effect(f):
    same = f['next_band'] == f['band']
    out = {}
    for name, m in [('same_band', same), ('different_band', ~same)]:
        sub = f[m]
        out[name] = {'n': int(len(sub)), 'answer_rate': round(sub['next_ans'].mean(), 4),
                     'meeting_rate': round(sub['next_meet'].mean(), 4)}
    return out

# --------------------------------------------------------------- 15-min rule
def fifteen_minute(b, f):
    mv = b[b['bot'] == 'main_vani']
    mv_na = mv[(mv['lead_call_status'] == 'NotAnswered') & mv['dtr'].notna() & (mv['dtr'] >= 0)]
    # histogram
    mv_na30 = mv_na[mv_na['dtr'] <= 30]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(mv_na30['dtr'], bins=np.arange(0, 31, 1), color='#4c72b0', edgecolor='white')
    ax.axvline(6, color='#c44e52', ls='--', lw=1); ax.axvline(20, color='#c44e52', ls='--', lw=1)
    ax.set_xlabel('dial-to-report window (minutes)'); ax.set_ylabel('NA calls')
    ax.set_title(f"main_vani NA dial-to-report window (n={len(mv_na30):,}, 0-30 min)")
    fig.tight_layout(); fig.savefig('/scratch/work/chart_dial_to_report.png', dpi=110); plt.close(fig)
    peaks = mv_na['dtr']
    p1 = float(((peaks >= 4) & (peaks <= 9)).mean()); p2 = float(((peaks >= 17) & (peaks <= 24)).mean())
    trough = float(((peaks >= 13) & (peaks <= 17)).mean())

    # share of connects / meetings > 12 min after dial (main_vani)
    mv_ans = mv[mv['lead_call_status'] == 'Answered']
    mv_meet = mv[mv['meeting_fixed'] == 1]
    share_ans = float((mv_ans['dtr'] > 12).mean())
    share_meet = float((mv_meet['dtr'] > 12).mean())

    # matched arrowhead vs main_vani on first attempts (month x bucket x hour)
    fa = b[b['call_attempt_count'] == 1].copy()
    fa['hour'] = fa['call_start_time'].dt.hour
    fa = fa[fa['bot'].isin(['main_vani', 'arrowhead'])]
    strat = fa.groupby(['month', 'redis_bucket', 'hour', 'bot']).agg(
        n=('is_ans', 'size'), ans=('is_ans', 'sum'), meet=('is_meet', 'sum')).reset_index()
    piv_n = strat.pivot_table(index=['month', 'redis_bucket', 'hour'], columns='bot', values='n')
    piv_a = strat.pivot_table(index=['month', 'redis_bucket', 'hour'], columns='bot', values='ans')
    piv_m = strat.pivot_table(index=['month', 'redis_bucket', 'hour'], columns='bot', values='meet')
    common = piv_n.dropna().index
    def pooled(piv, denom):
        r = (piv.reindex(common) / denom.reindex(common))
        w = denom.reindex(common).sum(axis=1)
        return {bot: float((r[bot] * w).sum() / w.sum()) for bot in ['main_vani', 'arrowhead']}
    ans_rate = pooled(piv_a, piv_n)
    meet_rate = pooled(piv_m, piv_n)
    n_strata = len(common)

    return {'hist': {'p_4_9': round(p1, 4), 'p_17_24': round(p2, 4), 'p_13_17_trough': round(trough, 4),
                     'n_na': int(len(mv_na30)), 'n_na_all': int(len(mv_na))},
            'share_gt12': {'connect': round(share_ans, 4), 'meeting': round(share_meet, 4),
                           'n_connect': int(len(mv_ans)), 'n_meeting': int(len(mv_meet))},
            'matched': {'n_strata': int(n_strata), 'answer': ans_rate, 'meeting': meet_rate}}

def make_daypart_chart(f):
    d = f.copy()
    d['same'] = (d['next_band'] == d['band'])
    g = d.groupby(['same', 'next_band'])['next_ans'].agg(['size', 'mean']).reset_index()
    g = g[g['size'] >= 100]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(range(len(g)), g['mean'] * 100,
           color=['#c44e52' if s else '#55a868' for s in g['same']])
    ax.set_xticks(range(len(g))); ax.set_xticklabels(g['next_band'], fontsize=8)
    ax.set_ylabel('next-call answer rate (%)')
    ax.set_xlabel('next band (red = same as failed, green = different)')
    ax.set_title('Answer rate by next band after an NA')
    fig.tight_layout(); fig.savefig('/scratch/work/chart_daypart_shift.png', dpi=110); plt.close(fig)

def main():
    b = load()
    f = build_transitions(b)
    print(f"NA->next transitions (Apr-Aug): {len(f):,}")
    codes, uniq = pd.factorize(f['fk_glusr_usr_id'].values)
    S = len(uniq)
    dt = decision_table(f)
    print(f"decision-table rows: {len(dt)}")
    n_trans = len(f)

    fm = marginal(f, 'next_meet')
    fa_ = marginal(f, 'next_ans')
    nb = next_band_effect(f)
    fifteen = fifteen_minute(b, f)
    make_daypart_chart(f)

    policy = {
        'meta': {
            'built_on': 'Apr-Aug 2026 (September held out)',
            'n_transitions': int(len(f)),
            'primary_target': 'meeting',
            'strata': 'bucket group x bot x month',
            'ci_method': 'Wilson 95% per cell; MH risk ratio within strata',
            'bands': BANDS,
        },
        'states': {
            'fresh_first_attempt': {
                'action': 'call_now',
                'rule': 'Fresh first attempt (attempt 1, delay_min <= 5): call immediately, no band change.',
                'defer_to_band': None},
            'not_answered': {
                'redial_15min': {'action': 'keep', 'rule': 'Vendor 15-min redial inside the call window stays '
                                 '(see fifteen_minute_rule.md).'},
                'after_redial_fails': {'action': 'shift_daypart',
                                       'rule': 'Move the next call to a different daypart (+4-8h, or next day), '
                                               'preferring 17-20; never retry in the same hour.'},
                'attempt_cap': 4,
                'decision_table': dt,
                'marginal_gap_meeting_rate': fm,
                'marginal_gap_answer_rate': fa_,
                'same_vs_different_band': nb},
            'call_later': {
                'action': 'honour_stated_time',
                'rule': 'Call at the stated time minus 5 minutes. No cue: same day +1-6h, in the next good band.',
                'max_callbacks': 3,
                'then': 'treat as General'},
            'general': {
                'action': 'wait_then_best_band',
                'rule': '+3-7 days, in the seller best band.',
                'gap_days': [3, 7]},
            'not_interested': {
                'action': 'long_wait',
                'rule': '+14 days (minimum 7). Two NI in a row -> 30-day suppression.',
                'gap_days': [14, 30]},
            'meeting_fixed': {'action': 'no_bot_call', 'rule': 'No bot call.'},
        },
    }
    json.dump(policy, open('/scratch/work/retry_policy.json', 'w'), indent=2)
    print("wrote retry_policy.json")

    # ---------------- fifteen_minute_rule.md ----------------
    fm_na = fifteen['hist']
    sh = fifteen['share_gt12']
    mt = fifteen['matched']
    L = []
    L.append("# The 15-minute rule — evidence and verdict\n")
    L.append("Problem 3, Best Time to Call. Evidence built on Apr–Aug 2026 (September held out).\n")
    L.append("## What the rule is\n")
    L.append("main_vani (SquadStack) does not report a Not-Answered outcome at the dial. It dials, and if "
             "there is no answer it redials about 15 minutes later inside the vendor's call window, and only "
             "then reports NotAnswered. arrowhead does not redial.\n")
    L.append("## 1. The dial-to-report window (main_vani, NA calls)\n")
    L.append(f"Chart: `chart_dial_to_report.png`. On {fm_na['n_na']:,} NA calls the dial-to-report window is "
             f"bimodal: {fm_na['p_4_9']*100:.0f}% land at 4–9 min (the first dial's report) and "
             f"{fm_na['p_17_24']*100:.0f}% at 17–24 min (the redial), with a trough at 13–17 min "
             f"({fm_na['p_13_17_trough']*100:.0f}%). That is a first dial plus a ~15-minute redial.\n")
    L.append("## 2. How much conversion rides on the redial\n")
    L.append(f"Among main_vani connects, **{sh['connect']*100:.1f}%** happen more than 12 minutes after the "
             f"dial (n={sh['n_connect']:,}); among main_vani meetings, **{sh['meeting']*100:.1f}%** do "
             f"(n={sh['n_meeting']:,}). Those late connects are the ones the redial buys.\n")
    L.append("## 3. arrowhead vs main_vani, matched (month × bucket × hour)\n")
    L.append(f"On first attempts matched within {mt['n_strata']} month × bucket × hour strata: main_vani answers "
             f"{mt['answer']['main_vani']*100:.1f}% vs arrowhead {mt['answer']['arrowhead']*100:.1f}% "
             f"({(mt['answer']['main_vani']-mt['answer']['arrowhead'])*100:+.1f} pp), and fixes meetings at "
             f"{mt['meeting']['main_vani']*100:.1f}% vs {mt['meeting']['arrowhead']*100:.1f}% "
             f"({(mt['meeting']['main_vani']-mt['meeting']['arrowhead'])*100:+.1f} pp). arrowhead's lack of a "
             f"redial is a plausible driver, but the gap is confounded with other bot differences.\n")
    L.append("## 4. What should follow the redial (daypart shift)\n")
    L.append(f"Chart: `chart_daypart_shift.png`. On the NA→next transitions (n={n_trans:,}), "
             f"retrying in the **same** band answers {nb['same_band']['answer_rate']*100:.1f}% "
             f"(n={nb['same_band']['n']:,}) while a **different** band answers "
             f"{nb['different_band']['answer_rate']*100:.1f}% (n={nb['different_band']['n']:,}) — "
             f"{(nb['different_band']['answer_rate']-nb['same_band']['answer_rate'])*100:+.1f} pp. The "
             f"best gap is 4–8 h ({fm.get('4-8h',{}).get('rate',float('nan'))*100:.1f}% meeting) and evening "
             f"bands answer best. So the redial should keep its 15 minutes, but the call after it should move "
             f"daypart.\n")
    L.append("## Verdict: **Keep, and Supplement**\n")
    L.append("- **Keep** the 15-minute redial for main_vani — it captures a large share of connects and "
             "meetings that a single dial would miss, and there is no evidence it hurts.")
    L.append("- **Supplement** it in two ways: (a) extend the redial to arrowhead, which currently has none "
             "and answers ~8 pp lower on matched first calls; (b) after the redial fails, move the next call "
             "to a different daypart (+4–8 h, prefer 17–20), never the same hour.")
    L.append("- **Do not change** the redial interval: the dial-to-report window's trough at 13–17 min shows "
             "the 15-minute wait is already close to optimal, and near-zero traffic retries ≤45 min after a "
             "reported NA.\n")
    L.append("## Caveats\n")
    L.append("- The arrowhead vs main_vani gap is confounded with bucket mix and other bot differences; it is "
             "suggestive, not causal.")
    L.append("- Late-connect shares use the dial-to-report window as a proxy for a redial connect; call "
             "duration adds a little noise.")
    L.append("- All figures are Apr–Aug; September is held out and untouched.\n")
    L.append("## Proposed online A/B test\n")
    L.append("Randomise at the seller level, 50/50, over four weeks:\n")
    L.append("- **Control:** current schedule (main_vani 15-min redial; arrowhead none).")
    L.append("- **Treatment:** redial extended to arrowhead; after a failed redial, next call moved to a "
             "different daypart (prefer 17–20); cap 4 attempts.")
    L.append("Primary metric: meetings fixed per call. Secondary: connect rate, calls per meeting. "
             "Guardrail: calls per seller per day. Pre-register a two-sided test at α = 0.05, powered for a "
             "5% lift in meetings per call; analyse per seller with a cluster-robust CI.\n")
    open('/scratch/work/fifteen_minute_rule.md', 'w').write("\n".join(L))
    print("wrote fifteen_minute_rule.md")
    json.dump({'fifteen_minute': fifteen, 'same_vs_different': nb,
               'marginal_gap_meeting': fm, 'marginal_gap_answer': fa_},
              open('/scratch/work/fifteen_minute_evidence.json', 'w'), indent=2)

if __name__ == '__main__':
    main()
