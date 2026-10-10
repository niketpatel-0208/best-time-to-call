"""Full-system September backtest: model (timing policy) + LLM callback layer (summaries)."""
import sys, numpy as np, pandas as pd, datetime as dt
R, S = sys.argv[1], sys.argv[2]
e = pd.read_pickle(S + '/sep_ext.pkl')
a = pd.read_csv(R + '/cowork_upload/data/attempts.csv', parse_dates=['call_start_time'],
                usecols=['data_hotlead_disposition_dtlid', 'fk_glusr_usr_id', 'call_start_time', 'call_attempt_count',
                         'meeting_fixed', 'disposition_label', 'lead_call_status'])
a = a[a.call_start_time.notna()].sort_values(['fk_glusr_usr_id', 'call_start_time']).reset_index(drop=True)
g = a.groupby('fk_glusr_usr_id')
a['n_id'] = g.data_hotlead_disposition_dtlid.shift(-1)
a['n_time'] = g.call_start_time.shift(-1)
a['n_mf'] = g.meeting_fixed.shift(-1)
a['n_ans'] = (g.lead_call_status.shift(-1) == 'Answered').astype(float).where(a.n_time.notna())
a['n_att'] = g.call_attempt_count.shift(-1)
e = e.merge(a[['data_hotlead_disposition_dtlid', 'n_id', 'n_time', 'n_mf', 'n_ans', 'n_att', 'call_attempt_count']],
            on='data_hotlead_disposition_dtlid', how='left')
sep = (a.call_start_time >= '2026-09-01')
base_calls, base_mf = int(sep.sum()), int(a.loc[sep, 'meeting_fixed'].sum())
print('baseline Sep', base_calls, base_mf, round(base_calls / base_mf, 2))

# ---------- 1. timing layer: calls where the seller gave a time / hint ----------
c = e[(e.disposition_label != 'Meeting Fixed') & (e.cue_strength != 'none') & ~e.wrong_number & ~e.dnc].copy()
c['has_next'] = c.n_time.notna()
print('\ncued follow-up calls', len(c), 'with a next attempt', int(c.has_next.sum()))
c = c[c.has_next].copy()

def honoured(r):
    n = r.n_time
    if pd.notna(r.target):
        return abs((n - r.target).total_seconds()) <= 30 * 60
    if pd.notna(r.win_start):
        return r.win_start <= n <= r.win_end
    if r.earliest is not None and not pd.isna(r.earliest):
        return r.earliest <= n.date() <= r.earliest + dt.timedelta(days=2)
    return False

c['hon'] = c.apply(honoured, axis=1)
def timing_class(r):
    if r.hon: return 'on_time'
    ref = r.target if pd.notna(r.target) else (r.win_start if pd.notna(r.win_start) else (pd.Timestamp(r.earliest) if r.earliest is not None and not pd.isna(r.earliest) else None))
    if ref is None: return 'other'
    return 'early' if r.n_time < ref else 'late'
c['timing'] = c.apply(timing_class, axis=1)
c['stratum'] = c.cue_strength + '|' + np.where(c.disposition_label == 'Call Later / Busy', 'CL', 'other')
print(pd.crosstab(c.stratum, c.timing, margins=True))
t = c.groupby(['stratum', 'hon']).agg(n=('n_mf', 'size'), mf=('n_mf', 'mean'), ans=('n_ans', 'mean')).unstack()
print('\nnext-call meeting rate by honoured (True) vs not (False)'); print(t.round(4))
print('\nby timing:'); print(c.groupby('timing').agg(n=('n_mf', 'size'), mf=('n_mf', 'mean'), ans=('n_ans', 'mean')).round(4))

L_MODEL = 0.174  # model's own DR lift for callbacks & follow-ups (backtest_report sec.5, the larger value = conservative)

def delta(df):
    out = 0.0
    for s, d in df.groupby('stratum'):
        h, u = d[d.hon], d[~d.hon]
        if len(h) < 30 or len(u) < 30:
            continue
        rh, ru = h.n_mf.mean(), u.n_mf.mean()
        out += len(u) * (rh - ru * (1 + L_MODEL))  # only the not-honoured calls change under our system
    return out

D = delta(c)
rng = np.random.default_rng(7)
sellers = c.fk_glusr_usr_id.unique()
idx = {s: i for i, s in enumerate(sellers)}
c['sidx'] = c.fk_glusr_usr_id.map(idx)
grp = [c.index[c.sidx == i] for i in range(len(sellers))]
boots = []
for _ in range(400):
    pick = rng.integers(0, len(sellers), len(sellers))
    boots.append(delta(c.loc[np.concatenate([grp[i] for i in pick])]))
lo, hi = np.percentile(boots, [2.5, 97.5])
print(f'\nLLM timing layer: extra meetings vs model-only = {D:.0f} (95% CI {lo:.0f} to {hi:.0f}) on {int((~c.hon).sum())} re-timed next calls')

# ---------- 2. suppression layer: wrong numbers and do-not-call ----------
a['ep'] = (a.call_attempt_count == 1).groupby(a.fk_glusr_usr_id).cumsum()
wn = e[e.wrong_number & (e.disposition_label != 'Meeting Fixed')][['data_hotlead_disposition_dtlid']]
dn = e[e.dnc & ~e.wrong_number][['data_hotlead_disposition_dtlid']]
key = a.set_index('data_hotlead_disposition_dtlid')
def later_calls(ids, same_episode):
    ev = key.loc[ids.data_hotlead_disposition_dtlid, ['fk_glusr_usr_id', 'call_start_time', 'ep']].reset_index(drop=True)
    ev.columns = ['fk_glusr_usr_id', 'ev_time', 'ev_ep']
    z = a.merge(ev, on='fk_glusr_usr_id')
    z = z[z.call_start_time > z.ev_time]
    if same_episode: z = z[z.ep == z.ev_ep]
    z = z.drop_duplicates('data_hotlead_disposition_dtlid')
    return z[z.call_start_time >= '2026-09-01']
zw = later_calls(wn, True); zd = later_calls(dn, False)
zs = pd.concat([zw, zd]).drop_duplicates('data_hotlead_disposition_dtlid')
print(f'\nSuppressed follow-ups: wrong-number {len(zw)} calls ({int(zw.meeting_fixed.sum())} meetings lost), '
      f'do-not-call {len(zd)} calls ({int(zd.meeting_fixed.sum())} meetings lost); total {len(zs)} calls, {int(zs.meeting_fixed.sum())} meetings')

# ---------- 3. combine with the model's Sep backtest ----------
model_mf, model_cpm = 7913, 15.8         # Max-meetings point (backtest_report sec.1)
model_calls = model_mf * model_cpm
eff_mf, eff_cpm = 7514, 14.1             # Max-efficiency point
for name, mf0, calls0 in [('Max meetings', model_mf, model_calls), ('Max efficiency', eff_mf, eff_mf * eff_cpm)]:
    mf = mf0 + D - zs.meeting_fixed.sum(); calls = calls0 - len(zs)
    mlo = mf0 + lo - zs.meeting_fixed.sum(); mhi = mf0 + hi - zs.meeting_fixed.sum()
    print(f'{name}: model-only {mf0} meetings ({mf0/base_mf-1:+.1%}), cpm {calls0/mf0:.1f} | full system {mf:.0f} meetings '
          f'({mf/base_mf-1:+.1%}, CI {mlo/base_mf-1:+.1%} to {mhi/base_mf-1:+.1%}), cpm {calls/mf:.2f}, per 1000 calls {1000*mf/calls:.1f}')
c[['data_hotlead_disposition_dtlid', 'fk_glusr_usr_id', 'disposition_label', 'cue_strength', 'cue_type', 'target', 'win_start', 'earliest', 'n_time', 'timing', 'hon', 'n_mf']].to_pickle(S + '/cued.pkl')
