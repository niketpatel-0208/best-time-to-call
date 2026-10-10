import sys, numpy as np, pandas as pd
S, OUT = sys.argv[1], sys.argv[2]
c = pd.read_pickle(S + '/cued.pkl')
ref = c.target.where(c.target.notna(), c.win_start)
ref = pd.to_datetime(ref.where(ref.notna(), pd.to_datetime(c.earliest)))
c['before'] = c.n_time < ref - pd.Timedelta(minutes=30) * c.target.notna()
c['grp'] = c.cue_strength
L = 0.174

def delta(df):
    d = 0.0
    for _, g in df.groupby('grp'):
        b, a = g[g.before], g[~g.before]
        d += len(b) * (a.n_mf.mean() - b.n_mf.mean() * (1 + L))
    return d

D = delta(c)
rng = np.random.default_rng(11)
sid = c.fk_glusr_usr_id.values; u, inv = np.unique(sid, return_inverse=True)
rows = [np.where(inv == i)[0] for i in range(len(u))]
bs = [delta(c.iloc[np.concatenate([rows[i] for i in rng.integers(0, len(u), len(u))])]) for _ in range(400)]
lo, hi = np.percentile(bs, [2.5, 97.5])
t = c.groupby(['grp', 'before']).agg(n=('n_mf', 'size'), mf=('n_mf', 'mean')).round(4)
print(t); print(f'delta {D:.1f} CI {lo:.1f} {hi:.1f}; before-calls {int(c.before.sum())}')
base_mf = 7383; sup_calls, sup_mf = 631, 7
res = []
for name, mf0, cpm0 in [('Max meetings', 7913, 15.8), ('Max efficiency', 7514, 14.1)]:
    calls0 = mf0 * cpm0
    for lab, mf, calls in [('model only', mf0, calls0), ('model + LLM layer', mf0 + D - sup_mf, calls0 - sup_calls)]:
        res.append(dict(setting=name, system=lab, meetings=round(mf), lift=mf / base_mf - 1, calls=round(calls),
                        cpm=calls / mf, per_1000=1000 * mf / calls,
                        lift_lo=(mf0 + lo - sup_mf) / base_mf - 1 if lab != 'model only' else None,
                        lift_hi=(mf0 + hi - sup_mf) / base_mf - 1 if lab != 'model only' else None))
r = pd.DataFrame(res); print(r.round(4).to_string())
r.to_csv(S + '/final_table.csv', index=False)
e = pd.read_pickle(S + '/sep_ext.pkl')
keep = ['data_hotlead_disposition_dtlid', 'fk_glusr_usr_id', 'disposition_label', 'cue_strength', 'cue_type', 'target',
        'win_start', 'win_end', 'earliest', 'wrong_number', 'dnc']
e[keep].rename(columns={'fk_glusr_usr_id': 'seller_id', 'target': 'callback_at'}).to_csv(OUT + '/sep_callback_extractions.csv', index=False)
print('saved')
