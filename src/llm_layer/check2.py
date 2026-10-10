import sys, numpy as np, pandas as pd
S = sys.argv[1]
e = pd.read_pickle(S + '/sep_ext.pkl')
lab = pd.read_csv(S + '/val_labels.csv')
v = lab.merge(e[['data_hotlead_disposition_dtlid', 'cue_strength', 'target', 'win_start', 'earliest', 'wrong_number', 'dnc']],
              on='data_hotlead_disposition_dtlid', suffixes=('_h', '_x'))
v['wn_h'] = v.wrong_number_h.astype(str).str.lower().eq('true'); v['dnc_h'] = v.do_not_call.astype(str).str.lower().eq('true')
v['cue_h'] = v.cue_strength_h != 'none'; v['cue_x'] = v.cue_strength_x != 'none'
n = len(v)
print('n', n)
print('has-cue agreement', round((v.cue_h == v.cue_x).mean(), 3), '| strength agreement', round((v.cue_strength_h == v.cue_strength_x).mean(), 3))
tp = (v.cue_h & v.cue_x).sum(); print('cue precision', round(tp / v.cue_x.sum(), 3), 'recall', round(tp / v.cue_h.sum(), 3))
print('wrong-number agreement', round((v.wn_h == v.wrong_number_x).mean(), 3), 'dnc agreement', round((v.dnc_h == v.dnc).mean(), 3))
def xd(r):
    for c in ['target', 'win_start']:
        if pd.notna(r[c]): return pd.Timestamp(r[c]).date()
    return r.earliest if r.earliest is not None and not pd.isna(r.earliest) else None
v['day_x'] = v.apply(xd, axis=1)
both = v[v.target_date.notna() & v.day_x.notna()]
print('same day where both have a date', round((pd.to_datetime(both.target_date).dt.date == both.day_x).mean(), 3), 'n', len(both))
print(pd.crosstab(v.cue_strength_h, v.cue_strength_x))

c = pd.read_pickle(S + '/cued.pkl')
x = c[c.target.notna()].copy()
x['off_h'] = (x.n_time - x.target).dt.total_seconds() / 3600
bins = [-1e9, -24, -6, -2, -0.5, 0.5, 2, 6, 24, 72, 1e9]
labs = ['>1d early', '6-24h early', '2-6h early', '30m-2h early', 'within 30m', '30m-2h late', '2-6h late', '6-24h late', '1-3d late', '>3d late']
x['bin'] = pd.cut(x.off_h, bins, labels=labs)
t = x.groupby('bin', observed=False).agg(n=('n_mf', 'size'), mf=('n_mf', 'mean'))
t['ci'] = 1.96 * np.sqrt(t.mf * (1 - t.mf) / t.n)
print('\nexplicit cues: next-call meeting rate by offset from the seller\'s stated time'); print(t.round(4))
x['after'] = x.off_h >= -0.5
print(x.groupby(x.off_h.ge(-0.5)).agg(n=('n_mf', 'size'), mf=('n_mf', 'mean')).round(4))
y = c[c.target.isna()].copy()
ref = pd.to_datetime(y.win_start.where(y.win_start.notna(), pd.to_datetime(y.earliest)))
y['after'] = y.n_time >= ref
print('\nimplicit cues: next call before vs on/after the hinted time'); print(y.groupby('after').agg(n=('n_mf', 'size'), mf=('n_mf', 'mean')).round(4))
