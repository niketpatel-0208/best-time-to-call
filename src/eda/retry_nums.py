import numpy as np, pandas as pd
a = pd.read_csv('data/attempts.csv', parse_dates=['call_start_time'],
                usecols=['fk_glusr_usr_id', 'call_start_time', 'disposition_label', 'meeting_fixed', 'lead_call_status', 'call_attempt_count'])
a = a[a.call_start_time.notna()].sort_values(['fk_glusr_usr_id', 'call_start_time']).reset_index(drop=True)
g = a.groupby('fk_glusr_usr_id')
a['n_t'] = g.call_start_time.shift(-1); a['n_mf'] = g.meeting_fixed.shift(-1)
a['n_ans'] = (g.lead_call_status.shift(-1) == 'Answered').astype(float)
a = a[a.n_t.notna()].copy()
a['gap_h'] = (a.n_t - a.call_start_time).dt.total_seconds() / 3600
band = lambda h: (h - 9) // 2
a['b0'] = band(a.call_start_time.dt.hour); a['b1'] = band(a.n_t.dt.hour)
def tab(d, bins, labels, col='gap_h'):
    d = d.assign(bin=pd.cut(d[col], bins, labels=labels))
    t = d.groupby('bin', observed=True).agg(n=('n_mf', 'size'), answer=('n_ans', 'mean'), meeting=('n_mf', 'mean'))
    return t.round(3)
na = a[a.disposition_label == 'Not Answered']
print('AFTER NOT ANSWERED: gap to next call'); print(tab(na, [0, 0.75, 1.5, 4, 8, 24, 48, 1e9], ['<45m', '45m-1.5h', '1.5-4h', '4-8h', '8-24h', '1-2d', '>2d']).T.to_string())
nd = na[(na.gap_h >= 12) & (na.gap_h <= 36)].copy()
nd['shift'] = nd.n_t.dt.hour - nd.call_start_time.dt.hour
print('\nNEXT-DAY retry after NA: hour shift vs failed hour')
print(tab(nd, [-24, -3.5, -0.5, 0.5, 1.5, 3.5, 24], ['earlier 4h+', 'earlier 1-3h', 'same hour', '+1h', '+2-3h', '+4h or more'], 'shift').T.to_string())
sd = na[na.gap_h < 12].copy(); sd['same_band'] = sd.b0 == sd.b1
print('\nSAME-DAY retry after NA: same 2-hour band vs a different band'); print(sd.groupby('same_band').agg(n=('n_mf', 'size'), answer=('n_ans', 'mean'), meeting=('n_mf', 'mean')).round(3).to_string())
ev = na.copy(); ev['nh'] = ev.n_t.dt.hour
ev = ev[(ev.gap_h >= 2)]
print('\nRETRY HOUR after NA (gap >= 2h)'); print(ev.groupby(pd.cut(ev.nh, [8, 10, 12, 14, 16, 18, 21], labels=['9-11', '11-13', '13-15', '15-17', '17-19', '19-21']), observed=True).agg(n=('n_mf', 'size'), answer=('n_ans', 'mean'), meeting=('n_mf', 'mean')).round(3).T.to_string())
for lab, bins, labels in [('Call Later / Busy', [0, 1, 6, 24, 48, 168, 1e9], ['<1h', '1-6h', '6h-1d', '1-2d', '2-7d', '>7d']),
                          ('General (talked)', [0, 24, 72, 168, 336, 1e9], ['<1d', '1-3d', '3-7d', '7-14d', '>14d']),
                          ('Not Interested', [0, 24, 72, 168, 336, 720, 1e9], ['<1d', '1-3d', '3-7d', '7-14d', '14-30d', '>30d'])]:
    print(f'\nAFTER {lab}'); print(tab(a[a.disposition_label == lab], bins, labels).T.to_string())
c = a.groupby(a.call_attempt_count.clip(upper=8)).meeting_fixed.sum(); print('\nmeetings by attempt no.', (c / c.sum()).round(3).to_dict())
