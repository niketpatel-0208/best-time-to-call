import sys, pandas as pd
sys.path.insert(0, sys.argv[2])
from extract import extract
R, S = sys.argv[1], sys.argv[2]
sm = pd.read_pickle(S + '/sep_sum.pkl')
a = pd.read_csv(R + '/cowork_upload/data/attempts.csv', parse_dates=['call_start_time'],
                usecols=['data_hotlead_disposition_dtlid', 'disposition_label', 'meeting_fixed', 'call_start_time'])
m = sm.merge(a, on='data_hotlead_disposition_dtlid', how='left')
m = m[m.has_sum].copy()
rows = [extract(t, c) for t, c in zip(m.lead_call_summary, m.call_start_time)]
x = pd.DataFrame(rows, index=m.index)
m = pd.concat([m, x], axis=1)
m.to_pickle(S + '/sep_ext.pkl')
print(len(m))
print(pd.crosstab(m.disposition_label, m.cue_strength, margins=True))
print(m.cue_type.value_counts())
print('wrong_number', m.wrong_number.sum(), pd.crosstab(m.disposition_label, m.wrong_number).iloc[:, 1].to_dict())
print('dnc', m.dnc.sum())
for ct in ['clock', 'relative_minutes', 'relative_hours', 'tomorrow', 'weekday', 'date', 'daypart', 'relative_days', 'event', 'unavailable_period', 'vague']:
    z = m[m.cue_type == ct]
    if len(z):
        r = z.sample(min(2, len(z)), random_state=3)
        for _, q in r.iterrows():
            print(f"[{ct}] {q.call_start_time} -> tgt={q.target} win={q.win_start} earliest={q.earliest} || {q.lead_call_summary[:260]}")
