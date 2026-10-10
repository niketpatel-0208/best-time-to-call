import sys, numpy as np, pandas as pd
OLD = sys.argv[1]
z = pd.read_pickle(OLD + '/later.pkl')          # later same-episode calls, credited to the latest summary before them
z['lab'] = z.src_label.map({'Call Later / Busy': 'CL', 'General (talked)': 'General', 'Not Interested': 'NI', 'Not Answered': 'NA'})
print('Later same-episode calls after an answered, non-meeting call:', len(z), 'meetings', int(z.meeting_fixed.sum()),
      'cpm', round(len(z) / z.meeting_fixed.sum(), 1))
t = z.groupby(['lab', 'klass']).agg(calls=('meeting_fixed', 'size'), meetings=('meeting_fixed', 'sum'), rate=('meeting_fixed', 'mean'))
t = t[t.calls >= 40]
print('\nWithin the SAME disposition label (what the model already knows), the summary class still separates outcomes:')
print(t.round(4).to_string())
base_calls, base_mf = 124782, 7383
pols = {
    'A: drop follow-ups after wrong number / do-not-call': ['1 wrong number', '2 do not call'],
    'B: A + hard no': ['1 wrong number', '2 do not call', '6 hard no'],
    'C: B + already served': ['1 wrong number', '2 do not call', '6 hard no', '3 already served'],
}
print()
for k, cls in pols.items():
    d = z[z.klass.isin(cls)]
    c, m = len(d), int(d.meeting_fixed.sum())
    print(f'{k}: calls saved {c} ({c/base_calls:.1%}), meetings lost {m} ({m/base_mf:.2%}), '
          f'removed calls convert at {m/c:.2%} vs 5.92% average; cpm {base_calls/base_mf:.2f} -> {(base_calls-c)/(base_mf-m):.2f}')
