import pandas as pd, numpy as np, os
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 50); pd.set_option('display.max_rows', 300)
S = os.path.dirname(os.path.abspath(__file__))
a = pd.read_pickle(os.path.join(S,'a2.pkl'))
D='disposition_label'
a = a.sort_values(['fk_glusr_usr_id','t']).reset_index(drop=True)
g = a.groupby('fk_glusr_usr_id')
a['prev_hr'] = g.t.shift(1).dt.hour
a['hr'] = a.t.dt.hour
# (a) NA -> next-day retry: same hour vs shifted
x = a[(a.prev_disp=='Not Answered') & (a.gap_min.between(18*60, 30*60))].copy()
x['shift'] = (x.hr - x.prev_hr)
x['sb'] = pd.cut(x['shift'], [-12,-4,-2,-1,0,1,3,12], labels=['<=-4','-3..-2','-1','0 (same hr)','+1','+2..+3','>=+4'])
print("== NA -> next-day retry: hour shift vs outcome"); print(x.groupby('sb', observed=True).agg(n=('ans','size'), ans=('ans','mean'), mf=('mf','mean')).round(4))
print("NA->next-day by retry hour"); print(x.groupby('hr').agg(n=('ans','size'), ans=('ans','mean'), mf=('mf','mean')).round(4))
# same-day NA retry by retry hour (attempt 2 only)
y = a[(a.prev_disp=='Not Answered') & (a.gap_min<=240) & (a.call_attempt_count==2)]
print("\n== same-day retry #2 after NA by hour"); print(y.groupby('hr').agg(n=('ans','size'), ans=('ans','mean'), mf=('mf','mean')).round(4))
# (b) re-engagement after terminal dispositions: days until next contact vs outcome
for d in ['Not Interested','General (talked)','Call Later / Busy','Meeting Fixed']:
    z = a[a.prev_disp==d].copy(); z['gd'] = pd.cut(z.gap_min/1440, [0,0.04,0.25,1,2,3,7,14,30,200])
    print(f"\n== after {d}: gap (days) to next call"); print(z.groupby('gd', observed=True).agg(n=('ans','size'), ans=('ans','mean'), mf=('mf','mean'), ni=(D, lambda s:(s=='Not Interested').mean())).round(4))
# (c) episode-level: number of attempts needed to fix meeting
ep = a.groupby('ep').agg(n=('ans','size'), mf=('mf','max'))
w = a[a.mf==1].groupby('ep').call_attempt_count.min()
print("\n== attempt # at which meeting fixed"); print(w.clip(upper=8).value_counts(normalize=True).sort_index().round(3))
# cumulative meetings vs attempts cap: what % of meetings would be lost if capped at k attempts
for k in [1,2,3,4,5,6]:
    print(f"cap {k}: meetings kept {(w<=k).mean():.3f}, calls kept {(a.call_attempt_count<=k).mean():.3f}")
# propensity overlap: hour distribution of attempt-1 by bucket
c1 = a[(a.call_attempt_count==1)&a.call_start_time.notna()]
print("\n== attempt-1 hour share by bucket (positivity)"); top=c1.redis_bucket.value_counts().head(6).index
print((pd.crosstab(c1[c1.redis_bucket.isin(top)].redis_bucket, c1.hr, normalize='index')*100).round(1))
