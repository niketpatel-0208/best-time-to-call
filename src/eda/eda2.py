import pandas as pd, numpy as np, os
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 50); pd.set_option('display.max_rows', 300)
S = os.path.dirname(os.path.abspath(__file__))
a = pd.read_pickle(os.path.join(S,'a.pkl'))
a['month'] = a.lead_sent_time.dt.to_period('M')
D = 'disposition_label'
print("== disposition by bot x month (main_vani)")
for b in ['main_vani','arrowhead']:
    x = a[a.lead_bot_version==b]
    print(b); print((pd.crosstab(x.month, x[D], normalize='index')*100).round(1))
c = a[a.call_start_time.notna()].copy()
c['hr'] = c.call_start_time.dt.hour
c['dow'] = c.call_start_time.dt.dayofweek
c['ans'] = (c.lead_call_status=='Answered').astype(int)
c['mf'] = c.meeting_fixed
def rates(g):
    t = c.groupby(g).agg(n=('ans','size'), ans=('ans','mean'), mf=('mf','mean'))
    t['mf_given_ans'] = c[c.ans==1].groupby(g).mf.mean()
    for d in ['Not Interested','Call Later / Busy','General (talked)']:
        t[d[:6]] = (c[D]==d).groupby(c[g] if isinstance(g,str) else [c[k] for k in g]).mean()
    return t.round(4)
print("\n== by hour"); print(rates('hr'))
print("\n== by dow"); print(rates('dow'))
print("\n== hour by attempt 1 only")
c1 = c[c.call_attempt_count==1]
print(c1.groupby('hr').agg(n=('ans','size'), ans=('ans','mean'), mf=('mf','mean')).round(4))
print("\n== hour x month answer rate"); print(c.pivot_table(index='hr', columns='month', values='ans', aggfunc='mean').round(3))
print("\n== hour x month meeting rate"); print(c.pivot_table(index='hr', columns='month', values='mf', aggfunc='mean').round(3))
print("\n== minute-of-hour hist (first 10 min share)"); print(c.call_start_time.dt.minute.value_counts().sort_index().head(15))
# dialing delay
c['delay_s'] = (c.call_start_time - c.lead_sent_time).dt.total_seconds()
print("\n== trigger delay (s) attempt1"); print(c.loc[c.call_attempt_count==1,'delay_s'].describe(percentiles=[.1,.25,.5,.75,.9,.95,.99]))
print("== delay all"); print(c.delay_s.describe(percentiles=[.1,.25,.5,.75,.9,.95,.99]))
c['lag_s'] = (c.vendor_response_time - c.call_start_time).dt.total_seconds()
print("== vendor response lag (s) by disposition"); print(c.groupby(D).lag_s.describe(percentiles=[.5,.9]))
# lead_sent hour distribution
print("\n== lead_sent hour dist"); print(a.lead_sent_time.dt.hour.value_counts().sort_index())
# daily volume
print("\n== daily volume stats"); dv = c.groupby(c.call_start_time.dt.date).size(); print(dv.describe()); print(dv.tail(20))
c.to_pickle(os.path.join(S,'c.pkl'))
