import pandas as pd, numpy as np, os
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 50); pd.set_option('display.max_rows', 300)
S = os.path.dirname(os.path.abspath(__file__))
a = pd.read_pickle(os.path.join(S,'a.pkl'))
D = 'disposition_label'
a['t'] = a.call_start_time.fillna(a.lead_sent_time)
a = a.sort_values(['fk_glusr_usr_id','t','data_hotlead_disposition_dtlid']).reset_index(drop=True)
g = a.groupby('fk_glusr_usr_id')
a['prev_disp'] = g[D].shift(1)
a['prev_t'] = g['t'].shift(1)
a['prev_tbro'] = g['lead_tbro_time'].shift(1)
a['prev_cnt'] = g['call_attempt_count'].shift(1)
a['gap_min'] = (a.t - a.prev_t).dt.total_seconds()/60
# episode: new when attempt count==1 or count not increasing or gap>3 days
a['new_ep'] = (a.prev_t.isna()) | (a.call_attempt_count==1) | (a.call_attempt_count<=a.prev_cnt) | (a.gap_min>3*1440)
a['ep'] = a.new_ep.cumsum()
a['ans'] = (a.lead_call_status=='Answered').astype(int)
a['mf'] = a.meeting_fixed
a.to_pickle(os.path.join(S,'a2.pkl'))
print("episodes", a.ep.nunique(), "sellers", a.fk_glusr_usr_id.nunique())
ep = a.groupby('ep').agg(seller=('fk_glusr_usr_id','first'), n=('ans','size'), mf=('mf','max'), anyans=('ans','max'),
                         start=('t','min'), end=('t','max'), last=(D,'last'), bucket=('redis_bucket','first'))
print("attempts/episode", ep.n.describe()); print("episode meeting rate", ep.mf.mean(), "episode connect rate", ep.anyans.mean())
print("calls per meeting (overall)", len(a)/a.mf.sum())
print("episode duration hrs", ((ep.end-ep.start).dt.total_seconds()/3600).describe(percentiles=[.5,.75,.9,.95,.99]))
print("last disposition of episode"); print(ep['last'].value_counts(normalize=True).round(3))
print("episodes per seller"); print(ep.groupby('seller').size().value_counts().sort_index().head(15))
# disposition transition within episodes
w = a[~a.new_ep]
print("\n== within-episode transition prev->next (row %)"); print((pd.crosstab(w.prev_disp, w[D], normalize='index')*100).round(1).assign(n=w.prev_disp.value_counts()))
x = a[a.new_ep & a.prev_disp.notna()]
print("\n== across-episode: last disposition of previous contact -> first outcome of new episode"); print((pd.crosstab(x.prev_disp, x[D], normalize='index')*100).round(1).assign(n=x.prev_disp.value_counts()))
# retry gap after Not Answered within episode
na = w[w.prev_disp=='Not Answered'].copy()
print("\n== gap after Not Answered (min)"); print(na.gap_min.describe(percentiles=[.05,.1,.25,.5,.75,.9,.95]))
bins = [0,10,14,17,20,30,45,60,90,120,180,240,360,720,1440,2880,1e9]
na['gb'] = pd.cut(na.gap_min, bins)
t = na.groupby('gb', observed=True).agg(n=('ans','size'), ans=('ans','mean'), mf=('mf','mean'))
t['mf_given_ans'] = na[na.ans==1].groupby('gb', observed=True).mf.mean(); print(t.round(4))
# stratify by previous attempt count to remove attempt confounding
print("\n== gap after NA: by prev attempt count (ans rate)")
na['pc'] = na.prev_cnt.clip(upper=5)
print(na.pivot_table(index='gb', columns='pc', values='ans', aggfunc='mean', observed=True).round(3))
print(na.pivot_table(index='gb', columns='pc', values='mf', aggfunc='mean', observed=True).round(4))
print(na.pivot_table(index='gb', columns='pc', values='ans', aggfunc='size', observed=True))
# gap after call later
cl = w[w.prev_disp=='Call Later / Busy'].copy()
cl['gb'] = pd.cut(cl.gap_min, bins)
print("\n== gap after Call Later"); t = cl.groupby('gb', observed=True).agg(n=('ans','size'), ans=('ans','mean'), mf=('mf','mean')); print(t.round(4))
# tbro adherence
for pdsp in ['Call Later / Busy','Not Answered','General (talked)','Not Interested']:
    y = a[(a.prev_disp==pdsp) & a.prev_tbro.notna()].copy()
    y['dev_min'] = (y.t - y.prev_tbro).dt.total_seconds()/60
    y['db'] = pd.cut(y.dev_min, [-1e9,-60,-15,-5,5,15,60,240,1440,1e9])
    print(f"\n== next call vs promised tbro after {pdsp}: (deviation min)"); print(y.dev_min.describe(percentiles=[.1,.25,.5,.75,.9]))
    t = y.groupby('db', observed=True).agg(n=('ans','size'), ans=('ans','mean'), mf=('mf','mean')); print(t.round(4))
# tbro offset from call time per disposition
a['tbro_off_min'] = (a.lead_tbro_time - a.t).dt.total_seconds()/60
print("\n== tbro offset from this call (min) by disposition"); print(a.groupby(D).tbro_off_min.describe(percentiles=[.1,.25,.5,.75,.9]).round(1))
print("\n== tbro hour for Call Later"); print(a.loc[a[D]=='Call Later / Busy','lead_tbro_time'].dt.hour.value_counts().sort_index())
print("tbro minute==0 share for call later", (a.loc[a[D]=='Call Later / Busy','lead_tbro_time'].dt.minute==0).mean())
print("\n== what happens after Not Interested: share of sellers called again within ep / later")
ni = a[a.prev_disp=='Not Interested']; print(ni[D].value_counts(normalize=True).round(3), len(ni))
