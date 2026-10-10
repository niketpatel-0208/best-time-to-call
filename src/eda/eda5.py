import pandas as pd, numpy as np, os, re
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 50); pd.set_option('display.max_rows', 300)
S = os.path.dirname(os.path.abspath(__file__))
a = pd.read_pickle(os.path.join(S,'a2.pkl'))
D='disposition_label'
c = a[a.call_start_time.notna()].copy(); c['hr']=c.call_start_time.dt.hour
c['delay_min'] = (c.call_start_time-c.lead_sent_time).dt.total_seconds()/60
c['vlag_min'] = (c.vendor_response_time-c.call_start_time).dt.total_seconds()/60
c1 = c[c.call_attempt_count==1].copy()
c1['db'] = pd.cut(c1.delay_min, [0,0.5,1,2,5,15,30,60,120,360,1440,1e9])
print("== attempt-1 outcome by trigger delay (speed-to-lead)")
print(c1.groupby('db', observed=True).agg(n=('ans','size'), ans=('ans','mean'), mf=('mf','mean'), sent_hr=('hr','median')).round(4))
c1['sent_hr'] = c1.lead_sent_time.dt.hour
fresh = c1[c1.delay_min<=5]
print("\n== FRESH (<=5 min delay) attempt-1 by hour: removes backlog effect")
print(fresh.groupby('hr').agg(n=('ans','size'), ans=('ans','mean'), mf=('mf','mean'),
     ni=(D, lambda x:(x=='Not Interested').mean()), cl=(D, lambda x:(x=='Call Later / Busy').mean())).round(4))
print("\n== fresh attempt-1 by dow x band (meeting)")
fresh = fresh.assign(band=pd.cut(fresh.hr,[8,10,12,14,16,18,21]), dow=fresh.call_start_time.dt.dayofweek)
print(fresh.pivot_table(index='dow', columns='band', values='mf', aggfunc='mean', observed=True).round(3))
# NA vendor lag distribution -> internal redial?
na = c[c[D]=='Not Answered']
print("\n== NA vendor lag (min) histogram"); print(pd.cut(na.vlag_min,[0,1,2,5,10,14,16,20,30,45,60,120,1e9]).value_counts().sort_index())
# NA tbro - vendor_response
print("== NA tbro - vendor_response (min)"); x=((na.lead_tbro_time-na.vendor_response_time).dt.total_seconds()/60); print(x.describe(percentiles=[.1,.25,.5,.75,.9]).round(1))
print(pd.cut(x,[-1e9,0,14,16,30,59,61,90,119,121,600,1e9]).value_counts().sort_index())
# next lead_sent vs prev tbro
nx = a[~a.new_ep & a.prev_tbro.notna()]
off = (nx.lead_sent_time - nx.prev_tbro).dt.total_seconds()/60
print("== next lead_sent - prev tbro (min)"); print(off.describe(percentiles=[.05,.1,.25,.5,.75,.9]).round(1))
print(pd.cut(off,[-1e9,-30,-16,-14,-10,-5,0,5,15,60,1e9]).value_counts().sort_index())
# weekly label drift
c['wk'] = c.call_start_time.dt.to_period('W')
mv = c[c.lead_bot_version=='main_vani']
print("\n== main_vani weekly answered disposition mix (Aug-Sep)")
t = (pd.crosstab(mv.wk, mv[D], normalize='index')*100).round(1); print(t.tail(12))
# answered-only mix by bot
print("\n== Answered-only disposition mix by bot")
ans = c[c.ans==1]; print((pd.crosstab(ans.lead_bot_version, ans[D], normalize='index')*100).round(1))
# General duration buckets: is 'General' mostly short / IVR / wrong number?
g = c[c[D]=='General (talked)']
print("\n== General (talked) duration buckets by month"); print(pd.crosstab(pd.cut(g.lead_call_duration,[-1,5,10,20,40,80,1e6]), g.call_start_time.dt.to_period('M'), normalize='columns').round(3))
# tbro set for General?
print("General tbro offset (min)"); print(((g.lead_tbro_time-g.call_start_time).dt.total_seconds()/60).describe(percentiles=[.1,.25,.5,.75,.9]))
# meeting slot (tbro for MF) hour dist and lead time
mf = c[c[D]=='Meeting Fixed']
print("\n== Meeting slot hour"); print(mf.lead_tbro_time.dt.hour.value_counts().sort_index())
print("meeting slot lead days"); print(((mf.lead_tbro_time-mf.call_start_time).dt.total_seconds()/86400).describe(percentiles=[.1,.5,.9]))
# repeat meeting fixed for same seller
print("sellers with >1 meeting fixed", (c.groupby('fk_glusr_usr_id').mf.sum()>1).sum())
# Hot-lead bucket x hour of attempt 1 (fresh): bucket-specific timing
top = fresh.redis_bucket.value_counts().head(6).index
print("\n== fresh attempt-1 bucket x band meeting rate"); print(fresh[fresh.redis_bucket.isin(top)].pivot_table(index='redis_bucket', columns='band', values='mf', aggfunc='mean', observed=True).round(3))
print(fresh[fresh.redis_bucket.isin(top)].pivot_table(index='redis_bucket', columns='band', values='mf', aggfunc='size', observed=True))
# Call later: callback at stated time vs sellers' meeting
cl = a[(a.prev_disp=='Call Later / Busy')]
print("\n== after Call Later, hour of next call vs outcome"); cc = cl[cl.call_start_time.notna()]
print(cc.groupby(cc.call_start_time.dt.hour).agg(n=('ans','size'), ans=('ans','mean'), mf=('mf','mean')).round(4))
