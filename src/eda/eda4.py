import pandas as pd, numpy as np, os, re
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 50); pd.set_option('display.max_rows', 300)
S = os.path.dirname(os.path.abspath(__file__))
a = pd.read_pickle(os.path.join(S,'a2.pkl')); s = pd.read_pickle(os.path.join(S,'s.pkl'))
D='disposition_label'
print(s.isna().mean().round(3))
for col in ['business_type','annual_turnover','nature_of_business','top_category_group','seller_state']:
    print(f"\n== {col}: top values"); print(s[col].value_counts(dropna=False).head(12))
print("\nnum_categories", s.num_categories.describe()); print("gst year", s.gst_registration_year.value_counts().sort_index().tail(12))
print("cities", s.seller_city.nunique(), "states", s.seller_state.nunique(), "cat groups", s.top_category_group.nunique(), "mcat1", s.top_category_1.nunique())
m = a.merge(s, on='fk_glusr_usr_id', how='left')
c = m[m.call_start_time.notna()].copy(); c['hr']=c.call_start_time.dt.hour
def seg(col, top=12):
    t = c.groupby(col).agg(n=('ans','size'), sellers=('fk_glusr_usr_id','nunique'), ans=('ans','mean'), mf=('mf','mean'))
    t['mf_ans'] = c[c.ans==1].groupby(col).mf.mean()
    return t.sort_values('n', ascending=False).head(top).round(4)
for col in ['business_type','annual_turnover','nature_of_business','top_category_group','seller_state']:
    print(f"\n== outcome by {col}"); print(seg(col))
c['gst_b'] = pd.cut(c.gst_registration_year, [0,2017,2019,2021,2023,2026])
print(seg('gst_b'))
c['ncat_b'] = pd.cut(c.num_categories, [0,1,3,5,10,20,50,1e5]); print(seg('ncat_b'))
# heterogeneity of hour effect by segment: meeting rate per hour band by turnover & nature
c['band'] = pd.cut(c.hr, [8,10,12,14,16,18,21], labels=['9-10','11-12','13-14','15-16','17-18','19-21'])
c1 = c[c.call_attempt_count==1]
for col in ['annual_turnover','nature_of_business','business_type']:
    top = c1[col].value_counts().head(6).index
    print(f"\n== attempt-1 answer rate: {col} x band"); print(c1[c1[col].isin(top)].pivot_table(index=col, columns='band', values='ans', aggfunc='mean', observed=True).round(3))
    print(f"== attempt-1 meeting rate: {col} x band"); print(c1[c1[col].isin(top)].pivot_table(index=col, columns='band', values='mf', aggfunc='mean', observed=True).round(3))
top = c1.top_category_group.value_counts().head(10).index
print("\n== attempt-1 answer: category group x band"); print(c1[c1.top_category_group.isin(top)].pivot_table(index='top_category_group', columns='band', values='ans', aggfunc='mean', observed=True).round(3))
print("\n== attempt-1 meeting: category group x band"); print(c1[c1.top_category_group.isin(top)].pivot_table(index='top_category_group', columns='band', values='mf', aggfunc='mean', observed=True).round(3))
top = c1.seller_state.value_counts().head(10).index
print("\n== attempt-1 answer: state x band"); print(c1[c1.seller_state.isin(top)].pivot_table(index='seller_state', columns='band', values='ans', aggfunc='mean', observed=True).round(3))
# seller history depth & overlap across months
a['month'] = a.t.dt.to_period('M')
sm = a.groupby('fk_glusr_usr_id').month.agg(['min','nunique'])
print("\n== seller first-month"); print(sm['min'].value_counts().sort_index()); print("months active"); print(sm['nunique'].value_counts().sort_index())
for cut in ['2026-08-01','2026-09-01']:
    tr = set(a.loc[a.t<cut,'fk_glusr_usr_id']); te = a[a.t>=cut]
    print(f"cut {cut}: test sellers {te.fk_glusr_usr_id.nunique()}, seen before {te.fk_glusr_usr_id.isin(tr).mean():.3f} of attempts, {np.mean([x in tr for x in te.fk_glusr_usr_id.unique()]):.3f} of sellers")
    tra = set(a.loc[(a.t<cut)&(a.ans==1),'fk_glusr_usr_id'])
    print(f"   test sellers with >=1 prior ANSWERED call: {np.mean([x in tra for x in te.fk_glusr_usr_id.unique()]):.3f}")
# seller-level hour consistency: for sellers with >=2 answered calls, do answers cluster?
ansd = c[c.ans==1].groupby('fk_glusr_usr_id').hr.agg(['size','std'])
print("\nsellers with >=2 answered:", (ansd['size']>=2).sum(), " >=3:", (ansd['size']>=3).sum(), " median hr std:", ansd.loc[ansd['size']>=2,'std'].median())
# per-seller predictive check: seller's prior answer rate (before Aug) vs answer in Aug-Sep
pre = c[c.call_start_time<'2026-08-01'].groupby('fk_glusr_usr_id').agg(pn=('ans','size'), pa=('ans','mean'), pm=('mf','max'))
post = c[c.call_start_time>='2026-08-01'].merge(pre, left_on='fk_glusr_usr_id', right_index=True, how='left')
post['pab'] = pd.cut(post.pa, [-0.01,0,0.25,0.5,0.75,1.0]).astype(str).where(post.pa.notna(), 'new')
print("\n== Aug-Sep outcome by seller's prior (Apr-Jul) answer rate"); print(post.groupby('pab').agg(n=('ans','size'), ans=('ans','mean'), mf=('mf','mean')).round(4))
print("\n== Aug-Sep outcome by prior meeting fixed"); print(post.groupby(post.pm.fillna(-1)).agg(n=('ans','size'), ans=('ans','mean'), mf=('mf','mean')).round(4))
# seller prior answered hour band vs current band match
pa_hr = c[(c.call_start_time<'2026-08-01')&(c.ans==1)].groupby('fk_glusr_usr_id').hr.median().rename('prior_ans_hr')
pna_hr = c[(c.call_start_time<'2026-08-01')].groupby('fk_glusr_usr_id').hr.median().rename('prior_any_hr')
post = post.merge(pa_hr, left_on='fk_glusr_usr_id', right_index=True, how='left').merge(pna_hr, left_on='fk_glusr_usr_id', right_index=True, how='left')
post['hr'] = post.call_start_time.dt.hour
post['dist_ans'] = (post.hr - post.prior_ans_hr).abs()
post['db'] = pd.cut(post.dist_ans, [-0.1,1,2,4,12])
print("\n== Aug-Sep answer rate by |hour - seller's prior median ANSWERED hour|"); print(post.groupby('db', observed=True).agg(n=('ans','size'), ans=('ans','mean'), mf=('mf','mean')).round(4))
post['db2'] = pd.cut((post.hr - post.prior_any_hr).abs(), [-0.1,1,2,4,12])
print("== control: by |hour - seller's prior median ANY-call hour|"); print(post.groupby('db2', observed=True).agg(n=('ans','size'), ans=('ans','mean'), mf=('mf','mean')).round(4))
