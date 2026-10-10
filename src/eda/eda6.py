import pandas as pd, numpy as np, os, re
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 50); pd.set_option('display.max_rows', 300); pd.set_option('display.max_colwidth', 300)
S = os.path.dirname(os.path.abspath(__file__))
D = os.environ.get("BTC_DATA_DIR", "data")
cs = pd.read_csv(os.path.join(D,"Call Summary of CAll Later calls.xlsx - result.csv"))
a = pd.read_pickle(os.path.join(S,'a2.pkl'))
print(cs.shape, cs.data_hotlead_disposition_dtlid.nunique(), cs.lead_call_summary.isna().sum())
m = cs.merge(a, on='data_hotlead_disposition_dtlid', how='left')
print("matched", m.fk_glusr_usr_id.notna().mean()); print(m.disposition_label.value_counts(dropna=False)); print(m.lead_bot_version.value_counts()); print(m.call_start_time.dt.to_period('M').value_counts().sort_index())
L = m.lead_call_summary.fillna('').str.lower()
pats = {
 'minutes (after N min)': r'\b(\d+|ten|five|fifteen|twenty|thirty|few)\s*(-|to)?\s*\d*\s*min',
 'hour(s) / in an hour': r'\b(an|one|two|\d+|half an|few|couple of)\s*(-\s*\d+\s*)?hours?\b',
 'morning': r'morning', 'afternoon': r'afternoon', 'evening': r'evening', 'night': r'\bnight\b',
 'lunch': r'lunch', 'tomorrow': r'tomorrow', 'today/later today': r'\btoday\b|later in the day',
 'weekday named': r'monday|tuesday|wednesday|thursday|friday|saturday|sunday',
 'next week': r'next week', 'days (N days)': r'\b(\d+|few|couple of|two|three|four|five|ten)\s*(-\s*\d+\s*)?days?\b',
 'month/date': r'next month|\b\d{1,2}(st|nd|rd|th)\b|end of the month',
 'clock time (e.g. 4 pm)': r'\b\d{1,2}(:\d\d)?\s*(am|pm|o.clock|baje)\b',
 'festival/season': r'diwali|navratri|festival|season|holi|eid|wedding|marriage',
 'driving/travel': r'driving|travel|outside|out of station|on the road|journey|train',
 'meeting/busy w customer': r'in a meeting|customer|client',
 'unwell/hospital': r'unwell|hospital|sick|ill\b|health',
 'will call back himself': r'(he|she|they|lead) (would|will) call back|call (us|the agent) back',
 'not interested mention': r'not interested|disinterest',
 'when free (no time)': r'when (he|she|they) (is|are)? ?free|when free',
}
rows=[]
for k,p in pats.items():
    hit = L.str.contains(p, regex=True)
    rows.append((k, hit.sum(), round(hit.mean()*100,1)))
print(pd.DataFrame(rows, columns=['pattern','n','%']).sort_values('n',ascending=False).to_string(index=False))
anyt = pd.Series(False,index=L.index)
for k in ['minutes (after N min)','hour(s) / in an hour','morning','afternoon','evening','night','lunch','tomorrow','today/later today','weekday named','next week','days (N days)','month/date','clock time (e.g. 4 pm)']:
    anyt |= L.str.contains(pats[k], regex=True)
print("summaries with any explicit time cue: %.1f%%" % (anyt.mean()*100))
m['off_h'] = (m.lead_tbro_time - m.call_start_time).dt.total_seconds()/3600
print("tbro filled:", m.lead_tbro_time.notna().mean())
for k in ['minutes (after N min)','hour(s) / in an hour','evening','morning','tomorrow','weekday named','days (N days)','when free (no time)']:
    h = L.str.contains(pats[k], regex=True)
    print(f"{k:28s} n={h.sum():4d} tbro offset hrs median={m.loc[h,'off_h'].median():.2f}  p25={m.loc[h,'off_h'].quantile(.25):.2f} p75={m.loc[h,'off_h'].quantile(.75):.2f}  tbro hour mode={m.loc[h,'lead_tbro_time'].dt.hour.mode().tolist()}")
# next outcome after these call-later
a2 = a[['fk_glusr_usr_id','prev_disp','t','ans','mf','disposition_label']]
nxt = a.sort_values(['fk_glusr_usr_id','t']).groupby('fk_glusr_usr_id').shift(-1)[['t','disposition_label','mf','ans']]
nxt.columns = ['n_t','n_disp','n_mf','n_ans']
aa = pd.concat([a[['data_hotlead_disposition_dtlid']], nxt], axis=1)
m = m.merge(aa, on='data_hotlead_disposition_dtlid', how='left')
m['gap_h'] = (m.n_t - m.call_start_time).dt.total_seconds()/3600
m['dev_h'] = (m.n_t - m.lead_tbro_time).dt.total_seconds()/3600
print("\nnext outcome after these CL calls"); print(m.n_disp.value_counts(normalize=True, dropna=False).round(3))
for k in ['minutes (after N min)','hour(s) / in an hour','evening','tomorrow','when free (no time)']:
    h = L.str.contains(pats[k], regex=True)
    print(f"{k:28s} next gap h median={m.loc[h,'gap_h'].median():.2f} next MF={m.loc[h,'n_mf'].mean():.3f} next ans={m.loc[h,'n_ans'].mean():.3f}")
print("\nsample summaries with minutes cue:"); print(m.loc[L.str.contains(pats['minutes (after N min)']), ['lead_call_summary','call_start_time','lead_tbro_time','n_t']].head(6).to_string())
print("\nsample with evening:"); print(m.loc[L.str.contains('evening'), ['call_start_time','lead_tbro_time','n_t','n_disp']].head(8).to_string())
