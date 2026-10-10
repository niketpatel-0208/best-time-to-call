"""What else do the September summaries tell us about the next call? Measured on actual outcomes."""
import sys, re, numpy as np, pandas as pd
R, OLD = sys.argv[1], sys.argv[2]
e = pd.read_pickle(OLD + '/sep_ext.pkl')
a = pd.read_csv(R + '/cowork_upload/data/attempts.csv', parse_dates=['call_start_time'],
                usecols=['data_hotlead_disposition_dtlid', 'fk_glusr_usr_id', 'call_start_time', 'call_attempt_count',
                         'meeting_fixed', 'disposition_label', 'lead_call_status'])
a = a[a.call_start_time.notna()].sort_values(['fk_glusr_usr_id', 'call_start_time']).reset_index(drop=True)
a['ep'] = (a.call_attempt_count == 1).groupby(a.fk_glusr_usr_id).cumsum()

L = e.lead_call_summary.str.lower()
ALREADY = r"already (been )?(visited|met|spoke|spoken|talked|in (touch|contact)|a (paid )?(member|customer|subscriber)|subscribed|taken|purchased|bought|has a (plan|package|membership)|have a (plan|package|membership)|working with|using|registered with)|executive (had |has )?already|already (has|have) (an|a) (executive|account manager)"
HARDNO = r"not interested|no interest|declined|refused|does not want|doesn't want|did not want|didn't want|do not want|don't want|no need|not need|not require|no requirement|uninterested|denied|rejected|hung up"
POS = r"agreed|showed interest|expressed interest|is interested|was interested|were interested|asked (about|for) (the )?(price|pricing|rates?|cost|charges|details|plans?|packages?|fees)|requested (more )?(details|information)|wants? to know|would like|keen|curious|open to"
SOFT = r"\bbut\b|\bhowever\b|later|busy|not (free|available)|another time|call back|callback"

def klass(i):
    r = e.loc[i]; l = L[i]
    if r.wrong_number: return '1 wrong number'
    if r.dnc: return '2 do not call'
    if re.search(ALREADY, l): return '3 already served'
    if r.cue_strength != 'none': return '4 gave time/hint'
    pos = re.search(POS, l); no = re.search(HARDNO, l)
    if no and not pos and not re.search(SOFT, l): return '6 hard no'
    if pos and not no: return '5 positive interest'
    if r.cue_type == 'vague': return '7 busy, no time'
    return '8 other'

e = e[e.disposition_label != 'Meeting Fixed'].copy()
e['klass'] = [klass(i) for i in e.index]
key = a.set_index('data_hotlead_disposition_dtlid')
e = e.join(key[['ep']], on='data_hotlead_disposition_dtlid')

# all later calls in the same episode (what a suppression rule would remove) and the very next call
ev = e[['data_hotlead_disposition_dtlid', 'fk_glusr_usr_id', 'call_start_time', 'ep', 'klass', 'disposition_label']].rename(
    columns={'data_hotlead_disposition_dtlid': 'src', 'call_start_time': 'src_time', 'ep': 'src_ep', 'disposition_label': 'src_label'})
z = a.merge(ev, on='fk_glusr_usr_id')
z = z[(z.call_start_time > z.src_time) & (z.ep == z.src_ep) & (z.call_start_time >= '2026-09-01')]
z = z.sort_values('call_start_time').drop_duplicates('data_hotlead_disposition_dtlid', keep='last')  # credit each later call to the latest summary before it
nxt = z.sort_values('call_start_time').drop_duplicates('src')

print('Answered non-meeting calls with a summary:', len(e))
t = e.groupby('klass').size().rename('calls').to_frame()
t['share'] = t.calls / t.calls.sum()
nn = nxt.groupby('klass').agg(next_calls=('meeting_fixed', 'size'), next_mf=('meeting_fixed', 'mean'),
                              next_ans=('lead_call_status', lambda s: (s == 'Answered').mean()))
ll = z.groupby('klass').agg(later_calls=('meeting_fixed', 'size'), later_meetings=('meeting_fixed', 'sum'))
t = t.join(nn).join(ll)
t['later_cpm'] = t.later_calls / t.later_meetings.replace(0, np.nan)
print(t.round(4).to_string())
print('\nAll Sep calls meeting rate', round(a[a.call_start_time >= '2026-09-01'].meeting_fixed.mean(), 4))
print('\nlabel vs class (row = disposition label of the summarised call)')
print(pd.crosstab(e.disposition_label, e.klass))
e[['data_hotlead_disposition_dtlid', 'klass']].to_pickle(OLD + '/klass.pkl'); z.to_pickle(OLD + '/later.pkl')
