"""
features.py — P1: build the clean analysis base table for Best Time to Call.

Follows the btc-data-rules skill:
- drop undialed rows (empty call_start_time)
- add episodes, 2-hour bands, delay_min, outcome4, bot, label_regime
- add point-in-time seller-history features (rows strictly before the attempt)
- join the seller profile
Writes /scratch/work/base_table.parquet
"""
import pandas as pd
import numpy as np

ATT = '/scratch/input/attempts.csv'
SEL = '/scratch/input/sellers.csv'
OUT = '/scratch/work/base_table.parquet'

DT = ['lead_sent_time', 'call_start_time', 'vendor_response_time', 'lead_tbro_time']

def load():
    a = pd.read_csv(ATT)
    s = pd.read_csv(SEL)
    for c in DT:
        a[c] = pd.to_datetime(a[c], errors='coerce')
    # drop undialed
    a = a[a['call_start_time'].notna()].copy()
    return a, s

def add_core(df):
    # stable time order per seller
    df = df.sort_values(['fk_glusr_usr_id', 'call_start_time',
                         'data_hotlead_disposition_dtlid']).reset_index(drop=True)

    # episode: new episode each time call_attempt_count resets to 1
    df['attempt_reset'] = (df['call_attempt_count'] == 1).astype(int)
    df['episode'] = df.groupby('fk_glusr_usr_id')['attempt_reset'].cumsum()

    # hour and 2-hour band (calls run 09:00-20:59)
    df['hour'] = df['call_start_time'].dt.hour
    # 2-hour bands; hour 21 (a handful of 21:00 calls) folds into the last band
    df['band'] = pd.cut(df['hour'], bins=[0, 10, 12, 14, 16, 18, 23],
                        labels=['9-10', '11-12', '13-14', '15-16', '17-18', '19-20'])

    # delay_min: trigger -> dial
    df['delay_min'] = (df['call_start_time'] - df['lead_sent_time']).dt.total_seconds() / 60.0

    # outcome4 (meeting_fixed is authoritative)
    d = df['disposition_label']
    df['outcome4'] = np.select(
        [df['meeting_fixed'] == 1,
         df['lead_call_status'].eq('NotAnswered'),
         d.eq('Call Later / Busy')],
        ['Meeting Fixed', 'Not Answered', 'Call Later'],
        default='Talked-no-progress')

    # bot
    b = df['lead_bot_version'].fillna('')
    df['bot'] = np.select([b.eq('main_vani'), b.eq('arrowhead')],
                          ['main_vani', 'arrowhead'], default='other')

    # label_regime
    dt = df['call_start_time'].dt.normalize()
    df['label_regime'] = np.select(
        [dt < '2026-08-03', dt <= '2026-08-09', dt <= '2026-09-06'],
        ['pre_3aug', 'aug_3_9', 'aug10_sep6'], default='from_sep7')
    return df

def add_history(df):
    """Point-in-time seller-history features: only rows strictly before the
    current attempt's call_start_time contribute."""
    df['is_answer'] = df['lead_call_status'].eq('Answered').astype(int)
    df['is_meeting'] = (df['meeting_fixed'] == 1).astype(int)
    df['is_ni'] = df['disposition_label'].eq('Not Interested').astype(int)
    df['is_cl'] = df['disposition_label'].eq('Call Later / Busy').astype(int)
    df['is_gen'] = df['disposition_label'].eq('General (talked)').astype(int)

    g = df.groupby('fk_glusr_usr_id', sort=False)

    # prior attempts = row count strictly before current
    df['prior_attempts'] = g.cumcount()

    # prior counts = cumulative sum minus the current row
    for src, dst in [('is_answer', 'prior_answers'), ('is_meeting', 'prior_meetings'),
                     ('is_ni', 'prior_ni'), ('is_cl', 'prior_cl'), ('is_gen', 'prior_gen')]:
        df[dst] = g[src].cumsum() - df[src]

    # smoothed rates, prior = global rate, strength k
    k = 5.0
    p0a = df['is_answer'].mean()
    p0m = df['is_meeting'].mean()
    df['smoothed_answer_rate'] = (df['prior_answers'] + k * p0a) / (df['prior_attempts'] + k)
    df['smoothed_meeting_rate'] = (df['prior_meetings'] + k * p0m) / (df['prior_attempts'] + k)

    # days since the last answered call (strictly before current)
    df['ans_time'] = df['call_start_time'].where(df['is_answer'] == 1)
    df['ans_shift'] = g['ans_time'].shift()
    df['last_ans_time'] = df.groupby('fk_glusr_usr_id', sort=False)['ans_shift'].cummax()
    df['days_since_last_answer'] = (df['call_start_time'] - df['last_ans_time']).dt.total_seconds() / 86400.0

    # within-episode previous-attempt features
    ge = df.groupby(['fk_glusr_usr_id', 'episode'], sort=False)
    df['prev_call_start'] = ge['call_start_time'].shift(1)
    df['prev_outcome'] = ge['outcome4'].shift(1)
    df['gap_min'] = (df['call_start_time'] - df['prev_call_start']).dt.total_seconds() / 60.0
    df['hour_shift'] = df['hour'] - df['prev_call_start'].dt.hour

    # first attempt in episode
    df['is_first_in_episode'] = df['prev_call_start'].isna()
    return df

def main():
    a, s = load()
    print('dialed rows:', len(a))
    a = add_core(a)
    a = add_history(a)

    # join seller profile
    before = len(a)
    a = a.merge(s, on='fk_glusr_usr_id', how='left', validate='many_to_one')
    print('after seller join:', len(a), 'rows (unchanged:', len(a) == before,
          '); true join misses:', (~a['fk_glusr_usr_id'].isin(s['fk_glusr_usr_id'])).sum())

    # ---- verification ----
    print('\n== VERIFY ==')
    print('dialed rows:', len(a))
    print(f"connect rate: {(a['is_answer'].mean()*100):.1f}%")
    print(f"meeting rate (dialed): {(a['is_meeting'].mean()*100):.1f}%")

    sep = a[a['call_start_time'].dt.to_period('M').astype(str) == '2026-09']
    cpm_sep = len(sep) / sep['is_meeting'].sum()
    print(f"Sep rows: {len(sep)}; Sep meetings: {int(sep['is_meeting'].sum())}; calls/meeting Sep: {cpm_sep:.1f}")

    m1 = a[a['call_attempt_count'] == 1]['is_meeting'].sum()
    tot = a['is_meeting'].sum()
    print(f"meetings on attempt 1: {int(m1)}/{int(tot)} = {m1/tot*100:.0f}%")

    print('\nepisodes:', a['episode'].nunique())
    print('bands:', sorted(a['band'].dropna().unique()))
    print('outcome4 mix:\n', a['outcome4'].value_counts())
    print('bot mix:\n', a['bot'].value_counts())
    print('label_regime mix:\n', a['label_regime'].value_counts())

    a.to_parquet(OUT, index=False)
    print('\nwrote', OUT, a.shape)

if __name__ == '__main__':
    main()
