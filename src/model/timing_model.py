"""
timing_model.py — P3: C1 timing model.

Hierarchical Beta-Binomial over the 2-hour bands, separately for meeting and answer.
Structure: global band curve x segment level x seller deviation (band-specific),
all empirical-Bayes shrunk. Segments: state, turnover, GST-year bucket, category
group, redis_bucket. Fit on Apr-Jul fresh first attempts; shrinkage tuned on Aug.
Writes seller_windows.csv.
"""
import os, resource
try:
    resource.setrlimit(resource.RLIMIT_AS, (resource.RLIM_INFINITY, resource.RLIM_INFINITY))
except Exception:
    pass
for _v in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:
    os.environ.setdefault(_v, '1')

import numpy as np
import pandas as pd
import json

BASE = '/scratch/work/base_table.parquet'
SEL = '/scratch/input/sellers.csv'
BANDS = ['9-10', '11-12', '13-14', '15-16', '17-18', '19-20']
BAND_START = {'9-10': 9, '11-12': 11, '13-14': 13, '15-16': 15, '17-18': 17, '19-20': 19}
DIMS = ['state', 'turnover', 'gst_bucket', 'cat', 'bucket']
COLS = ['fk_glusr_usr_id', 'band', 'call_attempt_count', 'delay_min', 'meeting_fixed',
        'lead_call_status', 'disposition_label', 'redis_bucket', 'call_start_time',
        'seller_state', 'annual_turnover', 'gst_registration_year', 'top_category_group']

def logit(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))

def sig(x):
    return 1.0 / (1.0 + np.exp(-x))

def load():
    b = pd.read_parquet(BASE, columns=COLS, use_threads=False)
    b['band'] = b['band'].astype(str)
    b['month'] = b['call_start_time'].dt.to_period('M').astype(str)
    b['fresh'] = (b['call_attempt_count'] == 1) & (b['delay_min'] <= 5)
    b['gst_bucket'] = pd.cut(b['gst_registration_year'], [0, 2015, 2019, 2023, 2027],
                             labels=['<=2015', '2016-2019', '2020-2023', '2024+']).astype(str)
    b['state'] = b['seller_state'].fillna('NA').astype(str)
    b['turnover'] = b['annual_turnover'].fillna('NA').astype(str)
    b['cat'] = b['top_category_group'].fillna('NA').astype(str)
    b['bucket'] = b['redis_bucket'].fillna('NA').astype(str)
    b['y_meet'] = (b['meeting_fixed'] == 1).astype(int)
    b['y_ans'] = (b['lead_call_status'] == 'Answered').astype(int)
    return b

def precompute(tr, ycol):
    y = tr[ycol].values.astype(float)
    g = y.mean()
    theta = tr.groupby('band')[ycol].mean().reindex(BANDS).fillna(g).to_dict()
    dim_stats = {}
    for d in DIMS:
        grp = tr.groupby(d)[ycol].agg(['mean', 'size'])
        dim_stats[d] = {'raw': {v: float(logit(r['mean']) - logit(g)) for v, r in grp.iterrows()},
                        'n': {v: float(r['size']) for v, r in grp.iterrows()}}
    gsb = tr.groupby(['fk_glusr_usr_id', 'band'])[ycol].agg(n='size', ys='sum').reset_index()
    return {'g': g, 'theta': theta, 'dim_stats': dim_stats, 'gsb': gsb,
            'seller': tr['fk_glusr_usr_id'].values, 'ny': len(y),
            'dcols': {d: tr[d].values for d in DIMS}}

def fit_from(pre, kappa, kappa2):
    u = {}
    for d in DIMS:
        raw, nn = pre['dim_stats'][d]['raw'], pre['dim_stats'][d]['n']
        u[d] = {v: raw[v] * (nn[v] / (nn[v] + kappa)) for v in raw}
    seg = np.zeros(pre['ny'])
    for d in DIMS:
        seg += pd.Series(pre['dcols'][d]).map(u[d]).fillna(0).values
    seg_s = pd.Series(seg).groupby(pre['seller']).mean()
    n_s = pd.Series(pre['seller']).groupby(pre['seller']).size()
    gsb = pre['gsb'].copy()
    gsb['seg'] = gsb['fk_glusr_usr_id'].map(seg_s)
    th = pre['theta']
    gsb['sbr'] = [sig(logit(th[b]) + s) for b, s in zip(gsb['band'], gsb['seg'])]
    gsb['v'] = (logit(gsb['ys'] / gsb['n']) - logit(gsb['sbr'])) * gsb['n'] / (gsb['n'] + kappa2)
    piv = gsb.pivot(index='fk_glusr_usr_id', columns='band', values='v').reindex(columns=BANDS).fillna(0.0)
    return {'g': pre['g'], 'theta': th, 'u': u, 'sv_df': piv, 'seg_s': seg_s, 'n_s': n_s}

def seg_shift_of(model, df):
    seg = np.zeros(len(df))
    for d in DIMS:
        seg += df[d].map(model['u'][d]).fillna(0).values
    return seg

def predict_matrix(model, dims_df, seller_ids):
    seg = seg_shift_of(model, dims_df)
    V = model['sv_df'].reindex(seller_ids).fillna(0.0).values
    base = seg[:, None] + V
    P = np.column_stack([sig(logit(model['theta'][b]) + base[:, j]) for j, b in enumerate(BANDS)])
    return P, seg

def nll_on(model, df, ycol):
    P, seg = predict_matrix(model, df, df['fk_glusr_usr_id'].values)
    bi = pd.Categorical(df['band'], categories=BANDS).codes
    pred = P[np.arange(len(df)), bi]
    y = df[ycol].values
    return -np.mean(y * np.log(np.clip(pred, 1e-9, 1)) + (1 - y) * np.log(np.clip(1 - pred, 1e-9, 1)))

def main():
    b = load()
    b['is_ans'] = (b['lead_call_status'] == 'Answered').astype(int)
    ans_count = b[b['month'] != '2026-09'].groupby('fk_glusr_usr_id')['is_ans'].sum()
    fresh = b[b['fresh']].copy()
    tr = fresh[fresh['month'].isin(['2026-04', '2026-05', '2026-06', '2026-07'])].copy()
    aug = fresh[fresh['month'] == '2026-08'].copy()
    print(f"fresh train (Apr-Jul) n={len(tr):,}  fresh Aug n={len(aug):,}")

    best = {}
    for ycol in ['y_meet', 'y_ans']:
        pre = precompute(tr, ycol)
        bestnll, bestkk = 1e9, (20, 5)
        for k in [5, 20, 50, 100]:
            for k2 in [2, 5, 10, 20]:
                m = fit_from(pre, k, k2)
                nll = nll_on(m, aug, ycol)
                if nll < bestnll:
                    bestnll, bestkk = nll, (k, k2)
        best[ycol] = {'kappa': bestkk[0], 'kappa2': bestkk[1], 'aug_nll': float(bestnll)}
        print(f"{ycol}: best kappa={bestkk[0]} kappa2={bestkk[1]} aug_nll={bestnll:.4f}")

    pre_m = precompute(tr, 'y_meet'); pre_a = precompute(tr, 'y_ans')
    m_meet = fit_from(pre_m, best['y_meet']['kappa'], best['y_meet']['kappa2'])
    m_ans = fit_from(pre_a, best['y_ans']['kappa'], best['y_ans']['kappa2'])
    print("band curve meeting:", {k: round(v * 100, 2) for k, v in m_meet['theta'].items()})
    print("band curve answer :", {k: round(v * 100, 2) for k, v in m_ans['theta'].items()})

    # ---------- seller_windows for every seller ----------
    s = pd.read_csv(SEL, usecols=['fk_glusr_usr_id', 'seller_state', 'annual_turnover',
                                  'gst_registration_year', 'top_category_group'])
    s = s.rename(columns={'fk_glusr_usr_id': 'seller_id'})
    s['state'] = s['seller_state'].fillna('NA').astype(str)
    s['turnover'] = s['annual_turnover'].fillna('NA').astype(str)
    s['cat'] = s['top_category_group'].fillna('NA').astype(str)
    s['gst_bucket'] = pd.cut(s['gst_registration_year'], [0, 2015, 2019, 2023, 2027],
                             labels=['<=2015', '2016-2019', '2020-2023', '2024+']).astype(str)
    b2 = b.sort_values('call_start_time')
    last = b2.groupby('fk_glusr_usr_id').agg(
        last_bucket=('bucket', 'last'), last_disp=('disposition_label', 'last'),
        last_time=('call_start_time', 'last')).reset_index().rename(columns={'fk_glusr_usr_id': 'seller_id'})
    s = s.merge(last, on='seller_id', how='left')
    s['bucket'] = s['last_bucket'].fillna('NA')

    Pm, seg_m = predict_matrix(m_meet, s, s['seller_id'].values)
    Pa, seg_a = predict_matrix(m_ans, s, s['seller_id'].values)

    order = np.argsort(-Pm, axis=1)
    rec_idx = order[:, 0]
    rec_band = np.array(BANDS)[rec_idx]
    backup1 = np.array(BANDS)[order[:, 1]]
    backup2 = np.array(BANDS)[order[:, 2]]
    p_meet_rec = Pm[np.arange(len(s)), rec_idx]
    p_ans_rec = Pa[np.arange(len(s)), rec_idx]
    margin = Pm[np.arange(len(s)), order[:, 0]] - Pm[np.arange(len(s)), order[:, 1]]
    n_i = pd.Series(s['seller_id']).map(m_meet['n_s']).fillna(0).values
    ans_i = pd.Series(s['seller_id']).map(ans_count).fillna(0).values
    dev_exists = (m_meet['sv_df'].abs().sum(axis=1) > 0)
    dev_i = pd.Series(s['seller_id']).map(dev_exists).fillna(False).values
    # confidence reflects history depth: answered calls, plus a seller-specific band deviation
    conf = np.where((ans_i >= 3) & dev_i, 'high',
                    np.where(ans_i >= 1, 'med', 'low'))
    reason = np.where(n_i >= 5, 'seller_history',
                      np.where(np.abs(seg_m) >= 0.10, 'segment_level', 'global_curve'))

    state_map = {'Not Answered': 'Not Answered', 'Call Later / Busy': 'Call Later',
                 'General (talked)': 'General', 'Not Interested': 'Not Interested',
                 'Meeting Fixed': 'Meeting Fixed'}
    cur_state = s['last_disp'].map(state_map).fillna('Fresh').values

    def band_start_ts(day, band):
        return pd.Timestamp(day) + pd.Timedelta(hours=BAND_START[band])
    nca, rdo = [], []
    for st_, t_last, rb in zip(cur_state, s['last_time'].values, rec_band):
        if st_ == 'Fresh' or pd.isna(t_last):
            nca.append(pd.NaT if pd.isna(t_last) else pd.Timestamp(t_last)); rdo.append(0); continue
        t = pd.Timestamp(t_last)
        if st_ == 'Meeting Fixed':
            nca.append(pd.NaT); rdo.append(None)
        elif st_ == 'Not Answered':
            cand = t + pd.Timedelta(hours=6)
            if not (9 <= cand.hour < 21):
                cand = band_start_ts((t + pd.Timedelta(days=1)).normalize(), rb)
            nca.append(cand); rdo.append(0)
        elif st_ == 'Call Later':
            cand = t + pd.Timedelta(hours=3)
            if not (9 <= cand.hour < 21):
                cand = band_start_ts((t + pd.Timedelta(days=1)).normalize(), rb)
            nca.append(cand); rdo.append(0)
        elif st_ == 'General':
            nca.append(band_start_ts((t + pd.Timedelta(days=5)).normalize(), rb)); rdo.append(5)
        elif st_ == 'Not Interested':
            nca.append(band_start_ts((t + pd.Timedelta(days=14)).normalize(), rb)); rdo.append(14)
        else:
            nca.append(pd.NaT); rdo.append(None)

    out = pd.DataFrame({
        'seller_id': s['seller_id'].values, 'current_state': cur_state,
        'rec_band': rec_band, 'backup_band_1': backup1, 'backup_band_2': backup2,
        'rec_day_offset': rdo, 'p_meeting': np.round(p_meet_rec, 4),
        'p_answer': np.round(p_ans_rec, 4), 'confidence': conf, 'reason_code': reason,
        'next_call_at': [pd.Timestamp(x).isoformat() if pd.notna(x) else '' for x in nca]})
    out.to_csv('/scratch/work/seller_windows.csv', index=False)
    print("wrote seller_windows.csv", out.shape)
    print("rec_band mix:", out['rec_band'].value_counts().to_dict())
    print("confidence mix:", out['confidence'].value_counts().to_dict())
    print("reason mix:", out['reason_code'].value_counts().to_dict())
    json.dump({'tuning': best, 'band_curve_meeting': m_meet['theta'],
               'band_curve_answer': m_ans['theta']},
              open('/scratch/work/timing_model_tuning.json', 'w'), indent=2)

if __name__ == '__main__':
    main()
