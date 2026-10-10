"""Quick feasibility baseline: point-in-time features, temporal split, LightGBM for P(answer) and P(meeting)."""
import pandas as pd, numpy as np, os, lightgbm as lgb
from sklearn.metrics import roc_auc_score, average_precision_score, log_loss
pd.set_option('display.width', 250)
S = os.path.dirname(os.path.abspath(__file__))
a = pd.read_pickle(os.path.join(S,'a2.pkl')); s = pd.read_pickle(os.path.join(S,'s.pkl'))
a = a[a.call_start_time.notna()].sort_values(['fk_glusr_usr_id','t']).reset_index(drop=True)
g = a.groupby('fk_glusr_usr_id')
# point-in-time seller history (strictly prior attempts)
a['h_n'] = g.cumcount()
a['h_ans'] = g.ans.cumsum() - a.ans
a['h_mf'] = g.mf.cumsum() - a.mf
for d,k in [('Not Interested','ni'),('Call Later / Busy','cl'),('General (talked)','gen')]:
    z = (a.disposition_label==d).astype(int); a['h_'+k] = z.groupby(a.fk_glusr_usr_id).cumsum() - z
a['h_ans_rate'] = (a.h_ans + 1) / (a.h_n + 2)
a['prev_hr'] = g.t.shift(1).dt.hour
hr = a.t.dt.hour
# prior answered-hour mean (expanding, strictly prior)
ah = hr.where(a.ans==1)
a['h_ans_hr_mean'] = ah.groupby(a.fk_glusr_usr_id).transform(lambda x: x.shift(1).expanding().mean())
a['hr'] = hr; a['dow'] = a.t.dt.dayofweek; a['minute'] = a.t.dt.minute
a['delay_min'] = (a.call_start_time - a.lead_sent_time).dt.total_seconds()/60
a['gap_min'] = a.gap_min.fillna(-1)
a['prev_disp'] = a.prev_disp.fillna('NONE')
a['hr_dist_ans'] = (a.hr - a.h_ans_hr_mean).abs()
m = a.merge(s, on='fk_glusr_usr_id', how='left')
cats = ['redis_bucket','prev_disp','business_type','annual_turnover','nature_of_business','top_category_group','seller_state']
for c in cats: m[c] = m[c].astype('category')
m['bot'] = m.lead_bot_version.fillna('NA').where(m.lead_bot_version.isin(['main_vani','arrowhead']), 'other').astype('category')
feats = ['hr','dow','call_attempt_count','delay_min','gap_min','h_n','h_ans','h_mf','h_ni','h_cl','h_gen','h_ans_rate','h_ans_hr_mean','hr_dist_ans',
         'gst_registration_year','num_categories','bot'] + cats
tr = m[m.t < '2026-08-01']; va = m[(m.t>='2026-08-01')&(m.t<'2026-09-01')]; te = m[m.t>='2026-09-01']
print("rows train/val/test", len(tr), len(va), len(te))
res = {}
for y in ['ans','mf']:
    p = dict(objective='binary', learning_rate=0.05, num_leaves=63, min_data_in_leaf=200, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, verbose=-1)
    dtr = lgb.Dataset(tr[feats], tr[y]); dva = lgb.Dataset(va[feats], va[y])
    b = lgb.train(p, dtr, 2000, valid_sets=[dva], callbacks=[lgb.early_stopping(100, verbose=False)])
    pt = b.predict(te[feats], num_iteration=b.best_iteration)
    print(f"\n[{y}] best_iter={b.best_iteration}  TEST(Sep) AUC={roc_auc_score(te[y],pt):.4f}  PR-AUC={average_precision_score(te[y],pt):.4f} base={te[y].mean():.4f}  logloss={log_loss(te[y],pt):.4f} vs const {log_loss(te[y],np.full(len(te),tr[y].mean())):.4f}")
    imp = pd.Series(b.feature_importance('gain'), index=feats).sort_values(ascending=False); print((imp/imp.sum()).round(3).head(15).to_string())
    te = te.assign(**{'p_'+y: pt})
    # decile lift
    te['dec'] = pd.qcut(te['p_'+y].rank(method='first'), 10, labels=False)
    print("lift by decile (top=9):", te.groupby('dec')[y].mean().round(4).tolist())
    # seen vs new sellers
    for nm, mask in [('new seller', te.h_n==0), ('repeat seller', te.h_n>0)]:
        print(f"   {nm}: n={mask.sum()} AUC={roc_auc_score(te.loc[mask,y], te.loc[mask,'p_'+y]):.4f}")
    res[y]=b
# hour sensitivity: for fresh attempt-1 rows in test, how much does predicted P(mf) vary across counterfactual hours?
f = te[(te.call_attempt_count==1)].sample(20000, random_state=0)
H = range(9,21); mat = np.column_stack([res['mf'].predict(f[feats].assign(hr=h), num_iteration=res['mf'].best_iteration) for h in H])
print("\nmean counterfactual P(mf) by hour (attempt-1 test sample):", dict(zip(H, mat.mean(0).round(4))))
best = np.array(list(H))[mat.argmax(1)]
print("distribution of model-best hour:", pd.Series(best).value_counts(normalize=True).sort_index().round(3).to_dict())
print("avg (best - actual-hour) predicted uplift:", (mat.max(1) - res['mf'].predict(f[feats], num_iteration=res['mf'].best_iteration)).mean().round(4), " relative:", ((mat.max(1)/res['mf'].predict(f[feats], num_iteration=res['mf'].best_iteration))-1).mean().round(3))
