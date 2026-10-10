import pandas as pd, numpy as np

A = pd.read_csv('/scratch/input/attempts.csv')
S = pd.read_csv('/scratch/input/sellers.csv')
L = pd.read_csv('/scratch/input/call_later_summaries.csv')

for c in ['lead_sent_time','call_start_time','vendor_response_time','lead_tbro_time']:
    A[c] = pd.to_datetime(A[c], errors='coerce')

print("== ROW COUNTS ==")
print("attempts rows:", len(A))
print("sellers rows:", len(S))
print("summaries rows:", len(L))

print("\n== DUPLICATE ATTEMPT IDS ==", A['data_hotlead_disposition_dtlid'].duplicated().sum())

undialed = A['call_start_time'].isna().sum()
print("\n== UNDIALED (empty call_start_time) ==", undialed)
Ad = A[A['call_start_time'].notna()].copy()
print("dialed attempts:", len(Ad))

print("\n== JOIN: sellers ==")
m = Ad['fk_glusr_usr_id'].isin(S['fk_glusr_usr_id'])
print(f"seller join match: {m.sum()} / {len(Ad)} = {m.mean()*100:.2f}%")

print("\n== JOIN: summaries ==")
lm = L['data_hotlead_disposition_dtlid'].isin(A['data_hotlead_disposition_dtlid'])
print(f"summaries matched: {lm.sum()} / {len(L)} = {lm.mean()*100:.2f}%")

print("\n== DATE RANGE ==")
print("attempts lead_sent_time:", A['lead_sent_time'].min(), "->", A['lead_sent_time'].max())
print("attempts call_start_time:", Ad['call_start_time'].min(), "->", Ad['call_start_time'].max())

print("\n== OUTCOME MIX (disposition_label) ==")
print(A['disposition_label'].value_counts())
print("total:", A['disposition_label'].notna().sum())

print("\n== OUTCOME MIX (dialed only) ==")
print(Ad['disposition_label'].value_counts())

print("\n== TARGET / CONNECT (dialed) ==")
conn = (Ad['lead_call_status']=='Answered').mean()
print(f"connect rate: {conn*100:.1f}%  (n={len(Ad)})")
mtg = Ad['meeting_fixed'].mean()
print(f"meeting per dialed call: {mtg*100:.1f}%  (n={len(Ad)})")
print(f"meetings total: {int(Ad['meeting_fixed'].sum())}")
print(f"calls per meeting: {len(Ad)/Ad['meeting_fixed'].sum():.1f}")

print("\n== LABEL MEETING MISMATCH ==")
mm = ((A['disposition_label']=='Meeting Fixed') & (A['meeting_fixed']==0)).sum()
print("label 'Meeting Fixed' but meeting_fixed=0:", mm)

print("\n== BOT VERSION (dialed) ==")
print(Ad['lead_bot_version'].value_counts(dropna=False).head(8))
