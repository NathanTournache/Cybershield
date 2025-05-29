# script de détection hiérarchique
"""
Usage : python detection_hierarchical.py  input.csv  0.8
→ produit anomalies_detected.csv  (prob ≥ seuil)
"""
import sys, os, joblib, pandas as pd, numpy as np
from datetime import datetime

log = lambda m: print(f"[{datetime.now():%H:%M:%S}] {m}")
if len(sys.argv)<2:
    sys.exit("Usage : python detection_hierarchical.py fichier_CICIDS.csv [seuil]")

FILE = sys.argv[1]; TH = float(sys.argv[2]) if len(sys.argv)>2 else .8
MODEL_DIR="models"

# ① load models ---------------------------------------------------------------
bin_clf  = joblib.load(f"{MODEL_DIR}/model_binary.pkl")
scaler   = joblib.load(f"{MODEL_DIR}/scaler.pkl")
spec_clf = {f: joblib.load(f"{MODEL_DIR}/model_{f}.pkl") for f in ['DoS','Probe','R2L','U2R']}

# ② charger CICIDS & mapper ---------------------------------------------------
df_raw = pd.read_csv(FILE, low_memory=False)
col_map = {                     # mêmes clés que précédemment
    'Flow Duration':'duration', 'Total Length of Fwd Packets':'src_bytes',
    'Total Length of Bwd Packets':'dst_bytes', 'Total Fwd Packets':'count',
    'Total Backward Packets':'srv_count', 'Fwd PSH Flags':'serror_rate',
    'Bwd PSH Flags':'srv_serror_rate', 'Bwd Packets/s':'same_srv_rate',
    'URG Flag Count':'num_failed_logins', 'Init_Win_bytes_forward':'root_shell',
    'Flow Packets/s':'logged_in', 'Subflow Fwd Bytes':'num_file_creations',
    'Fwd Header Length':'flag', 'Destination Port':'service'
}
df = pd.DataFrame({v: df_raw[k] if k in df_raw else 0 for k,v in col_map.items()})
for col in scaler.feature_names_in_:
    if col not in df: df[col]=0
df.replace([np.inf,-np.inf],np.nan,inplace=True); df.fillna(0,inplace=True)
df = df[scaler.feature_names_in_]
X_scaled = pd.DataFrame(scaler.transform(df), columns=df.columns)

# ③ prédiction hiérarchique ---------------------------------------------------
bin_pred = bin_clf.predict(X_scaled)
bin_proba = bin_clf.predict_proba(X_scaled)[:,1]

records=[]
for i,(is_anom,pb) in enumerate(zip(bin_pred,bin_proba)):
    if is_anom and pb>=TH:
        x = X_scaled.iloc[i].values.reshape(1,-1)
        best_family, best_p = None, 0
        for fam,clf in spec_clf.items():
            p = clf.predict_proba(x)[0,1]
            if p>best_p: best_family,best_p = fam,p
        records.append({
            'row_id': i, 'anomaly_prob': round(pb,3),
            'attack_family': best_family, 'family_prob': round(best_p,3)
        })

out = pd.DataFrame(records)
out.to_csv("anomalies_detected.csv", index=False)
log(f"✅ {len(out)} anomalies / {len(df)} lignes  → anomalies_detected.csv")
