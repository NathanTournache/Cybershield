# entraînement du classifieur binaire
"""
Entraînement : Normal   vs   Attack
Sauve : models/model_binary.pkl  +  models/scaler.pkl  +  models/encoders.pkl
"""
import os, joblib, numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from datetime import datetime

DATA_PATH = "data/KDDTrain+.txt"
MODEL_DIR = "models"; os.makedirs(MODEL_DIR, exist_ok=True)

log = lambda m: print(f"[{datetime.now():%H:%M:%S}] {m}")

COL_NAMES = [
    'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
    'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in',
    'num_compromised', 'root_shell', 'su_attempted', 'num_root',
    'num_file_creations', 'num_shells', 'num_access_files', 'num_outbound_cmds',
    'is_host_login', 'is_guest_login', 'count', 'srv_count', 'serror_rate',
    'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate',
    'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count',
    'dst_host_srv_count', 'dst_host_same_srv_rate', 'dst_host_diff_srv_rate',
    'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate',
    'dst_host_serror_rate', 'dst_host_srv_serror_rate', 'dst_host_rerror_rate',
    'dst_host_srv_rerror_rate', 'class', 'difficulty'
]

# -----------------------------------------------------------------------------
log("Chargement NSL-KDD …")
df = pd.read_csv(DATA_PATH, header=None, names=COL_NAMES)  # au lieu de sep=';'
df.drop(columns=['difficulty'], inplace=True)  # ajouter cette ligne
df.replace('?', np.nan, inplace=True)
df.dropna(inplace=True)


# ➊ Encodage textes -----------------------------------------------------------
cat_cols = ['protocol_type', 'service', 'flag']
encoders = {}
for col in cat_cols:
    le = LabelEncoder();  df[col] = le.fit_transform(df[col])
    encoders[col] = le
joblib.dump(encoders, f"{MODEL_DIR}/encoders.pkl")

# ➋ Binaire : 0 normal / 1 attaque -------------------------------------------
df['class'] = df['class'].apply(lambda x: 0 if x.strip() == 'normal' else 1)

# ➌ Sous-ensemble de features -------------------------------------------------
# ➌ Garder TOUTES les features numériques pour cohérence avec train_specialized
KEEP = [
    'duration', 'src_bytes', 'dst_bytes', 'land', 'wrong_fragment', 
    'urgent', 'hot', 'num_failed_logins', 'logged_in', 'num_compromised',
    'root_shell', 'su_attempted', 'num_root', 'num_file_creations',
    'num_shells', 'num_access_files', 'num_outbound_cmds',
    'is_host_login', 'is_guest_login', 'count', 'srv_count',
    'serror_rate', 'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate',
    'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate',
    'dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate',
    'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
    'dst_host_srv_diff_host_rate', 'dst_host_serror_rate',
    'dst_host_srv_serror_rate', 'dst_host_rerror_rate',
    'dst_host_srv_rerror_rate', 'protocol_type', 'service', 'flag', 'class'
]
df = df[KEEP]

# ➍ Normalisation -------------------------------------------------------------
scaler = MinMaxScaler()
X = pd.DataFrame(scaler.fit_transform(df.drop('class', axis=1)),
                 columns=df.drop('class', axis=1).columns)
y = df['class'].values
joblib.dump(scaler, f"{MODEL_DIR}/scaler.pkl")

# ➎ Split + Random-Forest -----------------------------------------------------
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.2, stratify=y, random_state=42)
clf = RandomForestClassifier(n_estimators=200, random_state=42)
clf.fit(Xtr, ytr)
log(f"Accuracy test : {clf.score(Xte, yte):.4f}")
joblib.dump(clf, f"{MODEL_DIR}/model_binary.pkl")
log("✅ Binaire sauvegardé.")
