"""
train_model.py
Pré‑processing + entraînement d'un modèle IDS sur NSL‑KDD
"""

import os, sys, joblib, numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from datetime import datetime

# ---------- 0. Helpers --------------------------------------------------------
def log(msg: str):
    """Affiche un message horodaté."""
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")

# ---------- 1. Chargement -----------------------------------------------------
DATA_PATH = "data/NSL-KDD.csv"          # <-- adapte si besoin
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

log("Chargement du dataset NSL‑KDD…")
df = pd.read_csv(DATA_PATH, sep=';')
log(f"Dimensions initiales : {df.shape}")

# ---------- 2. Nettoyage ------------------------------------------------------
log("Remplacement des ‘?’ et suppression des lignes incomplètes…")
df.replace('?', np.nan, inplace=True)
initial = df.shape[0]
df.dropna(inplace=True)
log(f"Lignes supprimées : {initial - df.shape[0]} (reste {df.shape[0]} lignes)")

# Supprime la colonne toujours nulle num_outbound_cmds
if 'num_outbound_cmds' in df.columns:
    df.drop(columns=['num_outbound_cmds'], inplace=True)

# ---------- 3. Encodage des colonnes catégorielles ---------------------------
cat_cols = df.select_dtypes(include='object').columns.tolist()
cat_cols.remove('class')                      # on gère le label à part
log(f"Colonnes texte à encoder : {cat_cols}")

for col in cat_cols:
    df[col] = LabelEncoder().fit_transform(df[col])

# Encodage du label : 0=normal, 1=attaque
df['class'] = df['class'].apply(lambda x: 0 if x == 'normal' else 1)

# ---------- 4. Sélection (mapping minimal CICIDS) ----------------------------
KEEP = [
    'duration', 'src_bytes', 'dst_bytes', 'count', 'srv_count',
    'serror_rate', 'srv_serror_rate', 'same_srv_rate',
    'logged_in', 'num_failed_logins', 'root_shell', 'num_file_creations',
    'flag', 'service', 'class'
]
df = df[KEEP]
log(f"Colonnes finales conservées : {KEEP}")

# ---------- 5. Normalisation (Min‑Max) ---------------------------------------
X = df.drop('class', axis=1)
y = df['class']

# Labellise les colonnes à forte amplitude pour l’explication
amplitude = X.max() - X.min()

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

# ---------- 5bis. Étiquetage manuel des colonnes normalisées ------------------
# Pour documenter clairement quelles colonnes sont normalisées (utile Flask ou docs)

cols_scaled = [
    'duration', 'src_bytes', 'dst_bytes', 
    'count', 'srv_count',
    'num_failed_logins', 'num_file_creations'
]

log("✅ Colonnes normalisées (MinMaxScaler) :")
for col in cols_scaled:
    min_val = X[col].min()
    max_val = X[col].max()
    log(f" - {col:<18} (min={min_val:<10} max={max_val})")

# ---------- 6. Séparation Train/Test -----------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, stratify=y, random_state=42
)
log(f"Train : {X_train.shape},  Test : {X_test.shape}")

# ---------- 7. Entraînement du modèle ----------------------------------------
log("Entraînement du Random Forest (100 arbres)…")
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)
score = clf.score(X_test, y_test)
log(f"Précision sur le jeu de test : {score:.4f}")

# ---------- 8. Sauvegarde -----------------------------------------------------
joblib.dump(clf,   os.path.join(MODEL_DIR, "rf_nslkdd.pkl"))
joblib.dump(scaler,os.path.join(MODEL_DIR, "scaler_nslkdd.pkl"))
log("Modèle et scaler sauvegardés dans /models")

# ---------- 9. Sauvegarde des encodeurs (flag, service) ------------------------
encoders = {}
for col in ['flag', 'service']:
    le = LabelEncoder()
    le.fit(df[col])
    encoders[col] = le
joblib.dump(encoders, os.path.join(MODEL_DIR, "encoders_nslkdd.pkl"))
log("Encodeurs sauvegardés dans /models")

# ---------- 10. Indications post‑run ------------------------------------------
log("Pré‑traitement terminé. Utilise rf_nslkdd.pkl + scaler_nslkdd.pkl dans Flask.")
