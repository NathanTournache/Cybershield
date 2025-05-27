from flask import Flask, request, render_template, redirect, url_for
import os
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from werkzeug.utils import secure_filename

# --------- CONFIG ---------
UPLOAD_FOLDER = 'uploads'
MODEL_DIR = 'models'
ALLOWED_EXTENSIONS = {'csv'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# --------- HELPERS ---------
def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def infer_main_cause(row, columns):
    causes = {
        'src_bytes': 'Trafic anormal sortant (DoS, exfiltration)',
        'dst_bytes': 'Trafic entrant suspect (scan ou DoS)',
        'count': 'Trop de connexions vers l\'hôte (scan ou brute force)',
        'srv_count': 'Accès répétés au même service (brute force)',
        'serror_rate': 'Échec de connexions TCP (scan SYN)',
        'srv_serror_rate': 'Échec de connexions au service (DoS)',
        'same_srv_rate': 'Répétition de service (reconnaissance)',
        'num_failed_logins': 'Échecs de login (brute-force)',
        'root_shell': 'Shell root obtenu (U2R)',
        'num_file_creations': 'Création de fichiers (malware/backdoor)',
        'logged_in': 'Connexion réussie (potentiellement légitime ou attaque avancée)',
        'flag': 'État de la session (REJ = échec suspect, SF = succès suspect)',
        'service': 'Service réseau ciblé (FTP/HTTP/Telnet/etc)'
    }
    suspect = row[columns].astype(float).abs().sort_values(ascending=False).idxmax()
    return f"{suspect}: {causes.get(suspect, 'Comportement suspect')}"

# --------- LOAD MODEL & SCALER ---------
log("🔵 Chargement du modèle et du scaler...")
clf = joblib.load(os.path.join(MODEL_DIR, "rf_nslkdd.pkl"))
scaler = joblib.load(os.path.join(MODEL_DIR, "scaler_nslkdd.pkl"))

# --------- ROUTES ---------
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        threshold = float(request.form.get('threshold', 0.8))

        if file.filename == '' or not allowed_file(file.filename):
            return "Fichier non valide."

        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        df_raw = pd.read_csv(filepath, low_memory=False)

        # MAPPING DES COLONNES
        column_map = {
            'Flow Duration': 'duration',
            'Total Length of Fwd Packets': 'src_bytes',
            'Total Length of Bwd Packets': 'dst_bytes',
            'Total Fwd Packets': 'count',
            'Total Backward Packets': 'srv_count',
            'Fwd PSH Flags': 'serror_rate',
            'Bwd PSH Flags': 'srv_serror_rate',
            'Bwd Packets/s': 'same_srv_rate',
            'URG Flag Count': 'num_failed_logins',
            'Init_Win_bytes_forward': 'root_shell',
            'Flow Packets/s': 'logged_in',
            'Subflow Fwd Bytes': 'num_file_creations',
            'Fwd Header Length': 'flag',
            'Destination Port': 'service'
        }

        cleaned_columns = {col: col.strip() for col in df_raw.columns}
        df_raw.rename(columns=cleaned_columns, inplace=True)

        df = pd.DataFrame()
        for raw_col, target_col in column_map.items():
            if raw_col in df_raw:
                df[target_col] = df_raw[raw_col]

        # Colonnes manquantes
        expected_columns = list(scaler.feature_names_in_)
        for col in expected_columns:
            if col not in df.columns:
                df[col] = 0

        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.dropna(inplace=True)
        df.reset_index(drop=True, inplace=True)
        df = df[expected_columns]

        df_scaled_array = scaler.transform(df)
        df_scaled = pd.DataFrame(df_scaled_array, columns=df.columns)

        y_pred = clf.predict(df_scaled)
        y_proba = clf.predict_proba(df_scaled)[:, 1]

        anomalies = []
        for idx, (pred, proba) in enumerate(zip(y_pred, y_proba)):
            if pred == 1 and proba >= threshold:
                ligne = df.loc[idx].copy()
                ligne["anomaly_probability"] = round(proba, 4)
                anomalies.append(ligne)

        df_anomalies = pd.DataFrame(anomalies)
        return render_template("result.html", tables=[df_anomalies.to_html(classes='data')], titles=df_anomalies.columns.values)

    return render_template("index.html")

# --------- RUN ---------
if __name__ == '__main__':
    app.run(debug=True)
