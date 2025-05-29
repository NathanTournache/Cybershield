# script principal Flask
from flask import Flask, render_template, request, redirect, url_for, send_file, jsonify
import os, joblib, pandas as pd, numpy as np, uuid, time
from werkzeug.utils import secure_filename

UPLOAD_FOLDER = 'uploads'; MODEL_DIR='models'
ALLOWED_EXT={'csv'}
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# --- charge modèles en mémoire ---
bin_clf  = joblib.load(f"{MODEL_DIR}/model_binary.pkl")
scaler   = joblib.load(f"{MODEL_DIR}/scaler.pkl")
spec_clf = {f: joblib.load(f"{MODEL_DIR}/model_{f}.pkl") for f in ['DoS','Probe','R2L','U2R']}

# --- Flask ---
app = Flask(__name__, static_folder='static')
app.config['UPLOAD_FOLDER']=UPLOAD_FOLDER

def allowed(fname): return '.' in fname and fname.rsplit('.',1)[1].lower() in ALLOWED_EXT
def map_cicids(df_raw):
    col_map = { ... }  # (★)  même dict qu’au dessus
    df = pd.DataFrame({v: df_raw[k] if k in df_raw else 0 for k,v in col_map.items()})
    for col in scaler.feature_names_in_:
        if col not in df: df[col]=0
    df.replace([np.inf,-np.inf],np.nan,inplace=True); df.fillna(0,inplace=True)
    df = df[scaler.feature_names_in_]
    return pd.DataFrame(scaler.transform(df), columns=df.columns)

# ------------------------------
@app.route('/', methods=['GET','POST'])
def index():
    if request.method=='POST':
        f=request.files['file']
        th=float(request.form.get('threshold',.8))
        if f.filename=='' or not allowed(f.filename): return "Fichier invalide"
        fname=secure_filename(f.filename); path=os.path.join(app.config['UPLOAD_FOLDER'], fname); f.save(path)

        df_raw=pd.read_csv(path, low_memory=False)
        X_scaled=map_cicids(df_raw)

        # étapes avec “fake” progression
        steps=['Pré-traitement','Normalisation','Détection binaire','Classification fine']
        results=[]
        for s in steps:
            time.sleep(.6)                 # simple pause (UI)
            results.append({'step':s,'time':f"{time.time():.0f}"})

        # prédictions
        bin_pred=bin_clf.predict(X_scaled); bin_proba=bin_clf.predict_proba(X_scaled)[:,1]
        anomalies=[]
        for idx,(flag,pb) in enumerate(zip(bin_pred,bin_proba)):
            if flag and pb>=th:
                x=X_scaled.iloc[idx].values.reshape(1,-1)
                fam,maxp=None,0
                for k,clf in spec_clf.items():
                    p=clf.predict_proba(x)[0,1]; 
                    if p>maxp: fam,maxp=k,p
                anomalies.append({'row':idx,'anomaly_prob':round(pb,3),'family':fam,'fam_prob':round(maxp,3)})
        out=pd.DataFrame(anomalies)
        rid=str(uuid.uuid4())[:8]; csv_path=f"uploads/anom_{rid}.csv"
        out.to_csv(csv_path,index=False)

        return render_template("result.html", rows=out.to_dict(orient='records'),
                               csvfile=os.path.basename(csv_path), steps=steps)

    return render_template("index.html")

@app.route('/download/<csvfile>')
def download(csvfile): return send_file(os.path.join('uploads',csvfile), as_attachment=True)

if __name__=='__main__':
    app.run(debug=True)
