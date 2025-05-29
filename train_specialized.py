# train_specialized_improved.py
import os
import joblib
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline
from datetime import datetime

def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")

DATA_PATH = "data/KDDTrain+.txt"
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

# Colonnes NSL-KDD
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

# Chargement et préparation des données
log("📥 Chargement du dataset...")
df = pd.read_csv(DATA_PATH, header=None, names=COL_NAMES)
df.drop(columns=['difficulty'], inplace=True)
df.replace('?', np.nan, inplace=True)
df.dropna(inplace=True)

# Encodage
encoders = joblib.load(os.path.join(MODEL_DIR, "encoders.pkl"))
for col in ['protocol_type', 'service', 'flag']:
    df[col] = encoders[col].transform(df[col])

# Labels
df["label"] = df["class"]
df["class"] = df["class"].apply(lambda x: 0 if x == "normal" else 1)

attack_mapping = {
    'normal': 'normal',
    'neptune': 'dos', 'smurf': 'dos', 'back': 'dos', 'teardrop': 'dos', 'pod': 'dos',
    'satan': 'probe', 'ipsweep': 'probe', 'nmap': 'probe', 'portsweep': 'probe',
    'guess_passwd': 'r2l', 'ftp_write': 'r2l', 'imap': 'r2l', 'phf': 'r2l', 'multihop': 'r2l',
    'warezclient': 'r2l', 'warezmaster': 'r2l',
    'buffer_overflow': 'u2r', 'loadmodule': 'u2r', 'rootkit': 'u2r', 'perl': 'u2r'
}
df["attack_type"] = df["label"].map(attack_mapping)
df = df[df["attack_type"].notnull()]

# Normalisation
scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
X_total = df[scaler.feature_names_in_]
X_scaled = pd.DataFrame(
    scaler.transform(X_total),
    columns=scaler.feature_names_in_,
    index=df.index
)

# Configuration améliorée avec plus de features et meilleurs modèles
ATTACKS = {
    "dos": {
        "features": [
            'count', 'dst_host_count', 'serror_rate', 'srv_serror_rate',
            'same_srv_rate', 'dst_host_same_srv_rate', 'srv_count',
            'dst_host_srv_count', 'src_bytes', 'dst_bytes', 'duration'
        ],
        "models": {
            "rf": RandomForestClassifier(
                n_estimators=300, 
                max_depth=15, 
                min_samples_split=10,
                min_samples_leaf=5,
                class_weight='balanced',
                random_state=42
            ),
            "gb": GradientBoostingClassifier(
                n_estimators=200,
                learning_rate=0.1,
                max_depth=8,
                random_state=42
            )
        },
        "sampling_strategy": "balanced_subsample"  # Sous-échantillonnage équilibré
    },
    "probe": {
        "features": [
            'same_srv_rate', 'diff_srv_rate', 'flag', 'srv_diff_host_rate',
            'dst_host_same_srv_rate', 'dst_host_diff_srv_rate',
            'dst_host_srv_diff_host_rate', 'count', 'srv_count'
        ],
        "models": {
            "rf": RandomForestClassifier(
                n_estimators=200,
                max_depth=12,
                min_samples_split=8,
                class_weight='balanced',
                random_state=42
            ),
            "svm": SVC(
                kernel='rbf',
                C=10,
                gamma='scale',
                class_weight='balanced',
                probability=True,
                random_state=42
            )
        },
        "sampling_strategy": "smote"
    },
    "r2l": {
        "features": [
            'num_failed_logins', 'logged_in', 'root_shell', 'su_attempted',
            'num_compromised', 'num_root', 'hot', 'is_guest_login',
            'count', 'srv_count', 'same_srv_rate'
        ],
        "models": {
            "mlp": MLPClassifier(
                hidden_layer_sizes=(128, 64, 32),
                max_iter=2000,
                alpha=0.01,
                learning_rate='adaptive',
                random_state=42
            ),
            "rf": RandomForestClassifier(
                n_estimators=250,
                max_depth=10,
                class_weight='balanced',
                random_state=42
            )
        },
        "sampling_strategy": "smote"
    },
    "u2r": {
        "features": [
            'root_shell', 'num_file_creations', 'num_shells', 'su_attempted',
            'num_root', 'num_access_files', 'hot', 'num_compromised',
            'logged_in', 'is_guest_login'
        ],
        "models": {
            "mlp": MLPClassifier(
                hidden_layer_sizes=(64, 32, 16),
                max_iter=3000,
                alpha=0.001,
                learning_rate='adaptive',
                random_state=42
            ),
            "gb": GradientBoostingClassifier(
                n_estimators=150,
                learning_rate=0.05,
                max_depth=6,
                random_state=42
            )
        },
        "sampling_strategy": "smote"
    }
}

def apply_sampling_strategy(X, y, strategy):
    """Applique la stratégie d'échantillonnage appropriée"""
    if strategy == "smote":
        return SMOTE(random_state=42).fit_resample(X, y)
    elif strategy == "adasyn":
        return ADASYN(random_state=42).fit_resample(X, y)
    elif strategy == "balanced_subsample":
        # Sous-échantillonnage de la classe majoritaire + SMOTE
        undersample = RandomUnderSampler(sampling_strategy=0.5, random_state=42)
        X_under, y_under = undersample.fit_resample(X, y)
        return SMOTE(random_state=42).fit_resample(X_under, y_under)
    else:
        return X, y

# Entraînement amélioré
best_models = {}

for attack_type, config in ATTACKS.items():
    log(f"\n🔍 Entraînement amélioré pour {attack_type.upper()}...")

    # Préparation des données
    df_attack = df[df["attack_type"] == attack_type]
    df_normal = df[df["attack_type"] == "normal"]
    
    if df_attack.empty:
        log(f"⚠️ Aucun échantillon pour {attack_type}, skipping.")
        continue

    # Équilibrage des classes - ratio plus intelligent
    normal_sample_size = min(len(df_attack) * 3, len(df_normal), 5000)
    df_normal_sampled = df_normal.sample(n=normal_sample_size, random_state=42)
    
    df_subset = pd.concat([df_attack, df_normal_sampled])
    X = X_scaled.loc[df_subset.index, config["features"]]
    y = [1 if df.loc[idx, "attack_type"] == attack_type else 0 for idx in df_subset.index]
    
    log(f"   Données initiales: {len(y)} échantillons ({sum(y)} attaques, {len(y)-sum(y)} normaux)")
    
    # Application de la stratégie d'échantillonnage
    X_resampled, y_resampled = apply_sampling_strategy(X, y, config["sampling_strategy"])
    log(f"   Après rééchantillonnage: {len(y_resampled)} échantillons ({sum(y_resampled)} attaques, {len(y_resampled)-sum(y_resampled)} normaux)")
    
    # Test de plusieurs modèles
    best_score = 0
    best_model = None
    best_model_name = None
    
    for model_name, model in config["models"].items():
        log(f"   🔄 Test du modèle {model_name}...")
        
        # Validation croisée
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        scores = []
        
        for train_idx, val_idx in cv.split(X_resampled, y_resampled):
            X_train, X_val = X_resampled.iloc[train_idx], X_resampled.iloc[val_idx]
            y_train, y_val = np.array(y_resampled)[train_idx], np.array(y_resampled)[val_idx]
            
            model.fit(X_train, y_train)
            score = f1_score(y_val, model.predict(X_val))
            scores.append(score)
        
        avg_score = np.mean(scores)
        log(f"      F1-Score moyen: {avg_score:.4f}")
        
        if avg_score > best_score:
            best_score = avg_score
            best_model = model
            best_model_name = model_name
    
    # Entraînement final du meilleur modèle
    log(f"   🏆 Meilleur modèle: {best_model_name} (F1={best_score:.4f})")
    best_model.fit(X_resampled, y_resampled)
    
    # Sauvegarde
    model_path = os.path.join(MODEL_DIR, f"clf_{attack_type}_improved.pkl")
    joblib.dump({
        'model': best_model,
        'features': config["features"],
        'model_type': best_model_name,
        'f1_score': best_score
    }, model_path)
    
    best_models[attack_type] = {
        'model': best_model,
        'features': config["features"],
        'model_type': best_model_name
    }
    
    log(f"✅ Modèle {attack_type} amélioré sauvegardé.")

# Évaluation sur le dataset de test
TEST_PATH = "data/KDDTest+.txt"
if os.path.exists(TEST_PATH):
    log("\n📊 ÉVALUATION DES MODÈLES AMÉLIORÉS...")
    
    # Chargement du test set
    df_test = pd.read_csv(TEST_PATH, header=None, names=COL_NAMES)
    df_test.drop(columns=['difficulty'], inplace=True)
    df_test.replace('?', np.nan, inplace=True)
    df_test.dropna(inplace=True)
    
    # Encodage
    for col in ['protocol_type', 'service', 'flag']:
        df_test[col] = df_test[col].apply(lambda x: x if x in encoders[col].classes_ else encoders[col].classes_[0])
        df_test[col] = encoders[col].transform(df_test[col])
    
    # Labels
    df_test["label"] = df_test["class"]
    df_test["attack_type"] = df_test["label"].map(attack_mapping)
    df_test = df_test[df_test["attack_type"].notnull()]
    
    # Normalisation
    X_test_scaled = pd.DataFrame(
        scaler.transform(df_test[scaler.feature_names_in_]),
        columns=scaler.feature_names_in_,
        index=df_test.index
    )
    
    # Évaluation de chaque modèle
    results = {}
    
    for attack_type in best_models.keys():
        log(f"\n🎯 ÉVALUATION {attack_type.upper()}...")
        
        model_info = best_models[attack_type]
        model = model_info['model']
        features = model_info['features']
        
        # Données de test
        df_attack_test = df_test[df_test["attack_type"] == attack_type]
        df_normal_test = df_test[df_test["attack_type"] == "normal"]
        
        if df_attack_test.empty:
            log(f"⚠️ Pas d'échantillons de test pour {attack_type}")
            continue
        
        df_test_subset = pd.concat([df_attack_test, df_normal_test])
        X_test = X_test_scaled.loc[df_test_subset.index, features]
        y_test = [1 if df_test.loc[idx, "attack_type"] == attack_type else 0 for idx in df_test_subset.index]
        
        # Prédictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
        
        # Métriques
        accuracy = model.score(X_test, y_test)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        # Matrice de confusion
        cm = confusion_matrix(y_test, y_pred)
        
        results[attack_type] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': cm,
            'model_type': model_info['model_type']
        }
        
        log(f"   Modèle: {model_info['model_type']}")
        log(f"   Échantillons: {len(y_test)} ({sum(y_test)} attaques, {len(y_test)-sum(y_test)} normaux)")
        log(f"   Accuracy: {accuracy:.4f}")
        log(f"   Precision: {precision:.4f}")
        log(f"   Recall: {recall:.4f}")
        log(f"   F1-Score: {f1:.4f}")
        log(f"   Matrice: TN={cm[0,0]}, FP={cm[0,1]}, FN={cm[1,0]}, TP={cm[1,1]}")
    
    # Sauvegarde des résultats
    joblib.dump(results, os.path.join(MODEL_DIR, "evaluation_results.pkl"))
    log("\n✅ Évaluation terminée et sauvegardée.")

else:
    log("⚠️ Fichier KDDTest+.txt non trouvé.")