"""
UCI Heart Disease - Revize Senaryo Karşılaştırmalı Analiz
==========================================================

Senaryolar:
- Senaryo 0: Baseline (RobustScaler + 10-Fold CV) - 6 Model
- Senaryo 1: + PCA (StandardScaler + PCA + 10-Fold CV) - 6 Model
- Senaryo 2: + Feature Engineering (RobustScaler + FE + 10-Fold CV) - 6 Model
- Senaryo 3: + SMOTE (RobustScaler + SMOTE + 10-Fold CV) - 6 Model
- Senaryo 4: + Optuna (RobustScaler + Optuna + 10-Fold CV) - 6 Model
- Senaryo 5: All Combined (Tüm teknikler) - En iyi ve en kötü 2 model

Her senaryoda test edilen modeller:
- Random Forest, SVM, Logistic Regression, Naive Bayes, XGBoost, KNN

NOT: Sadece Cleveland veri seti kullanılmaktadır.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Preprocessing
from sklearn.impute import KNNImputer
from sklearn.preprocessing import RobustScaler, LabelEncoder, StandardScaler
from sklearn.decomposition import PCA

# Modeller
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier

# Validasyon ve Metrikler
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict
from sklearn.metrics import confusion_matrix, roc_curve, auc

# Dengesiz Veri
from imblearn.over_sampling import SMOTE

# Optuna
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

# Görselleştirme ayarları
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (14, 8)

# Global değişken: Sonuç klasörü
RESULTS_DIR = None

# Model tanımları (varsayılan parametrelerle)
def get_default_models():
    """Tüm modellerin varsayılan halini döndür"""
    return {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Random Forest': RandomForestClassifier(random_state=42, n_jobs=-1),
        'SVM': SVC(probability=True, random_state=42),
        'Naive Bayes': GaussianNB(),
        'XGBoost': XGBClassifier(random_state=42, n_jobs=-1, use_label_encoder=False, 
                                  eval_metric='logloss', verbosity=0),
        'KNN': KNeighborsClassifier(n_jobs=-1)
    }

def create_results_folder():
    """Tarih ve saat damgalı sonuç klasörü oluştur"""
    global RESULTS_DIR
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    RESULTS_DIR = Path(__file__).parent / f"scenario_results_{timestamp}"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\n📂 Sonuç klasörü: {RESULTS_DIR}")
    return RESULTS_DIR

# ============================================================
# VERİ YÜKLEME VE TEMEL İŞLEME
# ============================================================

def load_cleveland_data(filepath):
    """Cleveland verisini yükle ve temel temizleme yap"""
    df = pd.read_csv(filepath)
    
    # Sadece Cleveland
    df = df[df['dataset'] == 'Cleveland'].copy()
    
    # Binary hedef
    df['target'] = (df['num'] > 0).astype(int)
    
    print(f"📊 Cleveland Veri Seti: {len(df)} örnek")
    print(f"   Sağlıklı: {sum(df['target']==0)}, Hasta: {sum(df['target']==1)}")
    
    return df

def basic_preprocessing(df):
    """Temel veri önişleme (tüm senaryolar için ortak)"""
    df_processed = df.copy()
    
    # Kategorik değişkenleri encode et
    categorical_cols = ['sex', 'cp', 'restecg', 'exang', 'slope', 'thal', 'fbs']
    
    for col in categorical_cols:
        if col in df_processed.columns:
            le = LabelEncoder()
            df_processed[col] = df_processed[col].fillna('missing')
            df_processed[col] = le.fit_transform(df_processed[col].astype(str))
    
    # Sayısal sütunları belirle
    exclude_cols = ['id', 'num', 'target', 'dataset']
    numeric_cols = [col for col in df_processed.select_dtypes(include=[np.number]).columns 
                   if col not in exclude_cols]
    
    # KNN Imputer - Eksik değerleri doldur
    imputer = KNNImputer(n_neighbors=5)
    df_processed[numeric_cols] = imputer.fit_transform(df_processed[numeric_cols])
    
    return df_processed

def add_feature_engineering(df):
    """Özellik mühendisliği ekle"""
    df_fe = df.copy()
    
    # Risk skoru
    df_fe['risk_score'] = (df_fe['age'] * df_fe['chol']) / 10000
    
    # Yaş kategorileri
    df_fe['age_group'] = pd.cut(
        df_fe['age'], 
        bins=[0, 40, 55, 70, 100],
        labels=[0, 1, 2, 3]
    ).astype(float).fillna(1).astype(int)
    
    # Kalp hızı / yaş oranı
    df_fe['hr_age_ratio'] = df_fe['thalch'] / (df_fe['age'] + 1)
    
    # Kan basıncı × kolesterol
    df_fe['bp_chol_interaction'] = (df_fe['trestbps'] * df_fe['chol']) / 10000
    
    return df_fe

def get_features_target(df, exclude_extra=[]):
    """Özellik ve hedef ayır"""
    exclude_cols = ['id', 'num', 'target', 'dataset'] + exclude_extra
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    X = df[feature_cols].values
    y = df['target'].values
    
    return X, y, feature_cols

# ============================================================
# MODEL DEĞERLENDİRME
# ============================================================

def evaluate_all_models(X, y, models=None, cv=10):
    """Tüm modelleri 10-Fold CV ile değerlendir"""
    if models is None:
        models = get_default_models()
    
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    results = {}
    
    for name, model in models.items():
        try:
            acc = cross_val_score(model, X, y, cv=skf, scoring='accuracy')
            f1 = cross_val_score(model, X, y, cv=skf, scoring='f1')
            rec = cross_val_score(model, X, y, cv=skf, scoring='recall')
            auc_score = cross_val_score(model, X, y, cv=skf, scoring='roc_auc')
            
            results[name] = {
                'accuracy': f"{acc.mean():.3f}±{acc.std():.3f}",
                'f1': f"{f1.mean():.3f}±{f1.std():.3f}",
                'recall': f"{rec.mean():.3f}±{rec.std():.3f}",
                'auc': f"{auc_score.mean():.3f}±{auc_score.std():.3f}",
                'f1_mean': f1.mean(),
                'acc_mean': acc.mean(),
                'recall_mean': rec.mean(),
                'auc_mean': auc_score.mean()
            }
        except Exception as e:
            print(f"   ⚠️ {name} hatası: {e}")
            results[name] = {
                'accuracy': "N/A", 'f1': "N/A", 'recall': "N/A", 'auc': "N/A",
                'f1_mean': 0, 'acc_mean': 0, 'recall_mean': 0, 'auc_mean': 0
            }
    
    return results

def print_results_table(results, title="Sonuçlar"):
    """Sonuç tablosunu yazdır"""
    print(f"\n{'─'*90}")
    print(f"{'Model':<22} {'Accuracy':<15} {'Recall':<15} {'F1-Score':<15} {'AUC':<15}")
    print(f"{'─'*90}")
    
    for name, res in results.items():
        print(f"{name:<22} {res['accuracy']:<15} {res['recall']:<15} "
              f"{res['f1']:<15} {res['auc']:<15}")
    print(f"{'─'*90}")
    
    # En iyi ve en kötü model
    valid_results = {k: v for k, v in results.items() if v['f1_mean'] > 0}
    if valid_results:
        best = max(valid_results.items(), key=lambda x: x[1]['f1_mean'])
        worst = min(valid_results.items(), key=lambda x: x[1]['f1_mean'])
        print(f"   🏆 En İyi: {best[0]} (F1={best[1]['f1']})")
        print(f"   📉 En Kötü: {worst[0]} (F1={worst[1]['f1']})")
        return best[0], worst[0]
    return None, None

# ============================================================
# SENARYO 0: BASELINE
# ============================================================

def scenario_0_baseline(df):
    """
    Senaryo 0: Baseline
    - RobustScaler
    - 10-Fold CV
    - 6 Model (varsayılan parametreler)
    """
    print("\n" + "="*70)
    print("📌 SENARYO 0: BASELINE")
    print("   RobustScaler + 10-Fold CV + 6 Model (varsayılan)")
    print("="*70)
    
    df_processed = basic_preprocessing(df)
    X, y, features = get_features_target(df_processed)
    
    # Ölçekleme
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)
    
    print(f"   Özellik sayısı: {len(features)}")
    print(f"   Örnek sayısı: {len(y)} (Sağlıklı: {sum(y==0)}, Hasta: {sum(y==1)})")
    
    # Tüm modelleri test et
    results = evaluate_all_models(X_scaled, y)
    best_model, worst_model = print_results_table(results, "Senaryo 0 Sonuçları")
    
    return results, features, best_model, worst_model

# ============================================================
# SENARYO 1: + PCA
# ============================================================

def scenario_1_pca(df):
    """
    Senaryo 1: + PCA
    - StandardScaler (PCA için zorunlu)
    - PCA (%95 varyans)
    - 10-Fold CV
    - 6 Model (varsayılan parametreler)
    """
    print("\n" + "="*70)
    print("📌 SENARYO 1: + PCA")
    print("   StandardScaler + PCA(%95) + 10-Fold CV + 6 Model")
    print("="*70)
    
    df_processed = basic_preprocessing(df)
    X, y, features = get_features_target(df_processed)
    
    # StandardScaler (PCA için zorunlu)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # PCA
    pca = PCA(n_components=0.95, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    
    print(f"   Orijinal özellik: {X_scaled.shape[1]}")
    print(f"   PCA sonrası: {X_pca.shape[1]} bileşen")
    print(f"   Açıklanan varyans: {pca.explained_variance_ratio_.sum():.2%}")
    
    # Tüm modelleri test et
    results = evaluate_all_models(X_pca, y)
    print_results_table(results, "Senaryo 1 Sonuçları")
    
    return results

# ============================================================
# SENARYO 2: + FEATURE ENGINEERING
# ============================================================

def scenario_2_feature_engineering(df):
    """
    Senaryo 2: + Feature Engineering
    - RobustScaler
    - +4 mühendislik özelliği
    - 10-Fold CV
    - 6 Model (varsayılan parametreler)
    """
    print("\n" + "="*70)
    print("📌 SENARYO 2: + FEATURE ENGINEERING")
    print("   RobustScaler + 4 Yeni Özellik + 10-Fold CV + 6 Model")
    print("="*70)
    
    df_processed = basic_preprocessing(df)
    df_fe = add_feature_engineering(df_processed)
    X, y, features = get_features_target(df_fe)
    
    # Ölçekleme
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)
    
    print(f"   Özellik sayısı: {len(features)} (+4 yeni)")
    print(f"   Yeni özellikler: risk_score, age_group, hr_age_ratio, bp_chol_interaction")
    
    # Tüm modelleri test et
    results = evaluate_all_models(X_scaled, y)
    print_results_table(results, "Senaryo 2 Sonuçları")
    
    return results, features

# ============================================================
# SENARYO 3: + SMOTE
# ============================================================

def scenario_3_smote(df):
    """
    Senaryo 3: + SMOTE
    - RobustScaler
    - SMOTE (sınıf dengeleme)
    - 10-Fold CV
    - 6 Model (varsayılan parametreler)
    """
    print("\n" + "="*70)
    print("📌 SENARYO 3: + SMOTE")
    print("   RobustScaler + SMOTE + 10-Fold CV + 6 Model")
    print("="*70)
    
    df_processed = basic_preprocessing(df)
    X, y, features = get_features_target(df_processed)
    
    # Ölçekleme
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)
    
    # SMOTE
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_scaled, y)
    
    print(f"   Orijinal: Sağlıklı={sum(y==0)}, Hasta={sum(y==1)}")
    print(f"   SMOTE sonrası: Sağlıklı={sum(y_resampled==0)}, Hasta={sum(y_resampled==1)}")
    
    # Tüm modelleri test et
    results = evaluate_all_models(X_resampled, y_resampled)
    print_results_table(results, "Senaryo 3 Sonuçları")
    
    return results

# ============================================================
# SENARYO 4: + OPTUNA
# ============================================================

def scenario_4_optuna(df, n_trials=30):
    """
    Senaryo 4: + Optuna
    - RobustScaler
    - Optuna hiperparametre optimizasyonu
    - 10-Fold CV
    - 6 Model (optimize edilmiş)
    """
    print("\n" + "="*70)
    print("📌 SENARYO 4: + OPTUNA")
    print(f"   RobustScaler + Optuna({n_trials} trial) + 10-Fold CV + 6 Model")
    print("="*70)
    
    df_processed = basic_preprocessing(df)
    X, y, features = get_features_target(df_processed)
    
    # Ölçekleme
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)
    
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    results = {}
    best_params_all = {}
    
    # 1. Logistic Regression
    print("\n   📌 1/6: Logistic Regression...")
    def lr_objective(trial):
        C = trial.suggest_float('C', 0.01, 10.0, log=True)
        penalty = trial.suggest_categorical('penalty', ['l1', 'l2'])
        solver = 'saga' if penalty == 'l1' else 'lbfgs'
        model = LogisticRegression(C=C, penalty=penalty, solver=solver, max_iter=1000, random_state=42)
        return cross_val_score(model, X_scaled, y, cv=skf, scoring='f1').mean()
    
    study = optuna.create_study(direction='maximize')
    study.optimize(lr_objective, n_trials=n_trials, show_progress_bar=True)
    best_params_all['Logistic Regression'] = study.best_params
    solver = 'saga' if study.best_params.get('penalty') == 'l1' else 'lbfgs'
    best_lr = LogisticRegression(**study.best_params, solver=solver, max_iter=1000, random_state=42)
    results['Logistic Regression'] = evaluate_single_model(best_lr, X_scaled, y, skf)
    
    # 2. Random Forest
    print("   📌 2/6: Random Forest...")
    def rf_objective(trial):
        model = RandomForestClassifier(
            n_estimators=trial.suggest_int('n_estimators', 50, 300),
            max_depth=trial.suggest_int('max_depth', 3, 20),
            min_samples_split=trial.suggest_int('min_samples_split', 2, 20),
            min_samples_leaf=trial.suggest_int('min_samples_leaf', 1, 10),
            random_state=42, n_jobs=-1
        )
        return cross_val_score(model, X_scaled, y, cv=skf, scoring='f1').mean()
    
    study = optuna.create_study(direction='maximize')
    study.optimize(rf_objective, n_trials=n_trials, show_progress_bar=True)
    best_params_all['Random Forest'] = study.best_params
    best_rf = RandomForestClassifier(**study.best_params, random_state=42, n_jobs=-1)
    results['Random Forest'] = evaluate_single_model(best_rf, X_scaled, y, skf)
    
    # 3. SVM
    print("   📌 3/6: SVM...")
    def svm_objective(trial):
        model = SVC(
            C=trial.suggest_float('C', 0.1, 100.0, log=True),
            gamma=trial.suggest_categorical('gamma', ['scale', 'auto']),
            kernel=trial.suggest_categorical('kernel', ['rbf', 'poly']),
            probability=True, random_state=42
        )
        return cross_val_score(model, X_scaled, y, cv=skf, scoring='f1').mean()
    
    study = optuna.create_study(direction='maximize')
    study.optimize(svm_objective, n_trials=n_trials, show_progress_bar=True)
    best_params_all['SVM'] = study.best_params
    best_svm = SVC(**study.best_params, probability=True, random_state=42)
    results['SVM'] = evaluate_single_model(best_svm, X_scaled, y, skf)
    
    # 4. Naive Bayes
    print("   📌 4/6: Naive Bayes...")
    def nb_objective(trial):
        model = GaussianNB(var_smoothing=trial.suggest_float('var_smoothing', 1e-12, 1e-6, log=True))
        return cross_val_score(model, X_scaled, y, cv=skf, scoring='f1').mean()
    
    study = optuna.create_study(direction='maximize')
    study.optimize(nb_objective, n_trials=n_trials, show_progress_bar=True)
    best_params_all['Naive Bayes'] = study.best_params
    best_nb = GaussianNB(**study.best_params)
    results['Naive Bayes'] = evaluate_single_model(best_nb, X_scaled, y, skf)
    
    # 5. XGBoost
    print("   📌 5/6: XGBoost...")
    def xgb_objective(trial):
        model = XGBClassifier(
            n_estimators=trial.suggest_int('n_estimators', 50, 300),
            max_depth=trial.suggest_int('max_depth', 3, 15),
            learning_rate=trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            subsample=trial.suggest_float('subsample', 0.6, 1.0),
            colsample_bytree=trial.suggest_float('colsample_bytree', 0.6, 1.0),
            random_state=42, n_jobs=-1, use_label_encoder=False, eval_metric='logloss', verbosity=0
        )
        return cross_val_score(model, X_scaled, y, cv=skf, scoring='f1').mean()
    
    study = optuna.create_study(direction='maximize')
    study.optimize(xgb_objective, n_trials=n_trials, show_progress_bar=True)
    best_params_all['XGBoost'] = study.best_params
    best_xgb = XGBClassifier(**study.best_params, random_state=42, n_jobs=-1, 
                             use_label_encoder=False, eval_metric='logloss', verbosity=0)
    results['XGBoost'] = evaluate_single_model(best_xgb, X_scaled, y, skf)
    
    # 6. KNN
    print("   📌 6/6: KNN...")
    def knn_objective(trial):
        model = KNeighborsClassifier(
            n_neighbors=trial.suggest_int('n_neighbors', 3, 21, step=2),
            weights=trial.suggest_categorical('weights', ['uniform', 'distance']),
            metric=trial.suggest_categorical('metric', ['euclidean', 'manhattan']),
            n_jobs=-1
        )
        return cross_val_score(model, X_scaled, y, cv=skf, scoring='f1').mean()
    
    study = optuna.create_study(direction='maximize')
    study.optimize(knn_objective, n_trials=n_trials, show_progress_bar=True)
    best_params_all['KNN'] = study.best_params
    best_knn = KNeighborsClassifier(**study.best_params, n_jobs=-1)
    results['KNN'] = evaluate_single_model(best_knn, X_scaled, y, skf)
    
    print_results_table(results, "Senaryo 4 Sonuçları")
    
    # Parametreleri kaydet
    save_best_params(best_params_all, "scenario_4_optuna_params.txt")
    
    return results, best_params_all

def evaluate_single_model(model, X, y, cv):
    """Tek model için CV değerlendirmesi"""
    acc = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
    f1 = cross_val_score(model, X, y, cv=cv, scoring='f1')
    rec = cross_val_score(model, X, y, cv=cv, scoring='recall')
    auc_score = cross_val_score(model, X, y, cv=cv, scoring='roc_auc')
    
    return {
        'accuracy': f"{acc.mean():.3f}±{acc.std():.3f}",
        'f1': f"{f1.mean():.3f}±{f1.std():.3f}",
        'recall': f"{rec.mean():.3f}±{rec.std():.3f}",
        'auc': f"{auc_score.mean():.3f}±{auc_score.std():.3f}",
        'f1_mean': f1.mean(),
        'acc_mean': acc.mean(),
        'recall_mean': rec.mean(),
        'auc_mean': auc_score.mean()
    }

def save_best_params(params_dict, filename):
    """En iyi parametreleri kaydet"""
    params_path = RESULTS_DIR / filename
    with open(params_path, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("OPTUNA - EN İYİ HİPERPARAMETRELER\n")
        f.write("=" * 60 + "\n")
        for model_name, params in params_dict.items():
            f.write(f"\n{model_name}:\n")
            for k, v in params.items():
                f.write(f"  {k}: {v}\n")
    print(f"   ✓ {filename} kaydedildi")

# ============================================================
# SENARYO 5: ALL COMBINED
# ============================================================

def scenario_5_all_combined(df, best_model_name, worst_model_name, n_trials=50):
    """
    Senaryo 5: Tüm Teknikler Birlikte
    - StandardScaler
    - Feature Engineering
    - PCA
    - SMOTE
    - Optuna
    - 10-Fold CV
    - En iyi ve en kötü 2 model
    """
    print("\n" + "="*70)
    print("📌 SENARYO 5: ALL COMBINED")
    print(f"   FE + PCA + SMOTE + Optuna + 10-Fold CV")
    print(f"   Test edilen modeller: {best_model_name} (en iyi), {worst_model_name} (en kötü)")
    print("="*70)
    
    # Veri hazırlama
    df_processed = basic_preprocessing(df)
    df_fe = add_feature_engineering(df_processed)
    X, y, features = get_features_target(df_fe)
    
    # StandardScaler + PCA
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    pca = PCA(n_components=0.95, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    
    # SMOTE
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_pca, y)
    
    print(f"   Pipeline: {len(features)} özellik → PCA: {X_pca.shape[1]} → SMOTE: {len(y_resampled)}")
    
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    results = {}
    
    # En iyi model için Optuna
    print(f"\n   🏆 {best_model_name} optimizasyonu...")
    best_result, best_params = optimize_model(best_model_name, X_resampled, y_resampled, skf, n_trials)
    results[f"{best_model_name} (Best)"] = best_result
    
    # En kötü model için Optuna
    print(f"\n   📉 {worst_model_name} optimizasyonu...")
    worst_result, worst_params = optimize_model(worst_model_name, X_resampled, y_resampled, skf, n_trials)
    results[f"{worst_model_name} (Worst)"] = worst_result
    
    print_results_table(results, "Senaryo 5 Sonuçları")
    
    # Karşılaştırma
    print("\n   📊 İyileştirme Analizi:")
    print(f"   {best_model_name}: Tüm tekniklerle optimize edildi")
    print(f"   {worst_model_name}: Tüm tekniklerle optimize edildi")
    
    return results, X_resampled, y_resampled, features

def optimize_model(model_name, X, y, skf, n_trials):
    """Belirli bir modeli Optuna ile optimize et"""
    
    if model_name == 'Logistic Regression':
        def objective(trial):
            C = trial.suggest_float('C', 0.01, 10.0, log=True)
            penalty = trial.suggest_categorical('penalty', ['l1', 'l2'])
            solver = 'saga' if penalty == 'l1' else 'lbfgs'
            model = LogisticRegression(C=C, penalty=penalty, solver=solver, max_iter=1000, random_state=42)
            return cross_val_score(model, X, y, cv=skf, scoring='f1').mean()
        
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        solver = 'saga' if study.best_params.get('penalty') == 'l1' else 'lbfgs'
        best_model = LogisticRegression(**study.best_params, solver=solver, max_iter=1000, random_state=42)
        
    elif model_name == 'Random Forest':
        def objective(trial):
            model = RandomForestClassifier(
                n_estimators=trial.suggest_int('n_estimators', 50, 300),
                max_depth=trial.suggest_int('max_depth', 3, 20),
                min_samples_split=trial.suggest_int('min_samples_split', 2, 20),
                min_samples_leaf=trial.suggest_int('min_samples_leaf', 1, 10),
                random_state=42, n_jobs=-1
            )
            return cross_val_score(model, X, y, cv=skf, scoring='f1').mean()
        
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        best_model = RandomForestClassifier(**study.best_params, random_state=42, n_jobs=-1)
        
    elif model_name == 'SVM':
        def objective(trial):
            model = SVC(
                C=trial.suggest_float('C', 0.1, 100.0, log=True),
                gamma=trial.suggest_categorical('gamma', ['scale', 'auto']),
                kernel=trial.suggest_categorical('kernel', ['rbf', 'poly']),
                probability=True, random_state=42
            )
            return cross_val_score(model, X, y, cv=skf, scoring='f1').mean()
        
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        best_model = SVC(**study.best_params, probability=True, random_state=42)
        
    elif model_name == 'Naive Bayes':
        def objective(trial):
            model = GaussianNB(var_smoothing=trial.suggest_float('var_smoothing', 1e-12, 1e-6, log=True))
            return cross_val_score(model, X, y, cv=skf, scoring='f1').mean()
        
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        best_model = GaussianNB(**study.best_params)
        
    elif model_name == 'XGBoost':
        def objective(trial):
            model = XGBClassifier(
                n_estimators=trial.suggest_int('n_estimators', 50, 300),
                max_depth=trial.suggest_int('max_depth', 3, 15),
                learning_rate=trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                subsample=trial.suggest_float('subsample', 0.6, 1.0),
                colsample_bytree=trial.suggest_float('colsample_bytree', 0.6, 1.0),
                random_state=42, n_jobs=-1, use_label_encoder=False, eval_metric='logloss', verbosity=0
            )
            return cross_val_score(model, X, y, cv=skf, scoring='f1').mean()
        
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        best_model = XGBClassifier(**study.best_params, random_state=42, n_jobs=-1,
                                   use_label_encoder=False, eval_metric='logloss', verbosity=0)
        
    elif model_name == 'KNN':
        def objective(trial):
            model = KNeighborsClassifier(
                n_neighbors=trial.suggest_int('n_neighbors', 3, 21, step=2),
                weights=trial.suggest_categorical('weights', ['uniform', 'distance']),
                metric=trial.suggest_categorical('metric', ['euclidean', 'manhattan']),
                n_jobs=-1
            )
            return cross_val_score(model, X, y, cv=skf, scoring='f1').mean()
        
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        best_model = KNeighborsClassifier(**study.best_params, n_jobs=-1)
    
    else:
        raise ValueError(f"Bilinmeyen model: {model_name}")
    
    result = evaluate_single_model(best_model, X, y, skf)
    return result, study.best_params

# ============================================================
# SONUÇ KARŞILAŞTIRMA VE GÖRSELLEŞTİRME
# ============================================================

def compare_all_scenarios(all_scenario_results):
    """Tüm senaryo sonuçlarını karşılaştır"""
    print("\n" + "="*70)
    print("📊 TÜM SENARYOLAR KARŞILAŞTIRMASI")
    print("="*70)
    
    # Her senaryo için ortalama F1 hesapla
    scenario_summary = []
    for scenario_name, results in all_scenario_results.items():
        avg_f1 = np.mean([r['f1_mean'] for r in results.values() if r['f1_mean'] > 0])
        best_f1 = max([r['f1_mean'] for r in results.values() if r['f1_mean'] > 0])
        best_model = max(results.items(), key=lambda x: x[1]['f1_mean'])[0]
        
        scenario_summary.append({
            'Senaryo': scenario_name,
            'Ortalama F1': f"{avg_f1:.3f}",
            'En İyi F1': f"{best_f1:.3f}",
            'En İyi Model': best_model,
            'avg_f1': avg_f1,
            'best_f1': best_f1
        })
    
    df_summary = pd.DataFrame(scenario_summary)
    print("\n" + df_summary.to_string(index=False))
    
    # CSV kaydet
    csv_path = RESULTS_DIR / 'scenario_comparison.csv'
    df_summary.to_csv(csv_path, index=False)
    print(f"\n✓ scenario_comparison.csv kaydedildi")
    
    # Görselleştirme
    plot_scenario_comparison(scenario_summary)
    
    return df_summary

def plot_scenario_comparison(scenario_summary):
    """Senaryo karşılaştırma grafiği"""
    scenarios = [s['Senaryo'] for s in scenario_summary]
    avg_f1 = [s['avg_f1'] for s in scenario_summary]
    best_f1 = [s['best_f1'] for s in scenario_summary]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Ortalama F1
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(scenarios)))
    bars1 = axes[0].barh(scenarios, avg_f1, color=colors, edgecolor='black')
    axes[0].set_xlabel('Ortalama F1-Score', fontsize=12)
    axes[0].set_title('Senaryo Bazında Ortalama F1', fontsize=14, fontweight='bold')
    axes[0].set_xlim(0, 1)
    
    for bar, score in zip(bars1, avg_f1):
        axes[0].text(score + 0.01, bar.get_y() + bar.get_height()/2, 
                    f'{score:.3f}', va='center', fontsize=10)
    
    # En İyi F1
    bars2 = axes[1].barh(scenarios, best_f1, color=colors, edgecolor='black')
    axes[1].set_xlabel('En İyi F1-Score', fontsize=12)
    axes[1].set_title('Senaryo Bazında En İyi Model F1', fontsize=14, fontweight='bold')
    axes[1].set_xlim(0, 1)
    
    for bar, score in zip(bars2, best_f1):
        axes[1].text(score + 0.01, bar.get_y() + bar.get_height()/2, 
                    f'{score:.3f}', va='center', fontsize=10)
    
    plt.tight_layout()
    save_path = RESULTS_DIR / 'scenario_comparison.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()  # plt.show() yerine close kullan
    print(f"✓ scenario_comparison.png kaydedildi")

def plot_model_heatmap(all_scenario_results):
    """Model × Senaryo F1 heatmap"""
    # Veri hazırla
    scenarios = list(all_scenario_results.keys())
    models = list(all_scenario_results[scenarios[0]].keys())
    
    # Sadece ortak modelleri al
    common_models = ['Logistic Regression', 'Random Forest', 'SVM', 'Naive Bayes', 'XGBoost', 'KNN']
    
    f1_matrix = []
    for scenario in scenarios:
        row = []
        for model in common_models:
            if model in all_scenario_results[scenario]:
                row.append(all_scenario_results[scenario][model]['f1_mean'])
            else:
                row.append(np.nan)
        f1_matrix.append(row)
    
    # Heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(f1_matrix, annot=True, fmt='.3f', cmap='RdYlGn',
                xticklabels=common_models, yticklabels=scenarios,
                vmin=0.5, vmax=0.9)
    plt.title('Model × Senaryo F1-Score Heatmap', fontsize=14, fontweight='bold')
    plt.xlabel('Model')
    plt.ylabel('Senaryo')
    plt.tight_layout()
    
    save_path = RESULTS_DIR / 'model_scenario_heatmap.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()  # plt.show() yerine close kullan
    print(f"✓ model_scenario_heatmap.png kaydedildi")

# ============================================================
# ANA FONKSİYON
# ============================================================

def main():
    """Ana çalıştırma fonksiyonu"""
    print("\n" + "="*70)
    print("  UCI HEART DISEASE - REVİZE SENARYO KARŞILAŞTIRMASI")
    print("  6 Senaryo × 6 Model Analizi")
    print("="*70)
    
    # Sonuç klasörü oluştur
    create_results_folder()
    
    # Veri yükle
    data_path = Path(__file__).parent / 'data' / 'heart_disease_uci.csv'
    df = load_cleveland_data(data_path)
    
    # Tüm senaryo sonuçlarını sakla
    all_results = {}
    
    # ============================================================
    # SENARYO 0: BASELINE
    # ============================================================
    results_0, features_0, best_model, worst_model = scenario_0_baseline(df)
    all_results['S0: Baseline'] = results_0
    
    print(f"\n   📌 Senaryo 5 için seçilen modeller:")
    print(f"      🏆 En İyi: {best_model}")
    print(f"      📉 En Kötü: {worst_model}")
    
    # ============================================================
    # SENARYO 1: + PCA
    # ============================================================
    results_1 = scenario_1_pca(df)
    all_results['S1: + PCA'] = results_1
    
    # ============================================================
    # SENARYO 2: + FEATURE ENGINEERING
    # ============================================================
    results_2, features_2 = scenario_2_feature_engineering(df)
    all_results['S2: + FE'] = results_2
    
    # ============================================================
    # SENARYO 3: + SMOTE
    # ============================================================
    results_3 = scenario_3_smote(df)
    all_results['S3: + SMOTE'] = results_3
    
    # ============================================================
    # SENARYO 4: + OPTUNA
    # ============================================================
    results_4, best_params_4 = scenario_4_optuna(df, n_trials=30)
    all_results['S4: + Optuna'] = results_4
    
    # ============================================================
    # SENARYO 5: ALL COMBINED
    # ============================================================
    results_5, X_final, y_final, features_final = scenario_5_all_combined(
        df, best_model, worst_model, n_trials=50
    )
    all_results['S5: All Combined'] = results_5
    
    # ============================================================
    # KARŞILAŞTIRMA VE GÖRSELLEŞTİRME
    # ============================================================
    df_comparison = compare_all_scenarios(all_results)
    
    # Heatmap (Senaryo 0-4 için, 5 farklı yapıda)
    heatmap_results = {k: v for k, v in all_results.items() if 'S5' not in k}
    plot_model_heatmap(heatmap_results)
    
    # ============================================================
    # ÖZET
    # ============================================================
    print("\n" + "="*70)
    print("  ANALİZ TAMAMLANDI!")
    print(f"  Sonuçlar: {RESULTS_DIR}")
    print("  Oluşturulan dosyalar:")
    print("    - scenario_comparison.csv")
    print("    - scenario_comparison.png")
    print("    - model_scenario_heatmap.png")
    print("    - scenario_4_optuna_params.txt")
    print("="*70)
    
    return all_results, df_comparison

if __name__ == "__main__":
    main()
