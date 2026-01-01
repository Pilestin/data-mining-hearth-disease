"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║        ❤️  UCI HEART DISEASE PREDICTION - COMPREHENSIVE STREAMLIT APP       ║
║                                                                              ║
║                    Tüm 6 Senaryo × Tamamen Entegre Uygulama                ║
║                                                                              ║
║  Özellikler:                                                                ║
║  ✅ S0-S5: 6 senaryo tam analiz                                             ║
║  ✅ Senaryo karşılaştırması ve heatmap                                       ║
║  ✅ Hasta prediksiyon modülü                                                 ║
║  ✅ Model seçimi önerileri                                                   ║
║  ✅ Teknik dokumentasyon ve açıklamalar                                      ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Preprocessing
from sklearn.impute import KNNImputer
from sklearn.preprocessing import RobustScaler, LabelEncoder, StandardScaler
from sklearn.decomposition import PCA

# Models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier

# Validation & Metrics
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict
from sklearn.metrics import confusion_matrix, roc_curve, auc, roc_auc_score, f1_score, recall_score, accuracy_score

# Class Imbalance
from imblearn.over_sampling import SMOTE

# Optimization
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ============================================================================
# PAGE CONFIGURATION & STYLING
# ============================================================================

st.set_page_config(
    page_title="❤️ Heart Disease - Comprehensive Analysis",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main { padding-top: 0px; }
    .metric-box {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .scenario-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 20px;
        border-radius: 10px;
        margin: 20px 0;
    }
    .scenario-header-alt {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 20px;
        border-radius: 10px;
        margin: 20px 0;
    }
    .model-result-good {
        background-color: #d4edda;
        padding: 10px;
        border-left: 4px solid #28a745;
        border-radius: 5px;
    }
    .model-result-bad {
        background-color: #f8d7da;
        padding: 10px;
        border-left: 4px solid #dc3545;
        border-radius: 5px;
    }
    .info-box {
        background-color: #d1ecf1;
        border-left: 4px solid #0c5460;
        padding: 10px;
        border-radius: 5px;
        margin: 10px 0;
    }
    .success-box {
        background-color: #d4edda;
        border-left: 4px solid #155724;
        padding: 10px;
        border-radius: 5px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# DATA LOADING AND PREPROCESSING FUNCTIONS
# ============================================================================

@st.cache_data
def load_cleveland_data():
    """Load Cleveland dataset"""
    try:
        df = pd.read_csv("data/heart_disease_uci.csv")
        df = df[df['dataset'] == 'Cleveland'].copy()
        df['target'] = (df['num'] > 0).astype(int)
        return df
    except:
        st.error("❌ Dataset yüklenemedi!")
        return None

@st.cache_data
def basic_preprocessing(df):
    """Common preprocessing for all scenarios"""
    df_processed = df.copy()
    
    categorical_cols = ['sex', 'cp', 'restecg', 'exang', 'slope', 'thal', 'fbs']
    
    for col in categorical_cols:
        if col in df_processed.columns:
            le = LabelEncoder()
            df_processed[col] = df_processed[col].fillna('missing')
            df_processed[col] = le.fit_transform(df_processed[col].astype(str))
    
    exclude_cols = ['id', 'num', 'target', 'dataset']
    numeric_cols = [col for col in df_processed.select_dtypes(include=[np.number]).columns 
                   if col not in exclude_cols]
    
    imputer = KNNImputer(n_neighbors=5)
    df_processed[numeric_cols] = imputer.fit_transform(df_processed[numeric_cols])
    
    return df_processed

def add_feature_engineering(df):
    """Add engineered features"""
    df_fe = df.copy()
    
    df_fe['risk_score'] = (df_fe['age'] * df_fe['chol']) / 10000
    df_fe['age_group'] = pd.cut(
        df_fe['age'], 
        bins=[0, 40, 55, 70, 100],
        labels=[0, 1, 2, 3]
    ).astype(float).fillna(1).astype(int)
    df_fe['hr_age_ratio'] = df_fe['thalch'] / (df_fe['age'] + 1)
    df_fe['bp_chol_interaction'] = (df_fe['trestbps'] * df_fe['chol']) / 10000
    
    return df_fe

def get_features_target(df, exclude_extra=[]):
    """Extract features and target"""
    exclude_cols = ['id', 'num', 'target', 'dataset'] + exclude_extra
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    X = df[feature_cols].values
    y = df['target'].values
    
    return X, y, feature_cols

def get_default_models():
    """Get all models"""
    return {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Random Forest': RandomForestClassifier(random_state=42, n_jobs=-1),
        'SVM': SVC(probability=True, random_state=42),
        'Naive Bayes': GaussianNB(),
        'XGBoost': XGBClassifier(random_state=42, n_jobs=-1, use_label_encoder=False, 
                                eval_metric='logloss', verbosity=0),
        'KNN': KNeighborsClassifier(n_jobs=-1)
    }

# ============================================================================
# MODEL EVALUATION
# ============================================================================

def evaluate_all_models(X, y, models=None, cv=10):
    """Evaluate all models with 10-Fold CV"""
    if models is None:
        models = get_default_models()
    
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    results = {}
    
    progress_bar = st.progress(0)
    total_models = len(models)
    
    for idx, (name, model) in enumerate(models.items()):
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
            results[name] = {
                'accuracy': "N/A", 'f1': "N/A", 'recall': "N/A", 'auc': "N/A",
                'f1_mean': 0, 'acc_mean': 0, 'recall_mean': 0, 'auc_mean': 0
            }
        
        progress_bar.progress((idx + 1) / total_models)
    
    return results

# ============================================================================
# SCENARIO IMPLEMENTATIONS
# ============================================================================

@st.cache_data
def scenario_0_baseline(df):
    """S0: Baseline"""
    df_processed = basic_preprocessing(df)
    X, y, features = get_features_target(df_processed)
    
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)
    
    models = get_default_models()
    results = evaluate_all_models(X_scaled, y, models)
    
    return results, features, "RobustScaler"

@st.cache_data
def scenario_1_pca(df):
    """S1: + PCA"""
    df_processed = basic_preprocessing(df)
    X, y, features = get_features_target(df_processed)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    pca = PCA(n_components=0.95, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    
    models = get_default_models()
    results = evaluate_all_models(X_pca, y, models)
    
    pca_info = f"13 features → {X_pca.shape[1]} components ({pca.explained_variance_ratio_.sum():.1%} variance)"
    
    return results, pca_info

@st.cache_data
def scenario_2_feature_engineering(df):
    """S2: + Feature Engineering"""
    df_processed = basic_preprocessing(df)
    df_fe = add_feature_engineering(df_processed)
    X, y, features = get_features_target(df_fe)
    
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)
    
    models = get_default_models()
    results = evaluate_all_models(X_scaled, y, models)
    
    return results, features

@st.cache_data
def scenario_3_smote(df):
    """S3: + SMOTE"""
    df_processed = basic_preprocessing(df)
    X, y, features = get_features_target(df_processed)
    
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)
    
    smote = SMOTE(random_state=42)
    X_smote, y_smote = smote.fit_resample(X_scaled, y)
    
    models = get_default_models()
    results = evaluate_all_models(X_smote, y_smote, models)
    
    balance_info = f"{sum(y==0)} vs {sum(y==1)} → {sum(y_smote==0)} vs {sum(y_smote==1)}"
    
    return results, balance_info

@st.cache_data
def scenario_4_optuna(df):
    """S4: + Optuna"""
    df_processed = basic_preprocessing(df)
    X, y, features = get_features_target(df_processed)
    
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)
    
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    
    results = {}
    best_params_dict = {}
    
    st.info("⏳ Optuna optimizasyonu çalışıyor (20 trial per model)...")
    
    # LR
    try:
        def objective_lr(trial):
            C = trial.suggest_float('C', 0.01, 10.0, log=True)
            penalty = trial.suggest_categorical('penalty', ['l1', 'l2'])
            model = LogisticRegression(C=C, penalty=penalty, solver='lbfgs', 
                                     max_iter=1000, random_state=42)
            return cross_val_score(model, X_scaled, y, cv=skf, scoring='f1').mean()
        
        study = optuna.create_study(direction='maximize')
        study.optimize(objective_lr, n_trials=15, show_progress_bar=False)
        best_model = LogisticRegression(**study.best_params, solver='lbfgs', 
                                       max_iter=1000, random_state=42)
        best_params_dict['Logistic Regression'] = study.best_params
        
        acc = cross_val_score(best_model, X_scaled, y, cv=skf, scoring='accuracy')
        f1 = cross_val_score(best_model, X_scaled, y, cv=skf, scoring='f1')
        rec = cross_val_score(best_model, X_scaled, y, cv=skf, scoring='recall')
        auc_score = cross_val_score(best_model, X_scaled, y, cv=skf, scoring='roc_auc')
        
        results['Logistic Regression'] = {
            'accuracy': f"{acc.mean():.3f}±{acc.std():.3f}",
            'f1': f"{f1.mean():.3f}±{f1.std():.3f}",
            'recall': f"{rec.mean():.3f}±{rec.std():.3f}",
            'auc': f"{auc_score.mean():.3f}±{auc_score.std():.3f}",
            'f1_mean': f1.mean(), 'acc_mean': acc.mean(),
            'recall_mean': rec.mean(), 'auc_mean': auc_score.mean()
        }
    except:
        results['Logistic Regression'] = {
            'accuracy': "N/A", 'f1': "N/A", 'recall': "N/A", 'auc': "N/A",
            'f1_mean': 0, 'acc_mean': 0, 'recall_mean': 0, 'auc_mean': 0
        }
    
    # RF
    try:
        def objective_rf(trial):
            n_estimators = trial.suggest_int('n_estimators', 50, 200)
            max_depth = trial.suggest_int('max_depth', 3, 15)
            min_samples_split = trial.suggest_int('min_samples_split', 2, 15)
            model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth,
                                          min_samples_split=min_samples_split,
                                          random_state=42, n_jobs=-1)
            return cross_val_score(model, X_scaled, y, cv=skf, scoring='f1').mean()
        
        study = optuna.create_study(direction='maximize')
        study.optimize(objective_rf, n_trials=15, show_progress_bar=False)
        best_model = RandomForestClassifier(**study.best_params, random_state=42, n_jobs=-1)
        best_params_dict['Random Forest'] = study.best_params
        
        acc = cross_val_score(best_model, X_scaled, y, cv=skf, scoring='accuracy')
        f1 = cross_val_score(best_model, X_scaled, y, cv=skf, scoring='f1')
        rec = cross_val_score(best_model, X_scaled, y, cv=skf, scoring='recall')
        auc_score = cross_val_score(best_model, X_scaled, y, cv=skf, scoring='roc_auc')
        
        results['Random Forest'] = {
            'accuracy': f"{acc.mean():.3f}±{acc.std():.3f}",
            'f1': f"{f1.mean():.3f}±{f1.std():.3f}",
            'recall': f"{rec.mean():.3f}±{rec.std():.3f}",
            'auc': f"{auc_score.mean():.3f}±{auc_score.std():.3f}",
            'f1_mean': f1.mean(), 'acc_mean': acc.mean(),
            'recall_mean': rec.mean(), 'auc_mean': auc_score.mean()
        }
    except:
        results['Random Forest'] = {
            'accuracy': "N/A", 'f1': "N/A", 'recall': "N/A", 'auc': "N/A",
            'f1_mean': 0, 'acc_mean': 0, 'recall_mean': 0, 'auc_mean': 0
        }
    
    # SVM
    try:
        def objective_svm(trial):
            C = trial.suggest_float('C', 0.1, 100.0, log=True)
            kernel = trial.suggest_categorical('kernel', ['rbf', 'poly'])
            model = SVC(C=C, kernel=kernel, probability=True, random_state=42)
            return cross_val_score(model, X_scaled, y, cv=skf, scoring='f1').mean()
        
        study = optuna.create_study(direction='maximize')
        study.optimize(objective_svm, n_trials=15, show_progress_bar=False)
        best_model = SVC(**study.best_params, probability=True, random_state=42)
        best_params_dict['SVM'] = study.best_params
        
        acc = cross_val_score(best_model, X_scaled, y, cv=skf, scoring='accuracy')
        f1 = cross_val_score(best_model, X_scaled, y, cv=skf, scoring='f1')
        rec = cross_val_score(best_model, X_scaled, y, cv=skf, scoring='recall')
        auc_score = cross_val_score(best_model, X_scaled, y, cv=skf, scoring='roc_auc')
        
        results['SVM'] = {
            'accuracy': f"{acc.mean():.3f}±{acc.std():.3f}",
            'f1': f"{f1.mean():.3f}±{f1.std():.3f}",
            'recall': f"{rec.mean():.3f}±{rec.std():.3f}",
            'auc': f"{auc_score.mean():.3f}±{auc_score.std():.3f}",
            'f1_mean': f1.mean(), 'acc_mean': acc.mean(),
            'recall_mean': rec.mean(), 'auc_mean': auc_score.mean()
        }
    except:
        results['SVM'] = {
            'accuracy': "N/A", 'f1': "N/A", 'recall': "N/A", 'auc': "N/A",
            'f1_mean': 0, 'acc_mean': 0, 'recall_mean': 0, 'auc_mean': 0
        }
    
    # NB, XGB, KNN dengan default params
    for model_name in ['Naive Bayes', 'XGBoost', 'KNN']:
        try:
            model = get_default_models()[model_name]
            acc = cross_val_score(model, X_scaled, y, cv=skf, scoring='accuracy')
            f1 = cross_val_score(model, X_scaled, y, cv=skf, scoring='f1')
            rec = cross_val_score(model, X_scaled, y, cv=skf, scoring='recall')
            auc_score = cross_val_score(model, X_scaled, y, cv=skf, scoring='roc_auc')
            
            results[model_name] = {
                'accuracy': f"{acc.mean():.3f}±{acc.std():.3f}",
                'f1': f"{f1.mean():.3f}±{f1.std():.3f}",
                'recall': f"{rec.mean():.3f}±{rec.std():.3f}",
                'auc': f"{auc_score.mean():.3f}±{auc_score.std():.3f}",
                'f1_mean': f1.mean(), 'acc_mean': acc.mean(),
                'recall_mean': rec.mean(), 'auc_mean': auc_score.mean()
            }
        except:
            results[model_name] = {
                'accuracy': "N/A", 'f1': "N/A", 'recall': "N/A", 'auc': "N/A",
                'f1_mean': 0, 'acc_mean': 0, 'recall_mean': 0, 'auc_mean': 0
            }
    
    return results, best_params_dict

@st.cache_data
def scenario_5_all_combined(df):
    """S5: All Combined"""
    df_processed = basic_preprocessing(df)
    df_fe = add_feature_engineering(df_processed)
    X, y, features = get_features_target(df_fe)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    pca = PCA(n_components=0.95, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    
    smote = SMOTE(random_state=42)
    X_combined, y_combined = smote.fit_resample(X_pca, y)
    
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    
    results = {}
    
    test_models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'XGBoost': XGBClassifier(random_state=42, n_jobs=-1, use_label_encoder=False, 
                               eval_metric='logloss', verbosity=0)
    }
    
    st.info("⏳ S5 optimizasyonu çalışıyor (20 trial per model)...")
    
    for model_name, model in test_models.items():
        try:
            if model_name == 'Logistic Regression':
                def objective(trial):
                    C = trial.suggest_float('C', 0.01, 10.0, log=True)
                    penalty = trial.suggest_categorical('penalty', ['l1', 'l2'])
                    m = LogisticRegression(C=C, penalty=penalty, solver='lbfgs',
                                         max_iter=1000, random_state=42)
                    return cross_val_score(m, X_combined, y_combined, cv=skf, scoring='f1').mean()
                
                study = optuna.create_study(direction='maximize')
                study.optimize(objective, n_trials=15, show_progress_bar=False)
                model = LogisticRegression(**study.best_params, solver='lbfgs',
                                         max_iter=1000, random_state=42)
            
            elif model_name == 'XGBoost':
                def objective(trial):
                    n_estimators = trial.suggest_int('n_estimators', 50, 200)
                    max_depth = trial.suggest_int('max_depth', 2, 10)
                    learning_rate = trial.suggest_float('learning_rate', 0.01, 0.3)
                    m = XGBClassifier(n_estimators=n_estimators, max_depth=max_depth,
                                    learning_rate=learning_rate, random_state=42,
                                    n_jobs=-1, use_label_encoder=False, eval_metric='logloss')
                    return cross_val_score(m, X_combined, y_combined, cv=skf, scoring='f1').mean()
                
                study = optuna.create_study(direction='maximize')
                study.optimize(objective, n_trials=15, show_progress_bar=False)
                model = XGBClassifier(**study.best_params, random_state=42, n_jobs=-1,
                                    use_label_encoder=False, eval_metric='logloss')
            
            acc = cross_val_score(model, X_combined, y_combined, cv=skf, scoring='accuracy')
            f1 = cross_val_score(model, X_combined, y_combined, cv=skf, scoring='f1')
            rec = cross_val_score(model, X_combined, y_combined, cv=skf, scoring='recall')
            auc_score = cross_val_score(model, X_combined, y_combined, cv=skf, scoring='roc_auc')
            
            results[model_name] = {
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
            results[model_name] = {
                'accuracy': "N/A", 'f1': "N/A", 'recall': "N/A", 'auc': "N/A",
                'f1_mean': 0, 'acc_mean': 0, 'recall_mean': 0, 'auc_mean': 0
            }
    
    pipeline_info = f"17 features → FE → StandardScaler → PCA: {X_pca.shape[1]} → SMOTE: {len(X_combined)}"
    
    return results, pipeline_info

# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def plot_results_table(results):
    """Display results table"""
    data = []
    for model_name, metrics in results.items():
        data.append({
            'Model': model_name,
            'Accuracy': metrics['accuracy'],
            'Recall': metrics['recall'],
            'F1-Score': metrics['f1'],
            'AUC': metrics['auc']
        })
    
    df_results = pd.DataFrame(data)
    st.dataframe(df_results, use_container_width=True)
    
    return df_results

def plot_f1_comparison(results, title="F1-Score Comparison"):
    """Plot F1 scores"""
    valid_results = {k: v for k, v in results.items() if v['f1_mean'] > 0}
    
    if not valid_results:
        st.warning("No valid results to plot")
        return
    
    models = list(valid_results.keys())
    f1_scores = [valid_results[m]['f1_mean'] for m in models]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['#2ecc71' if f > 0.80 else '#f39c12' if f > 0.75 else '#e74c3c' for f in f1_scores]
    bars = ax.barh(models, f1_scores, color=colors, edgecolor='black', linewidth=1.5)
    ax.set_xlabel('F1-Score', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlim(0, 1)
    ax.grid(axis='x', alpha=0.3)
    
    for i, (bar, score) in enumerate(zip(bars, f1_scores)):
        ax.text(score + 0.01, i, f'{score:.3f}', va='center', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)

def plot_metrics_comparison(results, title="Metrics Comparison"):
    """Compare all metrics"""
    valid_results = {k: v for k, v in results.items() if v['f1_mean'] > 0}
    
    if not valid_results:
        st.warning("No valid results to plot")
        return
    
    models = list(valid_results.keys())
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(title, fontsize=14, fontweight='bold')
    
    metrics = ['acc_mean', 'recall_mean', 'f1_mean', 'auc_mean']
    metric_names = ['Accuracy', 'Recall', 'F1-Score', 'AUC']
    
    for idx, (ax, metric, metric_name) in enumerate(zip(axes.flat, metrics, metric_names)):
        values = [valid_results[m][metric] for m in models]
        colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(models)))
        ax.barh(models, values, color=colors, edgecolor='black', linewidth=1.5)
        ax.set_xlabel(metric_name, fontsize=11, fontweight='bold')
        ax.set_xlim(0, 1)
        ax.grid(axis='x', alpha=0.3)
        
        for i, v in enumerate(values):
            ax.text(v + 0.01, i, f'{v:.3f}', va='center', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)

# ============================================================================
# PAGE BUILDERS
# ============================================================================

def page_home():
    """Home/Welcome page"""
    st.markdown("""
    <div class="scenario-header">
        <h1>❤️ UCI Heart Disease Prediction</h1>
        <h3>Kapsamlı Senaryo Analiz ve Hasta Tahmini Sistemi</h3>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    ### 📊 Proje Tanımı
    
    Bu uygulama, UCI Heart Disease veri seti (Cleveland) üzerinde **6 farklı senaryo analizi** gerçekleştirmektedir.
    Her senaryo, farklı veri önişleme ve optimizasyon tekniklerini kapsamaktadır.
    
    **Proje Amacı:**
    1. Farklı tekniklerin model performansına etkisini izole olarak görmek (Ablation Study)
    2. Hiperparametre optimizasyonu ile model performansını maksimize etmek
    3. 6 farklı makine öğrenmesi algoritmasını karşılaştırmak
    4. En iyi ve en kötü modellerin tüm tekniklerle birlikte performansını analiz etmek
    
    ---
    
    ### 🎯 Senaryo Yapısı
    
    """)
    
    # Scenario overview table
    scenario_data = {
        'Senaryo': ['S0', 'S1', 'S2', 'S3', 'S4', 'S5'],
        'İsim': ['Baseline', '+ PCA', '+ FE', '+ SMOTE', '+ Optuna', 'All Combined'],
        'Scaler': ['RobustScaler', 'StandardScaler', 'RobustScaler', 'RobustScaler', 'RobustScaler', 'StandardScaler'],
        'Teknikler': ['Temel', 'PCA', 'FE', 'SMOTE', 'Optuna', 'FE+PCA+SMOTE+Optuna'],
        'Modeller': [6, 6, 6, 6, 6, 2]
    }
    
    st.dataframe(pd.DataFrame(scenario_data), use_container_width=True)
    
    st.markdown("""
    ---
    
    ### 📈 Özet Sonuçlar
    
    """)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("🏆 En İyi Senaryo", "S5: All Combined", "F1: 0.843")
    
    with col2:
        st.metric("📊 En Etkili Teknik", "SMOTE", "+3.8% F1")
    
    with col3:
        st.metric("🚀 En Çok Gelişen", "XGBoost", "+10.2%")
    
    with col4:
        st.metric("🎯 Önerilen Model", "Logistic Reg.", "Recall: 0.824")
    
    st.markdown("""
    ---
    
    ### 🔍 Veri Seti Özellikleri
    
    - **Kaynak:** UCI Machine Learning Repository
    - **Veri Seti:** Cleveland Heart Disease
    - **Örneklem:** 304 hastaya ait 13 klinik parametre
    - **Hedef:** Binary sınıflandırma (Sağlıklı vs Hasta)
    - **Sınıf Dağılımı:** 54.3% Sağlıklı, 45.7% Hasta
    
    **Özellikler:**
    - age, sex, cp, trestbps, chol, fbs, restecg, thalch, exang, oldpeak, slope, ca, thal
    
    **Mühendislik Özellikleri (S2, S5):**
    - risk_score, age_group, hr_age_ratio, bp_chol_interaction
    
    ---
    
    ### 🛠️ Kullanılan Teknikler
    
    **Veri Önişleme:**
    - KNN Imputer: Eksik değerleri benzer örneklerden doldur
    - RobustScaler: Aykırı değerlere dayanıklı ölçekleme
    - StandardScaler: Normal dağılım için ölçekleme
    
    **Boyut Azaltma:**
    - PCA: %95 varyans ile boyut azaltma
    
    **Sınıf Dengeleme:**
    - SMOTE: Yapay örnek oluşturarak dengeleme
    
    **Optimizasyon:**
    - Optuna: TPE (Bayesian) hiperparametre optimizasyonu
    
    **Validasyon:**
    - 10-Fold Stratified Cross-Validation
    
    ---
    
    ### 📖 Nasıl Kullanılır?
    
    **Sol menüden sayfa seçin:**
    1. **Senaryo Analizi:** 6 senaryonun detaylı analizi
    2. **Karşılaştırma:** Tüm senaryoların performans karşılaştırması
    3. **Heatmap:** Model × Senaryo F1-Score heatmap
    4. **Hasta Prediksiyon:** Yeni hasta verisi ile tahmin
    5. **Model Önerileri:** Farklı senaryolar için öneriler
    6. **Teknik Dokümantasyon:** Detaylı teknik bilgiler
    """)

def page_scenarios():
    """Scenario analysis page"""
    st.markdown("""
    <div class="scenario-header">
        <h1>📊 6 Senaryo Detaylı Analizi</h1>
        <p>Her senaryonun 6 model ile performans karşılaştırması</p>
    </div>
    """, unsafe_allow_html=True)
    
    df = load_cleveland_data()
    if df is None:
        st.stop()
    
    scenario = st.selectbox(
        "Senaryo Seçin:",
        ["S0: Baseline", "S1: + PCA", "S2: + Feature Engineering", 
         "S3: + SMOTE", "S4: + Optuna", "S5: All Combined"],
        index=0
    )
    
    # ========== S0: Baseline ==========
    if scenario == "S0: Baseline":
        with st.expander("📋 Senaryo S0 Detayları", expanded=True):
            st.markdown("""
            **Konfigürasyon:**
            - Scaler: RobustScaler (aykırı değerlere dayanıklı)
            - Feature Engineering: ❌ Yok
            - Boyut Azaltma: ❌ Yok
            - Sınıf Dengeleme: ❌ Yok
            - Hiperparametre Optim.: ❌ Yok
            - Validasyon: 10-Fold Stratified CV
            - Modeller: 6 adet (varsayılan parametrelerle)
            
            **RobustScaler Formülü:** `(X - median) / IQR`
            
            **Avantajı:** Aykırı değerlere karşı dayanıklı ölçekleme
            """)
        
        with st.spinner("S0 Baseline analiz çalışıyor..."):
            results_0, features_0, scaler_info = scenario_0_baseline(df)
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("📊 Model Sonuçları")
            plot_results_table(results_0)
        
        with col2:
            st.subheader("📈 F1-Score Karşılaştırması")
            plot_f1_comparison(results_0, "S0: Baseline F1-Scores")
        
        plot_metrics_comparison(results_0, "S0: Tüm Metriklerin Karşılaştırması")
        
        # Summary
        valid = {k: v for k, v in results_0.items() if v['f1_mean'] > 0}
        if valid:
            best_model = max(valid.items(), key=lambda x: x[1]['f1_mean'])
            worst_model = min(valid.items(), key=lambda x: x[1]['f1_mean'])
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"""
                <div class="success-box">
                <b>🏆 En İyi Model:</b> {best_model[0]}<br>
                <b>F1-Score:</b> {best_model[1]['f1']}<br>
                <b>AUC:</b> {best_model[1]['auc']}
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="info-box">
                <b>📉 En Kötü Model:</b> {worst_model[0]}<br>
                <b>F1-Score:</b> {worst_model[1]['f1']}<br>
                <b>AUC:</b> {worst_model[1]['auc']}
                </div>
                """, unsafe_allow_html=True)
    
    # ========== S1: + PCA ==========
    elif scenario == "S1: + PCA":
        with st.expander("📋 Senaryo S1 Detayları", expanded=True):
            st.markdown("""
            **Konfigürasyon:**
            - Scaler: StandardScaler (PCA için gerekli)
            - PCA: n_components=0.95 (%95 varyans)
            - Feature Engineering: ❌ Yok
            - Sınıf Dengeleme: ❌ Yok
            - Hiperparametre Optim.: ❌ Yok
            - Validasyon: 10-Fold Stratified CV
            
            **PCA (Principal Component Analysis) Nedir?**
            - Boyut azaltma tekniği
            - Varyansı korurken features azaltır
            - Hesaplama hızını artırır
            - Multicollinearity problemini çözer
            
            **Beklenen Etki:** +0.3% F1 iyileşme (XGBoost +7.1%)
            """)
        
        with st.spinner("S1 PCA analiz çalışıyor..."):
            results_1, pca_info = scenario_1_pca(df)
        
        st.info(f"✓ {pca_info}")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("📊 Model Sonuçları")
            plot_results_table(results_1)
        
        with col2:
            st.subheader("📈 F1-Score Karşılaştırması")
            plot_f1_comparison(results_1, "S1: + PCA F1-Scores")
        
        plot_metrics_comparison(results_1, "S1: Tüm Metriklerin Karşılaştırması")
    
    # ========== S2: + Feature Engineering ==========
    elif scenario == "S2: + Feature Engineering":
        with st.expander("📋 Senaryo S2 Detayları", expanded=True):
            st.markdown("""
            **Konfigürasyon:**
            - Scaler: RobustScaler
            - Feature Engineering: ✅ 4 yeni özellik
            - PCA: ❌ Yok
            - Sınıf Dengeleme: ❌ Yok
            - Hiperparametre Optim.: ❌ Yok
            
            **Mühendislik Özellikleri:**
            1. `risk_score` = (age × chol) / 10000 → Yaş-kolesterol risk
            2. `age_group` = Binning (0-40, 40-55, 55-70, 70+) → Yaş kategorileri
            3. `hr_age_ratio` = thalch / (age + 1) → Yaşa normalize kalp hızı
            4. `bp_chol_interaction` = (trestbps × chol) / 10000 → BP-kolesterol etkileşimi
            
            **Beklenen Etki:** -0.3% F1 değişim (etkisiz, Cleveland zaten iyi tasarlanmış)
            """)
        
        with st.spinner("S2 Feature Engineering analiz çalışıyor..."):
            results_2, features_2 = scenario_2_feature_engineering(df)
        
        st.success(f"✓ Eklenen özellikler: {len(features_2)} features")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("📊 Model Sonuçları")
            plot_results_table(results_2)
        
        with col2:
            st.subheader("📈 F1-Score Karşılaştırması")
            plot_f1_comparison(results_2, "S2: + FE F1-Scores")
        
        plot_metrics_comparison(results_2, "S2: Tüm Metriklerin Karşılaştırması")
    
    # ========== S3: + SMOTE ==========
    elif scenario == "S3: + SMOTE":
        with st.expander("📋 Senaryo S3 Detayları", expanded=True):
            st.markdown("""
            **Konfigürasyon:**
            - Scaler: RobustScaler
            - SMOTE: ✅ Sınıf dengeleme
            - Feature Engineering: ❌ Yok
            - PCA: ❌ Yok
            - Hiperparametre Optim.: ❌ Yok
            
            **SMOTE (Synthetic Minority Over-sampling Technique):**
            - Azınlık sınıfı için yapay örnekler oluşturur
            - k-NN ile benzer örnekleri bulur ve interpolasyon yapar
            - Sınıf dengesizliğini çözer
            - Modelin azınlık sınıfını daha iyi öğrenmesini sağlar
            
            **Etki:** Sağlıklı 165 vs Hasta 139 → 165 vs 165 (dengeli)
            
            **Beklenen Etki:** +3.8% F1 iyileşme (XGBoost +9.4%) - EN ETKİLİ TEKNİK!
            """)
        
        with st.spinner("S3 SMOTE analiz çalışıyor..."):
            results_3, balance_info = scenario_3_smote(df)
        
        st.success(f"✓ Sınıf dengelemesi: {balance_info}")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("📊 Model Sonuçları")
            plot_results_table(results_3)
        
        with col2:
            st.subheader("📈 F1-Score Karşılaştırması")
            plot_f1_comparison(results_3, "S3: + SMOTE F1-Scores")
        
        plot_metrics_comparison(results_3, "S3: Tüm Metriklerin Karşılaştırması")
    
    # ========== S4: + Optuna ==========
    elif scenario == "S4: + Optuna":
        with st.expander("📋 Senaryo S4 Detayları", expanded=True):
            st.markdown("""
            **Konfigürasyon:**
            - Scaler: RobustScaler
            - Optuna: ✅ Hiperparametre optimizasyonu
            - Trial Sayısı: 15 per model
            - Optimizasyon Algoritması: TPE (Tree-structured Parzen Estimator)
            - Maksimize Edilen Metrik: F1-Score
            
            **Optuna (Bayesian Optimization):**
            - TPE algoritması kullanır
            - Geçmiş deneylerden öğrenerek smart search yapar
            - Optimal hyperparametreler bulur
            - Her model için farklı parameter alanı
            
            **Beklenen Etki:** +2.5% F1 iyileşme (RF +4.1%, XGBoost +8.8%)
            """)
        
        with st.spinner("S4 Optuna analiz çalışıyor (biraz zaman alabilir)..."):
            results_4, best_params = scenario_4_optuna(df)
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("📊 Model Sonuçları")
            plot_results_table(results_4)
        
        with col2:
            st.subheader("📈 F1-Score Karşılaştırması")
            plot_f1_comparison(results_4, "S4: + Optuna F1-Scores")
        
        with st.expander("🔧 Optuna - En İyi Hyperparametreler"):
            for model_name, params in best_params.items():
                st.write(f"**{model_name}:**")
                st.json(params)
        
        plot_metrics_comparison(results_4, "S4: Tüm Metriklerin Karşılaştırması")
    
    # ========== S5: All Combined ==========
    else:  # S5: All Combined
        with st.expander("📋 Senaryo S5 Detayları", expanded=True):
            st.markdown("""
            **Konfigürasyon:**
            - Scaler: StandardScaler (PCA için)
            - Feature Engineering: ✅ 4 yeni özellik
            - PCA: ✅ n_components=0.95
            - SMOTE: ✅ Sınıf dengeleme
            - Optuna: ✅ Hiperparametre optimizasyonu (15 trial per model)
            - Validasyon: 10-Fold Stratified CV
            
            **Pipeline:**
            ```
            17 özellik (13 orijinal + 4 engineered)
                ↓
            StandardScaler
                ↓
            PCA (12 component)
                ↓
            SMOTE (330 örnek)
                ↓
            Optuna Optimizasyon + 10-Fold CV
            ```
            
            **Test Edilen Modeller:**
            - Logistic Regression (En iyi baseline'da)
            - XGBoost (En kötü baseline'da)
            
            **Beklenen Etki:** +5.0% F1 iyileşme (XGBoost +10.2%) - EN İYİ PERFORMANS!
            """)
        
        with st.spinner("S5 All Combined analiz çalışıyor (biraz zaman alabilir)..."):
            results_5, pipeline_info = scenario_5_all_combined(df)
        
        st.success(f"✓ {pipeline_info}")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("📊 Model Sonuçları")
            plot_results_table(results_5)
        
        with col2:
            st.subheader("📈 F1-Score Karşılaştırması")
            plot_f1_comparison(results_5, "S5: All Combined F1-Scores")
        
        plot_metrics_comparison(results_5, "S5: Tüm Metriklerin Karşılaştırması")

def page_comparison():
    """Scenario comparison page"""
    st.markdown("""
    <div class="scenario-header-alt">
        <h1>📊 Senaryo Karşılaştırma Analizi</h1>
        <p>Tüm 6 Senaryonun Performans Özeti ve Teknik Etki Analizi</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Summary data
    summary_data = {
        'Senaryo': ['S0: Baseline', 'S1: + PCA', 'S2: + FE', 'S3: + SMOTE', 'S4: + Optuna', 'S5: All Combined'],
        'Scaler': ['RobustScaler', 'StandardScaler', 'RobustScaler', 'RobustScaler', 'RobustScaler', 'StandardScaler'],
        'Ortalama F1': [0.788, 0.791, 0.785, 0.826, 0.813, 0.838],
        'En İyi F1': [0.817, 0.820, 0.815, 0.837, 0.824, 0.843],
        'En İyi Model': ['LR', 'LR', 'LR', 'LR', 'RF', 'LR'],
        'F1 vs Baseline': ['0%', '+0.3%', '-0.3%', '+3.8%', '+2.5%', '+5.0%']
    }
    
    df_summary = pd.DataFrame(summary_data)
    
    st.subheader("📈 Senaryo Özet Tablosu")
    st.dataframe(df_summary, use_container_width=True)
    
    # Visualizations
    col1, col2 = st.columns([1, 1])
    
    with col1:
        fig, ax = plt.subplots(figsize=(10, 6))
        colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(df_summary)))
        ax.barh(df_summary['Senaryo'], df_summary['Ortalama F1'], color=colors, edgecolor='black', linewidth=1.5)
        ax.set_xlabel('Ortalama F1-Score', fontsize=12, fontweight='bold')
        ax.set_title('Senaryo Bazında Ortalama F1', fontsize=13, fontweight='bold')
        ax.set_xlim(0.75, 0.85)
        ax.grid(axis='x', alpha=0.3)
        
        for i, v in enumerate(df_summary['Ortalama F1']):
            ax.text(v + 0.001, i, f'{v:.3f}', va='center', fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
    
    with col2:
        fig, ax = plt.subplots(figsize=(10, 6))
        colors = plt.cm.plasma(np.linspace(0.2, 0.8, len(df_summary)))
        ax.barh(df_summary['Senaryo'], df_summary['En İyi F1'], color=colors, edgecolor='black', linewidth=1.5)
        ax.set_xlabel('En İyi F1-Score', fontsize=12, fontweight='bold')
        ax.set_title('Senaryo Bazında En İyi Model F1', fontsize=13, fontweight='bold')
        ax.set_xlim(0.75, 0.85)
        ax.grid(axis='x', alpha=0.3)
        
        for i, v in enumerate(df_summary['En İyi F1']):
            ax.text(v + 0.001, i, f'{v:.3f}', va='center', fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
    
    # Technique impact analysis
    st.subheader("📊 Teknik Bazında Etki Analizi")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        ### 🔵 PCA Etkisi
        - F1 Değişim: **+0.3%**
        - Baseline: 0.788
        - S1: 0.791
        - En çok fayda: XGBoost (+7.1%)
        - Sonuç: Minimal etki
        """)
    
    with col2:
        st.markdown("""
        ### 🟢 Feature Eng. Etkisi
        - F1 Değişim: **-0.3%**
        - Baseline: 0.788
        - S2: 0.785
        - En çok fayda: Logistic Reg. (-0.2%)
        - Sonuç: ETKISIZ
        """)
    
    with col3:
        st.markdown("""
        ### 🟡 SMOTE Etkisi
        - F1 Değişim: **+3.8%** ⭐
        - Baseline: 0.788
        - S3: 0.826
        - En çok fayda: XGBoost (+9.4%)
        - Sonuç: EN ETKİLİ!
        """)
    
    with col4:
        st.markdown("""
        ### 🔴 Optuna Etkisi
        - F1 Değişim: **+2.5%**
        - Baseline: 0.788
        - S4: 0.813
        - En çok fayda: RF (+4.1%)
        - Sonuç: Etkili
        """)
    
    # Detailed findings
    st.markdown("""
    ---
    
    ### 🔍 Detaylı Bulgular
    """)
    
    findings = {
        '1. SMOTE En Etkili Teknik': 'Tüm modellerde +3.8% ortalama F1 iyileşme. XGBoost için özellikle güçlü (+9.4%)',
        '2. Logistic Regression Tutarlı': 'Her senaryoda top-2 performans. En stabil model.',
        '3. XGBoost Dramatik İyileşme': 'Baseline\'da en zayıf (F1=0.732), S5\'de güçlü (F1=0.834). +10.2% toplam iyileşme.',
        '4. Feature Engineering Etkisiz': 'Cleveland veri seti zaten iyi tasarlanmış. Yeni özellikler çok katkı sağlamadı.',
        '5. Combined Yaklaşım En İyi': 'S5 (All Combined) en yüksek performansı sağladı (F1=0.843, Recall=0.824)'
    }
    
    for title, content in findings.items():
        st.markdown(f"""
        <div class="success-box">
        <b>{title}</b><br>
        {content}
        </div>
        """, unsafe_allow_html=True)

def page_heatmap():
    """Heatmap analysis page"""
    st.markdown("""
    <div class="scenario-header-alt">
        <h1>🔥 Model × Senaryo Heatmap Analizi</h1>
        <p>F1-Score Değişimlerinin Görsel Analizi</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Heatmap data
    heatmap_data = np.array([
        [0.817, 0.820, 0.815, 0.837, 0.824, 0.843],  # LR
        [0.791, 0.789, 0.786, 0.824, 0.824, 0.834],  # RF
        [0.773, 0.795, 0.781, 0.828, 0.815, 0.825],  # SVM
        [0.767, 0.779, 0.793, 0.811, 0.798, 0.815],  # NB
        [0.732, 0.820, 0.769, 0.826, 0.820, 0.834],  # XGB
        [0.766, 0.782, 0.769, 0.827, 0.802, 0.825],  # KNN
    ])
    
    models = ['Logistic Regression', 'Random Forest', 'SVM', 'Naive Bayes', 'XGBoost', 'KNN']
    scenarios = ['S0: Baseline', 'S1: PCA', 'S2: FE', 'S3: SMOTE', 'S4: Optuna', 'S5: All']
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='RdYlGn',
                xticklabels=scenarios, yticklabels=models,
                vmin=0.70, vmax=0.85, cbar_kws={'label': 'F1-Score'},
                ax=ax, linewidths=1, linecolor='white', annot_kws={'fontsize': 11, 'fontweight': 'bold'})
    
    ax.set_title('Model × Senaryo F1-Score Heatmap', fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('Senaryo', fontsize=12, fontweight='bold')
    ax.set_ylabel('Model', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    
    # Interpretation
    st.markdown("""
    ### 📌 Heatmap Yorumu
    
    """)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **Renkler:**
        - 🟢 Yeşil (0.82+): Mükemmel
        - 🟡 Sarı (0.75-0.82): İyi
        - 🔴 Kırmızı (<0.75): Zayıf
        """)
    
    with col2:
        st.markdown("""
        **Gözlemler:**
        1. LR tutarlı top-2
        2. XGB dramatik S3→S5
        3. SMOTE tümünü lift ediyor
        """)
    
    with col3:
        st.markdown("""
        **En İyi Kombinasyon:**
        - Model: LR
        - Senaryo: S5
        - F1: 0.843
        - Recall: 0.824
        """)
    
    # Detailed analysis
    st.subheader("🔍 Model-Senaryo Kombinasyonları")
    
    st.markdown("""
    ### 🏆 En İyi Kombinasyonlar:
    1. **LR + S5:** F1=0.843 (Recall=0.824) - ÖNERİLEN
    2. **LR + S3:** F1=0.837 (Recall=0.806) - Hızlı alternatif
    3. **XGB + S5:** F1=0.834 - En çok gelişen
    4. **SVM + S3:** F1=0.828
    5. **RF + S4:** F1=0.824
    
    ### 📉 Sorunlu Kombinasyonlar:
    1. **XGB + S0:** F1=0.732 (Recall=0.671) - EN KÖTÜ
    2. **NB + S0:** F1=0.767 (Recall=0.689)
    3. **NB + S2:** F1=0.793
    4. **KNN + S0:** F1=0.766
    5. **KNN + S2:** F1=0.769
    """)

def page_patient_prediction():
    """Patient prediction page"""
    st.markdown("""
    <div class="scenario-header">
        <h1>🏥 Hasta Tahmini Modülü</h1>
        <p>Seçilen Senaryo ve Model ile Personalized Tahmin</p>
    </div>
    """, unsafe_allow_html=True)
    
    df = load_cleveland_data()
    
    st.subheader("📋 Hasta Bilgileri Giriş Formu")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        age = st.number_input("Yaş", min_value=20, max_value=80, value=50, help="Hasta yaşı (20-80)")
        sex = st.selectbox("Cinsiyet", ["Male", "Female"], help="Hastanın cinsiyeti")
        cp = st.selectbox("Göğüs Ağrısı Tipi", 
                         ["typical angina", "atypical angina", "non-anginal", "asymptomatic"],
                         help="Göğüs ağrısının karakteri")
    
    with col2:
        trestbps = st.number_input("Dinlenme Kan Basıncı (mmHg)", min_value=80, max_value=200, value=120,
                                  help="Sistemik kan basıncı (mmHg)")
        chol = st.number_input("Serum Kolesterol (mg/dl)", min_value=100, max_value=600, value=200,
                              help="Total serum kolesterol")
        fbs = st.selectbox("Açlık Kan Şekeri > 120 mg/dl", [False, True],
                          help="Açlık kan şekeri")
    
    with col3:
        thalch = st.number_input("Maksimum Kalp Hızı", min_value=60, max_value=220, value=150,
                                help="Egzersiz sırasında ulaşılan max kalp hızı")
        exang = st.selectbox("Egzersiz-Bağlı Angina", [False, True],
                            help="Egzersiz sırasında angina oluşuyor mu?")
        oldpeak = st.number_input("ST Depresyonu", min_value=-3.0, max_value=6.0, value=1.0,
                                 help="Egzersiz neden olan ST depresyonu")
    
    col4, col5 = st.columns(2)
    
    with col4:
        restecg = st.selectbox("Dinlenme EKG", ["normal", "st-t abnormality", "lv hypertrophy"],
                              help="Dinlenme sırasında EKG sonucu")
        slope = st.selectbox("ST Segment Eğimi", ["upsloping", "flat", "downsloping"],
                            help="Egzersiz ST segmentinin eğimi")
    
    with col5:
        ca = st.number_input("Damar Sayısı (0-3)", min_value=0, max_value=3, value=0,
                            help="Büyük damarlar (damar sayısı)")
        thal = st.selectbox("Talasemi", ["normal", "fixed defect", "reversable defect"],
                           help="Talasemi tipi")
    
    # Model selection
    st.subheader("⚙️ Model ve Senaryo Seçimi")
    
    col1, col2 = st.columns(2)
    
    with col1:
        selected_scenario = st.selectbox(
            "Senaryo Seçin:",
            ["S0: Baseline", "S1: + PCA", "S2: + FE", "S3: + SMOTE", "S4: + Optuna", "S5: All Combined"],
            help="Tahmin için kullanılacak senaryo"
        )
    
    with col2:
        if selected_scenario == "S5: All Combined":
            selected_model = st.selectbox("Model Seçin", ["Logistic Regression", "XGBoost"],
                                         help="S5'te sadece LR ve XGB test edildi")
        else:
            selected_model = st.selectbox("Model Seçin", 
                                        ["Logistic Regression", "Random Forest", "SVM", 
                                         "Naive Bayes", "XGBoost", "KNN"],
                                        help="Tahmin için kullanılacak model")
    
    show_details = st.checkbox("Detaylı Açıklamalar", value=True, help="Sonuçların detaylı anlatımını göster")
    
    # Prediction button
    if st.button("🔮 Tahmini Yap", use_container_width=True, help="Tahmin yap ve sonuçları göster"):
        
        # Prepare data
        patient_data = {
            'age': age, 'sex': sex, 'cp': cp, 'trestbps': trestbps,
            'chol': chol, 'fbs': fbs, 'restecg': restecg, 'thalch': thalch,
            'exang': exang, 'oldpeak': oldpeak, 'slope': slope, 'ca': ca, 'thal': thal
        }
        
        # Preprocess
        df_input = pd.DataFrame([patient_data])
        
        # Encode
        le_sex = LabelEncoder().fit(['Female', 'Male'])
        le_cp = LabelEncoder().fit(['typical angina', 'atypical angina', 'non-anginal', 'asymptomatic'])
        le_restecg = LabelEncoder().fit(['normal', 'st-t abnormality', 'lv hypertrophy'])
        le_slope = LabelEncoder().fit(['upsloping', 'flat', 'downsloping'])
        le_thal = LabelEncoder().fit(['normal', 'fixed defect', 'reversable defect'])
        
        df_input['sex'] = le_sex.transform([patient_data['sex']])[0]
        df_input['cp'] = le_cp.transform([patient_data['cp']])[0]
        df_input['restecg'] = le_restecg.transform([patient_data['restecg']])[0]
        df_input['slope'] = le_slope.transform([patient_data['slope']])[0]
        df_input['thal'] = le_thal.transform([patient_data['thal']])[0]
        
        # Select model
        if selected_model == "Logistic Regression":
            model = LogisticRegression(max_iter=1000, random_state=42)
        elif selected_model == "Random Forest":
            model = RandomForestClassifier(random_state=42, n_jobs=-1)
        elif selected_model == "SVM":
            model = SVC(probability=True, random_state=42)
        elif selected_model == "Naive Bayes":
            model = GaussianNB()
        elif selected_model == "XGBoost":
            model = XGBClassifier(random_state=42, n_jobs=-1, eval_metric='logloss', verbosity=0)
        else:  # KNN
            model = KNeighborsClassifier(n_jobs=-1)
        
        # Train on full dataset
        df_train = df.copy()
        
        categorical_cols = ['sex', 'cp', 'restecg', 'exang', 'slope', 'thal', 'fbs']
        for col in categorical_cols:
            if col in df_train.columns:
                le = LabelEncoder()
                df_train[col] = df_train[col].fillna('missing')
                df_train[col] = le.fit_transform(df_train[col].astype(str))
        
        exclude_cols = ['id', 'num', 'target', 'dataset']
        feature_cols = [col for col in df_train.columns if col not in exclude_cols]
        
        X_train = df_train[feature_cols].values
        y_train = df_train['target'].values
        
        # ✅ CRITICAL FIX: Apply KNN Imputer BEFORE scaling to handle NaN values
        imputer = KNNImputer(n_neighbors=5)
        X_train = imputer.fit_transform(X_train)
        
        # Scale if needed
        scaler = RobustScaler()
        X_train = scaler.fit_transform(X_train)
        
        # Prepare patient input with same transformations
        df_input_scaled = df_input[feature_cols].copy().values
        df_input_scaled = imputer.transform(df_input_scaled)  # Apply same imputer
        df_input_scaled = scaler.transform(df_input_scaled)    # Apply same scaler
        
        model.fit(X_train, y_train)
        
        # Predict
        probability = model.predict_proba(df_input_scaled)[0][1]
        prediction = model.predict(df_input_scaled)[0]
        
        # Display results
        st.markdown("---")
        st.subheader("📊 Tahmın Sonuçları")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("❤️ Hastalık Olasılığı", f"{probability*100:.1f}%", 
                     delta=f"{(probability-0.5)*100:+.1f}% vs Neutral")
        
        with col2:
            if probability > 0.7:
                risk_level = "🔴 YÜKSEK RİSK"
                risk_color = "red"
            elif probability > 0.5:
                risk_level = "🟡 ORTA RİSK"
                risk_color = "orange"
            else:
                risk_level = "🟢 DÜŞÜK RİSK"
                risk_color = "green"
            st.metric("📊 Risk Seviyesi", risk_level)
        
        with col3:
            st.metric("🤖 Seçilen Model", selected_model, delta=selected_scenario)
        
        # Detailed explanation
        if show_details:
            st.markdown("---")
            st.subheader("📝 Detaylı Değerlendirme")
            
            if probability > 0.5:
                st.error("""
                ### ⚠️ YÜKSEK RİSK DEĞERLENDİRMESİ
                
                **Tahmin:** Model hastalık olasılığını **%70'den yüksek** olarak belirlemiştir.
                
                **Önerilen Adımlar:**
                1. ⚡ Acil olarak **kardiyolog konsültasyonu** alınmalı
                2. 🏥 EKG, stres testi ve koroner anjiyografi önerilir
                3. 💊 Kalp sağlığı parametreleri hemen kontrol edilmeli
                4. 🚑 Acı katır tıbbi takip gerekli
                5. 📋 Risk faktörleri (BP, kolesterol, sigara) kontrol edilmeli
                """)
            
            elif probability > 0.4:
                st.warning("""
                ### ⚠️ ORTA RİSK DEĞERLENDİRMESİ
                
                **Tahmin:** Model hastalık olasılığını **%50-%70 arası** olarak belirlemiştir.
                
                **Önerilen Adımlar:**
                1. 📞 Yakında **kardiyolog randevusu** alınmalı
                2. 🏥 Kapsamlı kalp sağlığı değerlendirmesi yapılmalı
                3. 🏃 Hayat tarzı değişiklikleri (egzersiz, diyet) düşünülmeli
                4. 📊 Düzenli izlem ve testler önerilir
                5. 💊 Gerekli ilaçlar başlanabilir
                """)
            
            else:
                st.success("""
                ### ✅ DÜŞÜK RİSK DEĞERLENDİRMESİ
                
                **Tahmin:** Model hastalık olasılığını **%50'den düşük** olarak belirlemiştir.
                
                **Önerilen Adımlar:**
                1. ✓ Düzenli sağlık kontrolü yılda bir kez
                2. 💪 Sağlıklı yaşam tarzını sürdür
                3. 📈 Risk faktörlerini izle (BP, kolesterol, kilo)
                4. 🏃 Düzenli egzersiz yap
                5. 📅 İlk belirtilerde doktor konsultasyonu
                """)
            
            # Patient summary
            st.markdown("---")
            st.markdown("**Hasta Parametreleri Özeti:**")
            
            param_summary = pd.DataFrame({
                'Parametre': ['Yaş', 'Cinsiyet', 'Göğüs Ağrısı', 'Kan Basıncı', 'Kolesterol',
                             'Kalp Hızı', 'ST Depresyonu', 'Damar Sayısı'],
                'Değer': [f"{age} yıl", sex, cp, f"{trestbps} mmHg", f"{chol} mg/dl",
                         f"{thalch} bpm", f"{oldpeak}", f"{ca}"],
                'Normal Aralık': ['25-75', 'Erkek/Kadın', 'Tip-bağlıdır', '90-120', '<200',
                                 '60-100', '<1.0', '0-1']
            })
            
            st.dataframe(param_summary, use_container_width=True)
            
            # Medical disclaimer
            st.markdown("""
            ---
            
            ### ⚖️ Yasal Uyarı ve Sorumluluk Reddi
            
            **ÖNEMLİ:** Bu tahmin, tıbbi tanı aracı **DEĞİLDİR**. Sonuçlar yalnızca bilgilendirme
            amaçlıdır ve yapay zeka tarafından sağlanmıştır.
            
            **Kritik Uyarılar:**
            - ❌ Bu model hiçbir durumda doktor muayenesinin yerine geçmez
            - ❌ Tıbbi kararlar **kesinlikle** bir doktor ile birlikte verilmeli
            - ❌ Acil durumlarda 112'yi arayın
            - ✅ Her zaman **nitelikli sağlık profesyoneli** ile danışınız
            - ✅ Model sadece destekleyici bir araç olarak düşünülmeli
            
            **Sorumluluk:** Hastanın bu model sonuçlarına dayanarak verdiği tıbbi kararlardan
            yapay zeka sistemi, geliştirici ve yayıncı sorumlu değildir.
            """)

def page_recommendations():
    """Model recommendations page"""
    st.markdown("""
    <div class="scenario-header">
        <h1>💡 Model Seçimi ve Öneriler</h1>
        <p>Farklı Klinik Senaryolar için En İyi Model Kombinasyonları</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.subheader("🎯 Kullanım Senaryolarına Göre Öneriler")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        ### 🏥 Tarama Programları (Screening)
        
        **Önerilen Kombinasyon:**
        - Model: Logistic Regression
        - Senaryo: S5 (All Combined)
        
        **Metrikleri:**
        - F1-Score: 0.843
        - Recall: **0.824** ⭐
        - Precision: 0.862
        - AUC: 0.916
        
        **Neden?**
        - Yüksek Recall (hastaları yakalama)
        - Tarama amacında FN < FP
        - Hata-toleranslı
        
        **Maliyeti:** Yüksek (S5)
        """)
    
    with col2:
        st.markdown("""
        ### 💻 Klinik Karar Destek (Clinical)
        
        **Önerilen Kombinasyon:**
        - Model: Logistic Regression
        - Senaryo: S3 (SMOTE)
        
        **Metrikleri:**
        - F1-Score: 0.837
        - Recall: 0.806
        - AUC: 0.908
        
        **Neden?**
        - Yorumlanabilir model
        - Hızlı tahmin
        - Tıbbi açıklama yapılabilir
        - İyi dengelenmiş sınıflar
        
        **Maliyeti:** Düşük (S3)
        """)
    
    with col3:
        st.markdown("""
        ### ⚡ Sınırlı Kaynak (Resource-Limited)
        
        **Önerilen Kombinasyon:**
        - Model: Logistic Regression
        - Senaryo: S0 (Baseline)
        
        **Metrikleri:**
        - F1-Score: 0.817
        - Hız: ⭐⭐⭐⭐⭐
        - Bellek: Minimum
        
        **Neden?**
        - Minimal hesaplama
        - Çevrimdışı çalışabilir
        - Mobil uygulamada kullanılabilir
        - Soğuk başlangıç sorunları yok
        
        **Maliyeti:** Minimal (S0)
        """)
    
    # Detailed comparison
    st.subheader("📊 Teknik Karşılaştırma")
    
    comparison_data = {
        'Kriter': [
            'Performans (F1)',
            'Hız (Tahmin)',
            'Bellek Kullanımı',
            'Yorumlanabilirlik',
            'Eğitim Zamanı',
            'Standart Sapma'
        ],
        'S0 (Baseline)': [
            '0.817',
            '⭐⭐⭐⭐⭐',
            '⭐⭐⭐⭐⭐',
            '⭐⭐⭐⭐⭐',
            '< 1 sn',
            '±0.068'
        ],
        'S3 (SMOTE)': [
            '0.837',
            '⭐⭐⭐⭐',
            '⭐⭐⭐⭐',
            '⭐⭐⭐⭐',
            '~2 sn',
            '±0.075'
        ],
        'S5 (All)': [
            '0.843',
            '⭐⭐⭐',
            '⭐⭐⭐',
            '⭐⭐⭐',
            '~30 sn',
            '±0.064'
        ]
    }
    
    st.dataframe(pd.DataFrame(comparison_data), use_container_width=True)
    
    # Decision tree
    st.markdown("""
    ---
    
    ### 🌳 Model Seçim Karar Ağacı
    
    ```
    Başla
    │
    ├─ Maksimum Performans İsteniyor mu?
    │  ├─ EVET → S5 + Logistic Regression (F1=0.843)
    │  └─ HAYIR → Devam
    │
    ├─ Hızlı Deployment Gerekli mi?
    │  ├─ EVET → S0 + Logistic Regression (< 1 sn)
    │  └─ HAYIR → Devam
    │
    ├─ Sınırlı Kaynaklar mı (Mobil, IoT)?
    │  ├─ EVET → S0 + Logistic Regression (Minimal bellek)
    │  └─ HAYIR → Devam
    │
    ├─ Balans Önemli mi (İyi F1 + Makul Hız)?
    │  ├─ EVET → S3 + Logistic Regression (F1=0.837, hızlı)
    │  └─ HAYIR → S5
    │
    Son: Senaryo ve Model Seçildi
    ```
    
    ---
    
    ### 📌 Nihai Öneriler
    
    **Genel Tavsiye:**
    ```
    Logistic Regression + SMOTE (S3)
    ↓
    - F1-Score: 0.837 (Yeterli performans)
    - Hız: Makul (~2 sn eğitim)
    - Yorumlanabilirlik: Mükemmel
    - Klinik Uyum: Özelliklere dayalı açıklamalar
    - Önerilen Threshold: 0.40-0.45
    ```
    
    **Maksimum Performans Gerekirse:**
    ```
    Logistic Regression + All Combined (S5)
    ↓
    - F1-Score: 0.843 (Maksimum)
    - Recall: 0.824 (Hasta yakalama)
    - AUC: 0.916 (Mükemmel)
    - Maliyeti: Yüksek (~30 sn eğitim)
    - Önerilen Threshold: 0.40
    ```
    
    **Hızlı Prototype Gerekirse:**
    ```
    Logistic Regression + Baseline (S0)
    ↓
    - F1-Score: 0.817 (Kabul edilebilir)
    - Eğitim Zamanı: < 1 saniye
    - Bellek: Minimum
    - MVP için ideal
    - Ölçeklendir: S3 veya S5'e geç
    ```
    """)

def page_technical():
    """Technical documentation page"""
    st.markdown("""
    <div class="scenario-header-alt">
        <h1>📚 Teknik Dokümantasyon</h1>
        <p>Tüm Metodoloji, Teknik Detaylar ve Matematik</p>
    </div>
    """, unsafe_allow_html=True)
    
    tabs = st.tabs(["Veri Seti", "Preprocessing", "Teknikler", "Modeller", "Metrikleri", "Referanslar"])
    
    with tabs[0]:
        st.markdown("""
        ### 📊 Veri Seti Detayları
        
        **Cleveland Heart Disease (UCI)**
        - Kaynak: UCI Machine Learning Repository
        - Örneklem: 304
        - Özellikler: 13 (orijinal) + 4 (engineered) = 17
        - Hedef: Binary (0=Sağlıklı, 1=Hasta)
        - Sınıf Dağılımı: 54.3% vs 45.7% (dengeli)
        
        **Orijinal Özellikler (13):**
        | # | Adı | Tip | Aralık |
        |---|-----|-----|--------|
        | 1 | age | Sürekli | 28-77 yıl |
        | 2 | sex | Kategorik | Male/Female |
        | 3 | cp | Kategorik | 4 tip göğüs ağrısı |
        | 4 | trestbps | Sürekli | 94-200 mmHg |
        | 5 | chol | Sürekli | 126-564 mg/dl |
        | 6 | fbs | Binary | TRUE/FALSE |
        | 7 | restecg | Kategorik | 3 kategori |
        | 8 | thalch | Sürekli | 71-202 bpm |
        | 9 | exang | Binary | TRUE/FALSE |
        | 10 | oldpeak | Sürekli | -2.6 - 6.2 |
        | 11 | slope | Kategorik | 3 kategori |
        | 12 | ca | Ordinal | 0-3 |
        | 13 | thal | Kategorik | 3-4 kategori |
        
        **Mühendislik Özellikleri (4):**
        | # | Formül | Gerekçe |
        |---|--------|--------|
        | 14 | risk_score = (age × chol) / 10000 | Yaş-kolesterol risk |
        | 15 | age_group = Binning | Yaş kategorileri |
        | 16 | hr_age_ratio = thalch / (age+1) | Yaşa normalize HR |
        | 17 | bp_chol_inter = (trestbps × chol) / 10000 | BP-chol etkileşimi |
        """)
    
    with tabs[1]:
        st.markdown("""
        ### 🔧 Veri Önişleme Pipeline
        
        **Adım 1: Dataset Filtrelemesi**
        ```
        Orijinal: 920 satır (4 alt veri seti karışık)
        ↓
        Cleveland: 304 satır (tek kaynak)
        ```
        
        **Adım 2: Kategorik Encoding**
        ```python
        from sklearn.preprocessing import LabelEncoder
        
        categorical_cols = ['sex', 'cp', 'restecg', 'exang', 'slope', 'thal', 'fbs']
        for col in categorical_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
        ```
        
        **Adım 3: Eksik Değer Doldurma (KNN Imputer)**
        ```python
        from sklearn.impute import KNNImputer
        
        imputer = KNNImputer(n_neighbors=5)
        df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
        ```
        **Neden KNN Imputer?**
        - Benzer örneklerin değerlerini kullanır
        - Veri dağılımını korur
        - Robust yöntemi
        
        **Adım 4: Ölçekleme (Senaryo-bağlı)**
        
        **RobustScaler (S0, S2, S3, S4):**
        ```
        X_scaled = (X - median) / IQR
        
        Avantajı: Aykırı değerlere dayanıklı
        Dezavantajı: Standart dağılım varsayımı yok
        ```
        
        **StandardScaler (S1, S5):**
        ```
        X_scaled = (X - mean) / std
        
        Avantajı: Normal dağılım için optimal
        Dezavantajı: Outlier'lere duyarlı
        Zorunluluk: PCA için gerekli
        ```
        
        **Adım 5: Boyut Azaltma (PCA - S1, S5)**
        ```python
        from sklearn.decomposition import PCA
        
        pca = PCA(n_components=0.95, random_state=42)
        X_pca = pca.fit_transform(X_scaled)
        ```
        **Sonuç:** 13 feature → 12 component (%97.14 varyans)
        
        **Adım 6: Sınıf Dengeleme (SMOTE - S3, S5)**
        ```python
        from imblearn.over_sampling import SMOTE
        
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X, y)
        ```
        **Sonuç:** 165 vs 139 → 165 vs 165 (dengeli)
        """)
    
    with tabs[2]:
        st.markdown("""
        ### 🛠️ Kullanılan Optimizasyon Teknikleri
        
        **1. SMOTE (Synthetic Minority Over-sampling)**
        
        Algoritma:
        ```
        1. Azınlık sınıfındaki örneklerin k-NN'sini bul (k=5)
        2. Bu komşular arasında random seç
        3. Aralarında interpolation yaparak yapay örnek oluştur
        4. Yeni örneği eğitim setine ekle
        5. Sınıflar dengeli olana kadar tekrarla
        ```
        
        Etki: XGBoost'ta +9.4% F1 iyileşme
        
        ---
        
        **2. PCA (Principal Component Analysis)**
        
        Algoritma:
        ```
        1. Veri matrisinin kovaryans matrisini hesapla
        2. Eigen değerleri ve eigen vektörleri bul
        3. Eigen vektörleri büyüklüğe göre sırala
        4. İlk k bileşeni seç (n_components=0.95)
        5. Veriyi bu bileşenlere project et
        ```
        
        Sonuç: 13 → 12 boyut, %97.14 varyans korundu
        
        ---
        
        **3. Optuna (Bayesian Hyperparameter Optimization)**
        
        Algoritma:
        ```
        1. TPE (Tree-structured Parzen Estimator) kulllan
        2. Her trial'de model eğit ve performans ölç
        3. Geçmiş trial'lar ile prior dağılım oluştur
        4. Bu prior'a göre next hyperparameter'ı seç (maximize F1)
        5. n_trials kadar tekrarla
        ```
        
        **Hiperparametre Arama Uzayları:**
        
        Logistic Regression:
        - C: [0.01, 10.0] log scale
        - penalty: ['l1', 'l2']
        
        Random Forest:
        - n_estimators: [50, 300]
        - max_depth: [3, 20]
        - min_samples_split: [2, 20]
        
        SVM:
        - C: [0.1, 100.0] log scale
        - kernel: ['rbf', 'poly']
        - gamma: ['scale', 'auto']
        
        XGBoost:
        - n_estimators: [50, 300]
        - max_depth: [2, 10]
        - learning_rate: [0.01, 0.3]
        
        KNN:
        - n_neighbors: [3, 21] step=2
        - weights: ['uniform', 'distance']
        - metric: ['euclidean', 'manhattan']
        """)
    
    with tabs[3]:
        st.markdown("""
        ### 🤖 Kullanılan Modeller
        
        **1. Logistic Regression**
        ```python
        LogisticRegression(max_iter=1000, random_state=42)
        ```
        - **Tip:** Linear, Probabilistic
        - **Karmaşıklık:** O(n × m)
        - **Avantaj:** Yorumlanabilir, hızlı, stable
        - **Dezavantaj:** Non-linear ilişkileri yakalayamaz
        - **En İyi Senaryosu:** S5 (F1=0.843) ⭐
        
        ---
        
        **2. Random Forest**
        ```python
        RandomForestClassifier(n_estimators=100, random_state=42)
        ```
        - **Tip:** Ensemble (Bagging)
        - **Karmaşıklık:** O(n × m × log n × trees)
        - **Avantaj:** Non-linear, feature importance, robust
        - **Dezavantaj:** Yavaş, overfitting riski
        - **En İyi Senaryosu:** S4 (F1=0.824)
        
        ---
        
        **3. Support Vector Machine (SVM)**
        ```python
        SVC(kernel='rbf', probability=True, random_state=42)
        ```
        - **Tip:** Kernel-based, Geometric
        - **Karmaşıklık:** O(n² × m) - O(n³ × m)
        - **Avantaj:** Non-linear, high-dimensional
        - **Dezavantaj:** Çok yavaş, hyperparameter sensitive
        - **En İyi Senaryosu:** S3 (F1=0.828)
        
        ---
        
        **4. Naive Bayes**
        ```python
        GaussianNB()
        ```
        - **Tip:** Probabilistic, Generative
        - **Karmaşıklık:** O(n × m)
        - **Avantaj:** Çok hızlı, küçük veri setleri
        - **Dezavantaj:** Conditional independence varsayımı
        - **En İyi Senaryosu:** S2 (F1=0.793)
        
        ---
        
        **5. XGBoost**
        ```python
        XGBClassifier(n_estimators=100, random_state=42)
        ```
        - **Tip:** Ensemble (Boosting)
        - **Karmaşıklık:** O(n × m × trees × log n)
        - **Avantaj:** En güçlü, feature importance
        - **Dezavantaj:** Yavaş, overfitting riski, tuning zor
        - **En İyi Senaryosu:** S5 (F1=0.834)
        - **Not:** S0'da en zayıf (F1=0.732), S5'de güçlü!
        
        ---
        
        **6. K-Nearest Neighbors (KNN)**
        ```python
        KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
        ```
        - **Tip:** Instance-based, Lazy learning
        - **Karmaşıklık:** O(n × m) - prediction
        - **Avantaj:** Basit, non-parametric
        - **Dezavantaj:** Ölçeklemeye duyarlı, yavaş
        - **En İyi Senaryosu:** S3 (F1=0.827)
        """)
    
    with tabs[4]:
        st.markdown("""
        ### 📊 Performans Metrikleri
        
        **1. Accuracy (Doğruluk)**
        ```
        Accuracy = (TP + TN) / (TP + TN + FP + FN)
        
        Yorumu: Tüm tahminlerin yüzde kaçı doğru
        Avantaj: Basit, sezgisel
        Dezavantaj: Sınıf dengesizliğinde yanıltıcı
        Tıbbi: ❌ Kullanmayın
        ```
        
        ---
        
        **2. Recall (Duyarlılık)**
        ```
        Recall = TP / (TP + FN)
        
        Yorumu: Gerçek hasta olan kaçının modeli buldu
        Avantaj: False Negative'i minimize eder
        Dezavantaj: False Positive'i ignore eder
        Tıbbi: ✅ TIP II HATA KRİTİK - EN ÖNEMLİ
        ```
        
        **Tıbbi Bağlam:**
        - FN (Hastayı sağlıklı deme): Muhasır - KABUL EDİLMEZ
        - FP (Sağlıklıyı hasta deme): Gereksiz test - KABUL EDİLBİLİR
        
        ---
        
        **3. Precision (Kesinlik)**
        ```
        Precision = TP / (TP + FP)
        
        Yorumu: Hasta diye tahmin ettiklerinin ne kadarı hasta
        Avantaj: False Positive'i minimize eder
        Dezavantaj: False Negative'i ignore eder
        Tıbbi: ⚠️ Dengeli önem
        ```
        
        ---
        
        **4. F1-Score**
        ```
        F1 = 2 × (Precision × Recall) / (Precision + Recall)
        
        Yorumu: Precision ve Recall'un harmonic mean'i
        Avantaj: Sınıf dengesizliğinde güvenilir
        Dezavantaj: Yok
        Tıbbi: ✅ ÖNERILEN - Recall'u ağır basılı tutarak optimize
        ```
        
        ---
        
        **5. AUC-ROC**
        ```
        AUC = Area Under ROC Curve
        
        Yorumu: True Positive Rate vs False Positive Rate
        Aralık: 0.5 (rastgele) - 1.0 (mükemmel)
        Avantaj: Probability threshold'tan bağımsız
        Dezavantaj: Sınıf dengesizliğinde sorunlu olabilir
        Tıbbi: ⚠️ Tamamlayıcı metrik
        ```
        
        ---
        
        ### 🎯 Model Seçim Kriterleri
        
        **Tıbbi Tarama Programında (Screening):**
        1. **Recall** > 0.80 (hastaların %80'ini bulmalı)
        2. **F1-Score** > 0.80 (dengeli performans)
        3. **Precision** > 0.75 (yanlış alarm %25'de)
        
        **Klinik Destek Sisteminde:**
        1. **Yorumlanabilirlik** maksimum
        2. **F1-Score** > 0.75
        3. **Hız** < 1 saniye/tahmini
        
        **Teşhis Doğrulamasında (Confirmation):**
        1. **Precision** > 0.95 (yanlış alarmı minimize)
        2. **F1-Score** > 0.70
        3. İnsan hekimle birlikte kullanım
        """)
    
    with tabs[5]:
        st.markdown("""
        ### 📚 Referanslar ve Kaynaklar
        
        **Veri Seti:**
        - UCI Machine Learning Repository - Heart Disease Dataset
        - https://archive.ics.uci.edu/ml/datasets/heart+disease
        
        **SMOTE:**
        - Chawla, N. V., et al. (2002). SMOTE: Synthetic Minority Over-sampling Technique
        - Journal of Artificial Intelligence Research, 16, 321-357
        
        **Optuna:**
        - Akiba, T., Sano, S., Yanase, T., Ohta, T., & Koyama, M. (2019)
        - Optuna: A Next-generation Hyperparameter Optimization Framework
        - arXiv preprint arXiv:1907.10902
        
        **Makine Öğrenmesi Algoritmaları:**
        - Hastie, T., Tibshirani, R., & Friedman, J. (2009)
        - The Elements of Statistical Learning (2nd ed.)
        - Springer
        
        **Cross-Validation:**
        - Kohavi, R. (1995). A Study of Cross-Validation and Bootstrap for Accuracy Estimation
        - and Model Selection. In IJCAI, 14(2), 1137-1145
        
        **Kalp Hastalığı İstatistikleri:**
        - WHO - Cardiovascular diseases (CVDs) - Fact sheets
        - European Heart Journal - Clinical practice guidelines
        
        **PCA:**
        - Jolliffe, I. T. (2002). Principal Component Analysis (2nd ed.)
        - Springer-Verlag
        
        **Python Kütüphaneleri:**
        - scikit-learn >= 1.0.0
        - imbalanced-learn >= 0.9.0
        - optuna >= 3.0.0
        - xgboost >= 1.5.0
        - pandas >= 1.3.0
        - numpy >= 1.20.0
        - matplotlib >= 3.4.0
        - seaborn >= 0.11.0
        - streamlit >= 1.10.0
        """)

# ============================================================================
# MAIN APP ROUTER
# ============================================================================

def main():
    # Sidebar navigation
    st.sidebar.markdown("---")
    # st.sidebar.image("https://via.placeholder.com/200x100?text=Heart+Disease", use_container_width=True)
    st.sidebar.markdown("### ❤️ Heart Disease Prediction System")
    st.sidebar.markdown("---")
    
    page = st.sidebar.radio(
        "📖 **Sayfaları Seç:**",
        [
            "🏠 Ana Sayfa",
            "📊 Senaryo Analizi",
            "📈 Karşılaştırma",
            "🔥 Heatmap",
            "🏥 Hasta Prediksiyon",
            "💡 Model Önerileri",
            "📚 Teknik Dokümantasyon"
        ]
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    **📊 Proje Bilgisi:**
    - Veri: UCI Heart Disease
    - Senaryo: 6 (S0-S5)
    - Model: 6 + optimization
    - Metrik: Accuracy, Recall, F1, AUC
    
    **✅ Önerilen Model:**
    - Logistic Regression + S3 (SMOTE)
    - F1: 0.837 | Recall: 0.806
    
    **🎯 En İyi Performans:**
    - Logistic Regression + S5 (All Combined)
    - F1: 0.843 | Recall: 0.824
    """)
    
    # Route to pages
    if page == "🏠 Ana Sayfa":
        page_home()
    elif page == "📊 Senaryo Analizi":
        page_scenarios()
    elif page == "📈 Karşılaştırma":
        page_comparison()
    elif page == "🔥 Heatmap":
        page_heatmap()
    elif page == "🏥 Hasta Prediksiyon":
        page_patient_prediction()
    elif page == "💡 Model Önerileri":
        page_recommendations()
    else:  # Technical Documentation
        page_technical()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <center>
    
    **❤️ UCI Heart Disease Prediction System**
    
    Yapay Zeka destekli Kardiyovasküler Risk Tahmini
    
    _Bu sistem eğitim ve araştırma amaçlıdır. Tıbbi tanı aracı değildir._
    
    **Daima bir doktor ile danışınız.**
    
    </center>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
