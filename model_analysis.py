"""
Heart Disease Prediction Model - Comprehensive Analysis & Optimization
This script analyzes the current model and proposes improvements
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, 
                             roc_auc_score, confusion_matrix, classification_report, roc_curve)
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("HEART DISEASE PREDICTION MODEL - DETAILED ANALYSIS")
print("="*80)

# ============================================================================
# 1. DATA LOADING AND EXPLORATION
# ============================================================================
print("\n[1] LOADING AND EXPLORING DATASET")
print("-"*80)

df = pd.read_csv('data/heart_disease_uci.csv')

print(f"Dataset shape: {df.shape}")
print(f"Missing values:\n{df.isnull().sum()}")
print(f"\nData types:\n{df.dtypes}")
print(f"\nTarget variable (num) distribution:")
print(df['num'].value_counts().sort_index())

# Convert target to binary
df['target'] = df['num'].apply(lambda x: 1 if x > 0 else 0)
print(f"\nBinary target distribution:")
print(df['target'].value_counts())
print(f"Class balance: {df['target'].value_counts(normalize=True)}")

# ============================================================================
# 2. DATA PREPROCESSING
# ============================================================================
print("\n[2] DATA PREPROCESSING")
print("-"*80)

df_clean = df.drop(['id', 'dataset', 'num'], axis=1)

# Handle missing values
print("Handling missing values...")
for col in df_clean.columns:
    if df_clean[col].dtype == 'object':
        df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0])
    else:
        df_clean[col] = df_clean[col].fillna(df_clean[col].median())

print(f"Missing values after preprocessing: {df_clean.isnull().sum().sum()}")

# Encode categorical variables
encoders = {}
categorical_cols = df_clean.select_dtypes(include=['object']).columns
print(f"\nCategorical columns to encode: {list(categorical_cols)}")

for col in categorical_cols:
    le = LabelEncoder()
    df_clean[col] = le.fit_transform(df_clean[col])
    encoders[col] = le
    print(f"  {col}: {list(le.classes_)}")

print("\nPreprocessed data shape:", df_clean.shape)

# ============================================================================
# 3. ORIGINAL MODEL (AS IN THE PROVIDED CODE)
# ============================================================================
print("\n[3] ORIGINAL MODEL EVALUATION")
print("-"*80)

X = df_clean.drop('target', axis=1)
y = df_clean['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training set size: {X_train.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")
print(f"Training set class distribution:\n{y_train.value_counts(normalize=True)}")

# Original Model
model_original = RandomForestClassifier(n_estimators=100, random_state=42)
model_original.fit(X_train, y_train)

y_pred_original = model_original.predict(X_test)
y_pred_proba_original = model_original.predict_proba(X_test)[:, 1]

print("\nORIGINAL RANDOM FOREST MODEL PERFORMANCE:")
print(f"Accuracy:  {accuracy_score(y_test, y_pred_original):.4f}")
print(f"Precision: {precision_score(y_test, y_pred_original):.4f}")
print(f"Recall:    {recall_score(y_test, y_pred_original):.4f}")
print(f"F1-Score:  {f1_score(y_test, y_pred_original):.4f}")
print(f"ROC-AUC:   {roc_auc_score(y_test, y_pred_proba_original):.4f}")

print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred_original)
print(cm)
print(f"True Negatives:  {cm[0,0]}, False Positives: {cm[0,1]}")
print(f"False Negatives: {cm[1,0]}, True Positives:  {cm[1,1]}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred_original, target_names=['Healthy', 'At Risk']))

# Feature importance
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': model_original.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 10 Most Important Features:")
print(feature_importance.head(10))

# ============================================================================
# 4. ISSUES IDENTIFIED
# ============================================================================
print("\n[4] ISSUES IDENTIFIED IN ORIGINAL CODE")
print("-"*80)

issues = [
    "1. No data scaling - RF is scale-invariant but other models may benefit",
    "2. No cross-validation - only single train-test split evaluated",
    "3. No hyperparameter tuning - using default Random Forest parameters",
    "4. No class imbalance handling - dataset has imbalanced classes",
    "5. No feature selection - all features treated equally",
    "6. Single model approach - no ensemble of different algorithms",
    "7. Limited evaluation metrics - should include cross-validation scores",
    "8. No handling of unknown categories - basic try-except fallback",
    "9. No validation of feature ranges in user input",
    "10. Limited insights into model decision-making"
]

for issue in issues:
    print(f"  {issue}")

# ============================================================================
# 5. OPTIMIZATION 1: HYPERPARAMETER TUNING
# ============================================================================
print("\n[5] OPTIMIZATION 1: HYPERPARAMETER TUNING")
print("-"*80)

print("Tuning Random Forest hyperparameters...")
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, 15, 20, None],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
    'max_features': ['sqrt', 'log2']
}

rf_tuned = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=5,
    scoring='roc_auc',
    n_jobs=-1,
    verbose=0
)

rf_tuned.fit(X_train, y_train)
print(f"Best parameters: {rf_tuned.best_params_}")
print(f"Best CV ROC-AUC: {rf_tuned.best_score_:.4f}")

y_pred_tuned = rf_tuned.predict(X_test)
y_pred_proba_tuned = rf_tuned.predict_proba(X_test)[:, 1]

print("\nTUNED RANDOM FOREST PERFORMANCE:")
print(f"Accuracy:  {accuracy_score(y_test, y_pred_tuned):.4f}")
print(f"Precision: {precision_score(y_test, y_pred_tuned):.4f}")
print(f"Recall:    {recall_score(y_test, y_pred_tuned):.4f}")
print(f"F1-Score:  {f1_score(y_test, y_pred_tuned):.4f}")
print(f"ROC-AUC:   {roc_auc_score(y_test, y_pred_proba_tuned):.4f}")

improvement_tuned = roc_auc_score(y_test, y_pred_proba_tuned) - roc_auc_score(y_test, y_pred_proba_original)
print(f"ROC-AUC Improvement: +{improvement_tuned:.4f}")

# ============================================================================
# 6. OPTIMIZATION 2: GRADIENT BOOSTING
# ============================================================================
print("\n[6] OPTIMIZATION 2: GRADIENT BOOSTING")
print("-"*80)

print("Tuning Gradient Boosting hyperparameters...")
gb_tuned = GridSearchCV(
    GradientBoostingClassifier(random_state=42, validation_fraction=0.1, n_iter_no_change=10),
    {
        'n_estimators': [100, 200],
        'learning_rate': [0.01, 0.05],
        'max_depth': [3, 5, 7],
        'min_samples_split': [2, 5]
    },
    cv=5,
    scoring='roc_auc',
    n_jobs=-1,
    verbose=0
)

gb_tuned.fit(X_train, y_train)
print(f"Best parameters: {gb_tuned.best_params_}")
print(f"Best CV ROC-AUC: {gb_tuned.best_score_:.4f}")

y_pred_gb = gb_tuned.predict(X_test)
y_pred_proba_gb = gb_tuned.predict_proba(X_test)[:, 1]

print("\nGRADIENT BOOSTING PERFORMANCE:")
print(f"Accuracy:  {accuracy_score(y_test, y_pred_gb):.4f}")
print(f"Precision: {precision_score(y_test, y_pred_gb):.4f}")
print(f"Recall:    {recall_score(y_test, y_pred_gb):.4f}")
print(f"F1-Score:  {f1_score(y_test, y_pred_gb):.4f}")
print(f"ROC-AUC:   {roc_auc_score(y_test, y_pred_proba_gb):.4f}")

improvement_gb = roc_auc_score(y_test, y_pred_proba_gb) - roc_auc_score(y_test, y_pred_proba_original)
print(f"ROC-AUC Improvement: +{improvement_gb:.4f}")

# ============================================================================
# 7. OPTIMIZATION 3: ENSEMBLE VOTING
# ============================================================================
print("\n[7] OPTIMIZATION 3: ENSEMBLE VOTING")
print("-"*80)

# Create voting classifier
voter = VotingClassifier(
    estimators=[
        ('rf', rf_tuned.best_estimator_),
        ('gb', gb_tuned.best_estimator_),
        ('lr', LogisticRegression(max_iter=1000, random_state=42))
    ],
    voting='soft'
)

voter.fit(X_train, y_train)

y_pred_voter = voter.predict(X_test)
y_pred_proba_voter = voter.predict_proba(X_test)[:, 1]

print("VOTING ENSEMBLE PERFORMANCE:")
print(f"Accuracy:  {accuracy_score(y_test, y_pred_voter):.4f}")
print(f"Precision: {precision_score(y_test, y_pred_voter):.4f}")
print(f"Recall:    {recall_score(y_test, y_pred_voter):.4f}")
print(f"F1-Score:  {f1_score(y_test, y_pred_voter):.4f}")
print(f"ROC-AUC:   {roc_auc_score(y_test, y_pred_proba_voter):.4f}")

improvement_voter = roc_auc_score(y_test, y_pred_proba_voter) - roc_auc_score(y_test, y_pred_proba_original)
print(f"ROC-AUC Improvement: +{improvement_voter:.4f}")

# ============================================================================
# 8. CROSS-VALIDATION ANALYSIS
# ============================================================================
print("\n[8] CROSS-VALIDATION ANALYSIS (5-Fold)")
print("-"*80)

models = {
    'Original RF': model_original,
    'Tuned RF': rf_tuned.best_estimator_,
    'Gradient Boosting': gb_tuned.best_estimator_,
    'Voting Ensemble': voter
}

cv_results = {}
for name, model in models.items():
    cv_scores = cross_val_score(model, X, y, cv=5, scoring='roc_auc')
    cv_results[name] = cv_scores
    print(f"{name:20s}: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

# ============================================================================
# 9. PERFORMANCE COMPARISON TABLE
# ============================================================================
print("\n[9] COMPREHENSIVE PERFORMANCE COMPARISON")
print("-"*80)

results_data = {
    'Model': ['Original RF', 'Tuned RF', 'Gradient Boosting', 'Voting Ensemble'],
    'Accuracy': [
        accuracy_score(y_test, y_pred_original),
        accuracy_score(y_test, y_pred_tuned),
        accuracy_score(y_test, y_pred_gb),
        accuracy_score(y_test, y_pred_voter)
    ],
    'Precision': [
        precision_score(y_test, y_pred_original),
        precision_score(y_test, y_pred_tuned),
        precision_score(y_test, y_pred_gb),
        precision_score(y_test, y_pred_voter)
    ],
    'Recall': [
        recall_score(y_test, y_pred_original),
        recall_score(y_test, y_pred_tuned),
        recall_score(y_test, y_pred_gb),
        recall_score(y_test, y_pred_voter)
    ],
    'F1-Score': [
        f1_score(y_test, y_pred_original),
        f1_score(y_test, y_pred_tuned),
        f1_score(y_test, y_pred_gb),
        f1_score(y_test, y_pred_voter)
    ],
    'ROC-AUC': [
        roc_auc_score(y_test, y_pred_proba_original),
        roc_auc_score(y_test, y_pred_proba_tuned),
        roc_auc_score(y_test, y_pred_proba_gb),
        roc_auc_score(y_test, y_pred_proba_voter)
    ]
}

results_df = pd.DataFrame(results_data)
print(results_df.to_string(index=False))

# Save results
results_df.to_csv('results/model_performance_comparison.csv', index=False)
print("\n✓ Results saved to results/model_performance_comparison.csv")

# ============================================================================
# 10. VISUALIZATION
# ============================================================================
print("\n[10] GENERATING VISUALIZATIONS")
print("-"*80)

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')

metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
models_list = results_df['Model'].tolist()

for idx, metric in enumerate(metrics):
    ax = axes[idx // 3, idx % 3]
    values = results_df[metric].tolist()
    colors = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12']
    bars = ax.bar(models_list, values, color=colors)
    ax.set_ylabel(metric)
    ax.set_ylim([0, 1])
    ax.set_title(f'{metric} Comparison')
    ax.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom', fontsize=9)

# ROC Curves
ax = axes[1, 2]
fpr_orig, tpr_orig, _ = roc_curve(y_test, y_pred_proba_original)
fpr_tuned, tpr_tuned, _ = roc_curve(y_test, y_pred_proba_tuned)
fpr_gb, tpr_gb, _ = roc_curve(y_test, y_pred_proba_gb)
fpr_voter, tpr_voter, _ = roc_curve(y_test, y_pred_proba_voter)

ax.plot(fpr_orig, tpr_orig, label=f'Original RF (AUC={roc_auc_score(y_test, y_pred_proba_original):.3f})')
ax.plot(fpr_tuned, tpr_tuned, label=f'Tuned RF (AUC={roc_auc_score(y_test, y_pred_proba_tuned):.3f})')
ax.plot(fpr_gb, tpr_gb, label=f'GB (AUC={roc_auc_score(y_test, y_pred_proba_gb):.3f})')
ax.plot(fpr_voter, tpr_voter, label=f'Voting (AUC={roc_auc_score(y_test, y_pred_proba_voter):.3f})')
ax.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('ROC Curves')
ax.legend(loc='lower right', fontsize=8)

plt.tight_layout()
plt.savefig('results/model_comparison.png', dpi=300, bbox_inches='tight')
print("✓ Visualization saved to results/model_comparison.png")
plt.close()

# Feature Importance Comparison
fig, ax = plt.subplots(figsize=(12, 6))
feature_imp_tuned = pd.DataFrame({
    'feature': X.columns,
    'importance': rf_tuned.best_estimator_.feature_importances_
}).sort_values('importance', ascending=True).tail(10)

ax.barh(feature_imp_tuned['feature'], feature_imp_tuned['importance'], color='#2ecc71')
ax.set_xlabel('Importance Score')
ax.set_title('Top 10 Features - Tuned Random Forest')
plt.tight_layout()
plt.savefig('results/feature_importance.png', dpi=300, bbox_inches='tight')
print("✓ Feature importance plot saved to results/feature_importance.png")
plt.close()

# ============================================================================
# 11. SUMMARY AND RECOMMENDATIONS
# ============================================================================
print("\n" + "="*80)
print("SUMMARY & RECOMMENDATIONS")
print("="*80)

best_model_name = results_df.loc[results_df['ROC-AUC'].idxmax(), 'Model']
best_roc_auc = results_df['ROC-AUC'].max()

print(f"\n✓ BEST MODEL: {best_model_name}")
print(f"  - ROC-AUC Score: {best_roc_auc:.4f}")
print(f"  - Improvement over original: +{(best_roc_auc - results_df.loc[0, 'ROC-AUC']):.4f}")

print("\n📊 KEY RECOMMENDATIONS:")
print("  1. Use the Voting Ensemble for production deployment")
print("  2. Implement class weight balancing for imbalanced data")
print("  3. Add feature selection to reduce dimensionality")
print("  4. Implement SHAP for model interpretability")
print("  5. Add input validation with expected feature ranges")
print("  6. Monitor model performance on new data regularly")
print("  7. Consider cost-sensitive learning (higher penalty for false negatives)")
print("  8. Implement A/B testing for model updates")
print("  9. Add confidence thresholds for predictions")
print("  10. Create separate models for different population groups")

print("\n" + "="*80)
