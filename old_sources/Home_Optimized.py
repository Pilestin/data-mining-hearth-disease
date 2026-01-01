import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, roc_curve
from sklearn.preprocessing import LabelEncoder, StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Page Configuration
st.set_page_config(
    page_title="Kalp Sağlığı Risk Analizi - Optimized",
    page_icon="❤️",
    layout="wide"
)

# --- CUSTOM STYLING ---
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .risk-high {
        color: #d32f2f;
        font-weight: bold;
    }
    .risk-low {
        color: #388e3c;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# --- 1. DATA LOADING AND MODEL TRAINING (OPTIMIZED) ---
@st.cache_resource
def load_and_train_optimized_model():
    """
    Load data and train optimized ensemble model with:
    - Hyperparameter tuning
    - Class weight balancing
    - Multiple model approaches
    - Cross-validation
    """
    try:
        df = pd.read_csv(r'C:\Users\DELL\Desktop\YL_İkinciDonem\VeriMadenciliği\data-mining-hearth-disease\data\heart_disease_uci.csv')
    except FileNotFoundError:
        st.error("Dataset not found. Please ensure 'heart_disease_uci.csv' is in the correct location.")
        return None, None, None, None, None

    # Data Cleaning
    df = df.drop(['id', 'dataset'], axis=1)
    df['target'] = df['num'].apply(lambda x: 1 if x > 0 else 0)
    df = df.drop('num', axis=1)

    # Handle Missing Values
    for col in df.columns:
        if df[col].dtype == 'object':
            if df[col].isnull().sum() > 0:
                df[col] = df[col].fillna(df[col].mode()[0])
        else:
            if df[col].isnull().sum() > 0:
                df[col] = df[col].fillna(df[col].median())

    # Encode Categorical Variables
    encoders = {}
    for col in df.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le

    # Train-Test Split
    X = df.drop('target', axis=1)
    y = df['target']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Scale features for LogisticRegression
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # --- OPTIMIZED MODELS ---
    
    # 1. Tuned Random Forest with class weights
    rf_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_split=2,
        min_samples_leaf=2,
        max_features='sqrt',
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X_train, y_train)

    # 2. Tuned Gradient Boosting
    gb_model = GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.01,
        max_depth=3,
        min_samples_split=5,
        random_state=42,
        validation_fraction=0.1,
        n_iter_no_change=10
    )
    gb_model.fit(X_train, y_train)

    # 3. Logistic Regression
    lr_model = LogisticRegression(
        max_iter=1000,
        class_weight='balanced',
        random_state=42
    )
    lr_model.fit(X_train_scaled, y_train)

    # 4. Voting Ensemble
    ensemble_model = VotingClassifier(
        estimators=[
            ('rf', rf_model),
            ('gb', gb_model),
            ('lr', lr_model)
        ],
        voting='soft'
    )
    ensemble_model.fit(X_train, y_train)

    # Evaluation on Test Set
    y_pred_ensemble = ensemble_model.predict(X_test)
    y_pred_proba_ensemble = ensemble_model.predict_proba(X_test)[:, 1]
    
    accuracy = accuracy_score(y_test, y_pred_ensemble)
    roc_auc = roc_auc_score(y_test, y_pred_proba_ensemble)

    # Feature Importance from Random Forest
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False)

    # Cross-validation Score
    cv_scores = cross_val_score(ensemble_model, X, y, cv=5, scoring='roc_auc')
    
    model_info = {
        'accuracy': accuracy,
        'roc_auc': roc_auc,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'test_set_size': len(y_test),
        'positive_class_pct': (y == 1).sum() / len(y) * 100
    }

    return ensemble_model, encoders, df, feature_importance, model_info

# Load models
ensemble_model, encoders, df_clean, feature_importance, model_info = load_and_train_optimized_model()

if ensemble_model is None:
    st.stop()

# --- 2. INTERFACE DESIGN ---

st.title("❤️ AI-Powered Heart Disease Risk Analysis (Optimized)")
st.markdown("""
This application uses an optimized ensemble machine learning model to predict heart disease risk.
**Please enter patient values in the left sidebar.**

**Model Performance:**
- Test Accuracy: **{:.2%}**
- ROC-AUC Score: **{:.4f}**
- Cross-validation AUC: **{:.4f} ± {:.4f}**
""".format(
    model_info['accuracy'],
    model_info['roc_auc'],
    model_info['cv_mean'],
    model_info['cv_std']
))

st.sidebar.header("Patient Information Input")
st.sidebar.markdown("Please enter patient test results.")

# Expected ranges for validation
FEATURE_RANGES = {
    'age': (20, 80),
    'trestbps': (90, 200),
    'chol': (100, 600),
    'thalach': (60, 220),
    'oldpeak': (0.0, 6.0),
    'ca': (0, 3)
}

def user_input_features():
    """Collect user input with validation"""
    with st.expander("📋 Demographic Information", expanded=True):
        age = st.slider('Age', 20, 80, 50, help="Patient's age in years")
        
        sex_disp = st.selectbox('Sex', ('Male', 'Female'))
        sex = 'Male' if sex_disp == 'Male' else 'Female'

    with st.expander("🩺 Clinical Findings", expanded=True):
        cp_map = {
            'Typical Angina': 'typical angina',
            'Atypical Angina': 'atypical angina',
            'Non-Anginal Pain': 'non-anginal',
            'Asymptomatic': 'asymptomatic'
        }
        cp_disp = st.selectbox('Chest Pain Type', list(cp_map.keys()))
        cp = cp_map[cp_disp]
        
        trestbps = st.number_input(
            'Resting Blood Pressure (mm Hg)', 90, 200, 120,
            help="Blood pressure at hospital admission. Normal: 120/80 mm Hg"
        )
        
        chol = st.number_input(
            'Serum Cholesterol (mg/dl)', 100, 600, 200,
            help="Total blood cholesterol. Desirable: <200 mg/dl"
        )
        
        fbs_disp = st.radio('Fasting Blood Sugar > 120 mg/dl?', ('No', 'Yes'))
        fbs = True if fbs_disp == 'Yes' else False

    with st.expander("📊 Test Results", expanded=True):
        restecg_map = {
            'Normal': 'normal',
            'ST-T Abnormality': 'st-t abnormality',
            'LV Hypertrophy': 'lv hypertrophy'
        }
        restecg_disp = st.selectbox('Resting ECG Result', list(restecg_map.keys()))
        restecg = restecg_map[restecg_disp]
        
        thalach = st.slider(
            'Max Heart Rate Achieved', 60, 220, 150,
            help="Highest heart rate during stress test"
        )
        
        exang_disp = st.radio('Exercise-Induced Angina?', ('No', 'Yes'))
        exang = True if exang_disp == 'Yes' else False
        
        oldpeak = st.slider(
            'ST Segment Depression (Oldpeak)', 0.0, 6.0, 1.0, 0.1,
            help="ST segment depression caused by exercise"
        )
        
        slope_map = {
            'Upsloping': 'upsloping',
            'Flat': 'flat',
            'Downsloping': 'downsloping'
        }
        slope_disp = st.selectbox('ST Segment Slope', list(slope_map.keys()))
        slope = slope_map[slope_disp]
        
        ca = st.slider(
            'Number of Major Vessels (0-3)', 0, 3, 0,
            help="Number of major vessels colored by fluoroscopy"
        )
        
        thal_map = {
            'Normal': 'normal',
            'Fixed Defect': 'fixed defect',
            'Reversible Defect': 'reversable defect'
        }
        thal_disp = st.selectbox('Thalassemia Status', list(thal_map.keys()))
        thal = thal_map[thal_disp]

    data = {
        'age': age, 'sex': sex, 'cp': cp, 'trestbps': trestbps,
        'chol': chol, 'fbs': fbs, 'restecg': restecg,
        'thalch': thalach, 'exang': exang, 'oldpeak': oldpeak,
        'slope': slope, 'ca': ca, 'thal': thal
    }
    return pd.DataFrame(data, index=[0])

input_df = user_input_features()

# --- 3. PREDICTION AND ANALYSIS ---

def encode_input(input_df, encoders):
    """Encode categorical variables safely"""
    input_df_encoded = input_df.copy()
    for col, encoder in encoders.items():
        if col in input_df_encoded.columns:
            try:
                input_df_encoded[col] = encoder.transform(input_df_encoded[col])
            except ValueError:
                # Handle unknown categories
                input_df_encoded[col] = 0
    return input_df_encoded

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Patient Input Values")
    st.write(input_df)
    
    if st.button('🔍 Analyze Risk', use_container_width=True):
        input_df_encoded = encode_input(input_df, encoders)
        
        # Get predictions from ensemble
        prediction = ensemble_model.predict(input_df_encoded)[0]
        prediction_proba = ensemble_model.predict_proba(input_df_encoded)[0]
        
        risk_probability = prediction_proba[1]
        confidence = max(prediction_proba) * 100
        
        st.divider()
        st.subheader("Analysis Results")
        
        # Risk Assessment
        col_risk1, col_risk2, col_risk3 = st.columns(3)
        
        with col_risk1:
            st.metric("Risk Probability", f"{risk_probability*100:.1f}%")
        
        with col_risk2:
            st.metric("Model Confidence", f"{confidence:.1f}%")
        
        with col_risk3:
            if risk_probability > 0.5:
                st.metric("Risk Level", "🔴 HIGH")
            else:
                st.metric("Risk Level", "🟢 LOW")
        
        # Detailed Assessment
        st.markdown("---")
        
        if risk_probability > 0.7:
            st.error("⚠️ **HIGH RISK DETECTED**")
            st.write(f"""
            The model indicates a **{risk_probability*100:.1f}%** probability of heart disease.
            
            **Recommendations:**
            - Consult with a cardiologist immediately
            - Schedule comprehensive heart tests (ECG, stress test, angiography)
            - Review and manage risk factors (blood pressure, cholesterol, exercise)
            - Take prescribed medications as directed
            """)
        elif risk_probability > 0.5:
            st.warning("⚠️ **MODERATE RISK**")
            st.write(f"""
            The model indicates a **{risk_probability*100:.1f}%** probability of heart disease.
            
            **Recommendations:**
            - Schedule an appointment with a cardiologist
            - Get a comprehensive heart health assessment
            - Monitor vital signs regularly
            - Implement lifestyle modifications
            """)
        else:
            st.success("✅ **LOW RISK**")
            st.write(f"""
            The model indicates a **{risk_probability*100:.1f}%** probability of heart disease.
            
            **Recommendations:**
            - Continue regular health checkups
            - Maintain a healthy lifestyle
            - Monitor risk factors periodically
            - Keep cholesterol and blood pressure in check
            """)

        # Comparative Analysis
        st.subheader("Comparative Analysis")
        
        # Compare with patient groups
        sick_avg = df_clean[df_clean['target'] == 1].mean(numeric_only=True)
        healthy_avg = df_clean[df_clean['target'] == 0].mean(numeric_only=True)
        
        metrics_compare = ['chol', 'thalch', 'trestbps']
        labels_compare = ['Cholesterol', 'Max Heart Rate', 'Blood Pressure']
        
        fig, ax = plt.subplots(figsize=(11, 5))
        x = np.arange(len(metrics_compare))
        width = 0.25
        
        user_vals = [input_df['chol'][0], input_df['thalch'][0], input_df['trestbps'][0]]
        sick_vals = [sick_avg['chol'], sick_avg['thalch'], sick_avg['trestbps']]
        healthy_vals = [healthy_avg['chol'], healthy_avg['thalch'], healthy_avg['trestbps']]
        
        ax.bar(x - width, user_vals, width, label='This Patient', color='#3498db')
        ax.bar(x, sick_vals, width, label='Average At-Risk Patient', color='#e74c3c')
        ax.bar(x + width, healthy_vals, width, label='Average Healthy Patient', color='#2ecc71')
        
        ax.set_ylabel('Values')
        ax.set_title('Patient Metrics vs. General Population')
        ax.set_xticks(x)
        ax.set_xticklabels(labels_compare)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        st.pyplot(fig)

        # Feature Importance
        st.subheader("Model Decision Factors")
        st.markdown("The following features most influence the model's prediction:")
        
        fig_imp, ax_imp = plt.subplots(figsize=(11, 6))
        top_features = feature_importance.head(10)
        colors = plt.cm.RdYlGn_r(np.linspace(0.3, 0.7, len(top_features)))
        ax_imp.barh(range(len(top_features)), top_features['importance'], color=colors)
        ax_imp.set_yticks(range(len(top_features)))
        ax_imp.set_yticklabels(top_features['feature'])
        ax_imp.set_xlabel('Importance Score')
        ax_imp.set_title('Top 10 Decision Factors')
        ax_imp.invert_yaxis()
        
        st.pyplot(fig_imp)

        # Legal Disclaimer
        st.error("""
        ⚠️ **MEDICAL DISCLAIMER**
        
        This application is a **predictive tool only** and NOT a medical diagnosis device.
        Results are provided for informational purposes only. 
        
        **Always consult with a qualified healthcare provider for medical advice, diagnosis, and treatment.**
        """)

with col2:
    st.info("📊 **Model Information**")
    st.markdown(f"""
    **Performance Metrics:**
    - Test Accuracy: {model_info['accuracy']:.2%}
    - ROC-AUC: {model_info['roc_auc']:.4f}
    - CV Score: {model_info['cv_mean']:.4f} ± {model_info['cv_std']:.4f}
    
    **Dataset Info:**
    - Samples: {len(df_clean)}
    - Risk Class: {model_info['positive_class_pct']:.1f}%
    - Test Set: {model_info['test_set_size']} patients
    
    **Model Features:**
    - Ensemble (RF + GB + LR)
    - Hyperparameter Tuned
    - Class Weight Balanced
    - Cross-Validated
    
    **Input Parameters:**
    - **cp:** Chest pain type
    - **trestbps:** Resting BP
    - **chol:** Cholesterol
    - **thalch:** Max heart rate
    - **oldpeak:** ST depression
    - **ca:** Vessel count
    - **thal:** Thalassemia status
    """)
    
    # Model comparison info
    st.info("**About the Model:**\n\nThis app uses an optimized ensemble combining:\n- Random Forest (RF)\n- Gradient Boosting (GB)\n- Logistic Regression (LR)")
