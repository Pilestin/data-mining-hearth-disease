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
    page_title="Kalp Sağlığı Risk Analizi",
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

# --- 1. DATA LOADING AND MODEL TRAINING ---
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
        df = pd.read_csv('data/heart_disease_uci.csv')
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

st.title("❤️ Yapay Zeka Destekli Kalp Hastalığı Risk Analizi")
st.markdown("""
Bu uygulama, kalp hastalığı riskini tahmin etmek için optimize edilmiş topluluk makine öğrenimi modeli kullanmaktadır.
**Lütfen hasta bilgilerini aşağıdaki formlara giriniz.**

**Model Performansı:**
- Test Doğruluğu: **{:.2%}**
- ROC-AUC Skoru: **{:.4f}**
- Çapraz Doğrulama AUC: **{:.4f} ± {:.4f}**
""".format(
    model_info['accuracy'],
    model_info['roc_auc'],
    model_info['cv_mean'],
    model_info['cv_std']
))

st.sidebar.header("Hasta Bilgileri Girişi")
st.sidebar.markdown("Lütfen hasta test sonuçlarını giriniz.")

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
    """Kullanıcı girişini doğrulama ile topla"""
    with st.expander("📋 Demografik Bilgiler", expanded=True):
        age = st.slider('Yaş', 20, 80, 50, help="Hastanın yaşı (yıl cinsinden). Yaş arttıkça kalp hastalığı riski genelde artar.")
        
        sex_disp = st.selectbox('Cinsiyet', ('Erkek', 'Kadın'), help="Erkeklerde kalp hastalığı riski genelde daha yüksektir.")
        sex = 'Male' if sex_disp == 'Erkek' else 'Female'

    with st.expander("🩺 Klinik Bulgular", expanded=True):
        cp_map = {
            'Tipik Anjina': 'typical angina',
            'Atipik Anjina': 'atypical angina',
            'Anjinal Olmayan Ağrı': 'non-anginal',
            'Asemptomatik': 'asymptomatic'
        }
        cp_disp = st.selectbox('Göğüs Ağrısı Tipi', list(cp_map.keys()), 
                               help="Göğüs ağrısının türü kalp hastalığı için önemli bir belirteçtir. Tipik anjina en riskli olanıdır.")
        cp = cp_map[cp_disp]
        
        trestbps = st.number_input(
            'İstirahat Kan Basıncı (mm Hg)', 90, 200, 120,
            help="Hastaneye başvuru sırasındaki dinlenme tansiyonu. Normal: 120/80 mm Hg. Yüksek tansiyon risk faktörüdür."
        )
        
        chol = st.number_input(
            'Serum Kolesterolu (mg/dl)', 100, 600, 200,
            help="Kandaki toplam kolesterol. İdeal: <200 mg/dl. Yüksek kolesterol damar tıkanıklığına yol açabilir."
        )
        
        fbs_disp = st.radio('Açlık Kan Şekeri > 120 mg/dl?', ('Hayır', 'Evet'),
                           help="Yüksek açlık kan şekeri diyabet ve kalp hastalığı riski göstergesidir.")
        fbs = True if fbs_disp == 'Evet' else False

    with st.expander("📊 Test Sonuçları", expanded=True):
        restecg_map = {
            'Normal': 'normal',
            'ST-T Dalga Anormalliği': 'st-t abnormality',
            'Sol Ventrikül Hipertrofisi': 'lv hypertrophy'
        }
        restecg_disp = st.selectbox('İstirahat EKG Sonucu', list(restecg_map.keys()),
                                   help="Dinlenme halindeki EKG sonucu. Anormallikler kalp sorunu işareti olabilir.")
        restecg = restecg_map[restecg_disp]
        
        thalach = st.slider(
            'Maksimum Kalp Atış Hızı', 60, 220, 150,
            help="Efor testi sırasında ulaşılan en yüksek nabız. Düşük değerler kalp sorunu gösterebilir."
        )
        
        exang_disp = st.radio('Egzersize Bağlı Anjina?', ('Hayır', 'Evet'),
                             help="Efor sırasında göğüs ağrısı oluşuyor mu? Evet cevabı yüksek risk göstergesidir.")
        exang = True if exang_disp == 'Evet' else False
        
        oldpeak = st.slider(
            'ST Depresyonu (Oldpeak)', 0.0, 6.0, 1.0, 0.1,
            help="Egzersiz sırasında EKG'de oluşan ST segment çökmesi. Yüksek değerler iskemi belirtisidir."
        )
        
        slope_map = {
            'Yukarı Eğimli': 'upsloping',
            'Düz': 'flat',
            'Aşağı Eğimli': 'downsloping'
        }
        slope_disp = st.selectbox('ST Segment Eğimi', list(slope_map.keys()),
                                 help="Efor sırasındaki ST segment eğimi. Yukarı eğimli genelde iyidir, düz/aşağı eğimli risklidir.")
        slope = slope_map[slope_disp]
        
        ca = st.slider(
            'Ana Damar Sayısı (0-3)', 0, 3, 0,
            help="Floroskopi ile görüntülenen tıkalı/daralmış ana damar sayısı. Sayı arttıkça risk artar."
        )
        
        thal_map = {
            'Normal': 'normal',
            'Sabit Kusur': 'fixed defect',
            'Tersine Çevrilebilir Kusur': 'reversable defect'
        }
        thal_disp = st.selectbox('Talasemi Durumu', list(thal_map.keys()),
                                help="Kalbe giden kan akışı durumu. Kusurlar iskemi veya kalıcı hasar gösterebilir.")
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
    st.subheader("Girilen Hasta Bilgileri")
    st.write(input_df)
    
    if st.button('🔍 Risk Analizi Yap', use_container_width=True):
        input_df_encoded = encode_input(input_df, encoders)
        
        # Get predictions from ensemble
        prediction = ensemble_model.predict(input_df_encoded)[0]
        prediction_proba = ensemble_model.predict_proba(input_df_encoded)[0]
        
        risk_probability = prediction_proba[1]
        confidence = max(prediction_proba) * 100
        
        st.divider()
        st.subheader("Analiz Sonuçları")
        
        # Risk Değerlendirmesi
        col_risk1, col_risk2, col_risk3 = st.columns(3)
        
        with col_risk1:
            st.metric("Risk Olasılığı", f"{risk_probability*100:.1f}%")
        
        with col_risk2:
            st.metric("Model Güveni", f"{confidence:.1f}%")
        
        with col_risk3:
            if risk_probability > 0.5:
                st.metric("Risk Seviyesi", "🔴 YÜKSEK")
            else:
                st.metric("Risk Seviyesi", "🟢 DÜŞÜK")
        
        # Detailed Assessment
        st.markdown("---")
        
        if risk_probability > 0.7:
            st.error("⚠️ **YÜKSEK RİSK TESPİT EDİLDİ**")
            st.write(f"""
            Model, **%{risk_probability*100:.1f}** kalp hastalığı olasılığı göstermektedir.
            
            **Öneriler:**
            - Derhal bir kardiyolog ile görüşünüz
            - Kapsamlı kalp testleri yaptırınız (EKG, efor testi, anjiyografi)
            - Risk faktörlerini gözden geçirin ve yönetin (tansiyon, kolesterol, egzersiz)
            - Reçete edilen ilaçları düzenli kullanınız
            - Sağlıklı beslenme ve yaşam tarzı değişiklikleri yapınız
            """)
        elif risk_probability > 0.5:
            st.warning("⚠️ **ORTA DÜZEY RİSK**")
            st.write(f"""
            Model, **%{risk_probability*100:.1f}** kalp hastalığı olasılığı göstermektedir.
            
            **Öneriler:**
            - Bir kardiyolog ile randevu alınız
            - Kapsamlı kalp sağlığı değerlendirmesi yaptırınız
            - Yaşamsal belirtileri düzenli olarak kontrol ediniz
            - Yaşam tarzı değişiklikleri uygulayınız
            - Düzenli egzersiz ve sağlıklı beslenme programı başlatınız
            """)
        else:
            st.success("✅ **DÜŞÜK RİSK**")
            st.write(f"""
            Model, **%{risk_probability*100:.1f}** kalp hastalığı olasılığı göstermektedir.
            
            **Öneriler:**
            - Düzenli sağlık kontrollerine devam ediniz
            - Sağlıklı yaşam tarzını sürdürünüz
            - Risk faktörlerini periyodik olarak takip ediniz
            - Kolesterol ve kan basıncını kontrol altında tutunuz
            - Dengeli beslenme ve düzenli egzersize özen gösteriniz
            """)

        # Karşılaştırmalı Analiz
        st.subheader("Karşılaştırmalı Analiz")
        
        # Hasta grupları ile karşılaştır
        sick_avg = df_clean[df_clean['target'] == 1].mean(numeric_only=True)
        healthy_avg = df_clean[df_clean['target'] == 0].mean(numeric_only=True)
        
        metrics_compare = ['chol', 'thalch', 'trestbps']
        labels_compare = ['Kolesterol', 'Maks Kalp Hızı', 'Kan Basıncı']
        
        fig, ax = plt.subplots(figsize=(11, 5))
        x = np.arange(len(metrics_compare))
        width = 0.25
        
        user_vals = [input_df['chol'][0], input_df['thalch'][0], input_df['trestbps'][0]]
        sick_vals = [sick_avg['chol'], sick_avg['thalch'], sick_avg['trestbps']]
        healthy_vals = [healthy_avg['chol'], healthy_avg['thalch'], healthy_avg['trestbps']]
        
        ax.bar(x - width, user_vals, width, label='Bu Hasta', color='#3498db')
        ax.bar(x, sick_vals, width, label='Ortalama Riskli Hasta', color='#e74c3c')
        ax.bar(x + width, healthy_vals, width, label='Ortalama Sağlıklı Hasta', color='#2ecc71')
        
        ax.set_ylabel('Değerler')
        ax.set_title('Hasta Metrikleri vs Genel Popülasyon')
        ax.set_xticks(x)
        ax.set_xticklabels(labels_compare)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        st.pyplot(fig)

        # Özellik Önemi
        st.subheader("Model Karar Faktörleri")
        st.markdown("Aşağıdaki özellikler modelin tahminini en çok etkileyen faktörlerdir:")
        
        fig_imp, ax_imp = plt.subplots(figsize=(11, 6))
        top_features = feature_importance.head(10)
        colors = plt.cm.RdYlGn_r(np.linspace(0.3, 0.7, len(top_features)))
        ax_imp.barh(range(len(top_features)), top_features['importance'], color=colors)
        ax_imp.set_yticks(range(len(top_features)))
        ax_imp.set_yticklabels(top_features['feature'])
        ax_imp.set_xlabel('Önem Skoru')
        ax_imp.set_title('En Önemli 10 Karar Faktörü')
        ax_imp.invert_yaxis()
        
        st.pyplot(fig_imp)

        # Yasal Uyarı
        st.error("""
        ⚠️ **TIBBİ SORUMLULUK REDDİ**
        
        Bu uygulama **yalnızca tahmin aracıdır** ve tıbbi teşhis cihazı DEĞİLDİR.
        Sonuçlar yalnızca bilgilendirme amaçlıdır. 
        
        **Tıbbi öneri, teşhis ve tedavi için MUTLAKA uzman bir sağlık kuruluşuna başvurunuz.**
        """)

with col2:
    st.info("📊 **Model Bilgileri**")
    st.markdown(f"""
    **Performans Metrikleri:**
    - Test Doğruluğu: {model_info['accuracy']:.2%}
    - ROC-AUC: {model_info['roc_auc']:.4f}
    - Çapraz Doğrulama Skoru: {model_info['cv_mean']:.4f} ± {model_info['cv_std']:.4f}
    
    **Veri Seti Bilgisi:**
    - Örnekler: {len(df_clean)}
    - Risk Sınıfı: %{model_info['positive_class_pct']:.1f}
    - Test Seti: {model_info['test_set_size']} hasta
    
    **Model Özellikleri:**
    - Topluluk Modeli (RF + GB + LR)
    - Hiperparametre Ayarlamalı
    - Sınıf Ağırlığı Dengelenmiş
    - Çapraz Doğrulanmış
    
    **Giriş Parametreleri:**
    - **cp:** Göğüs ağrısı tipi
    - **trestbps:** İstirahat kan basıncı
    - **chol:** Kolesterol
    - **thalch:** Maksimum kalp hızı
    - **oldpeak:** ST depresyonu
    - **ca:** Ana damar sayısı
    - **thal:** Talasemi durumu
    """)
    
    # Model karşılaştırma bilgisi
    st.info("**Model Hakkında:**\n\nBu uygulama optimize edilmiş topluluk modeli kullanır:\n- Rastgele Orman (RF)\n- Gradyan Arttırma (GB)\n- Lojistik Regresyon (LR)")
    
    # Analiz görsellerini göster
    import os
    
    if os.path.exists('results/feature_importance.png'):
        st.markdown("---")
        st.subheader("Özellik Önem Analizi")
        st.image('results/feature_importance.png', caption='Model Analiz Raporu: Özellik Önemi', use_container_width=True)
    
    if os.path.exists('results/model_comparison.png'):
        st.markdown("---")
        st.subheader("Model Karşılaştırma")
        st.image('results/model_comparison.png', caption='Model Performans Karşılaştırması', use_container_width=True)
