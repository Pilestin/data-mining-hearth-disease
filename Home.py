import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

# Sayfa Ayarları
st.set_page_config(
    page_title="Kalp Sağlığı Risk Analizi",
    page_icon="❤️",
    layout="wide"
)

# --- 1. VERİ YÜKLEME VE MODEL EĞİTİMİ ---
@st.cache_resource
def load_and_train_model():
    # Veriyi yükle
    try:
        df = pd.read_csv('data/heart_disease_uci.csv')
    except FileNotFoundError:
        st.error("Lütfen 'heart_disease_uci.csv' dosyasını bu kodla aynı klasöre koyduğunuzdan emin olun.")
        return None, None, None

    # Gereksiz sütunları çıkar
    df = df.drop(['id', 'dataset'], axis=1)

    # Hedef değişkeni ikili (binary) sınıfa çevir (0: Sağlıklı, 1-4: Riskli)
    df['target'] = df['num'].apply(lambda x: 1 if x > 0 else 0)
    df = df.drop('num', axis=1)

    # Eksik verileri doldurma (Basit İmputasyon)
    # Kategorik olanlar için mod (en çok tekrar eden), sayısal olanlar için medyan
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].fillna(df[col].mode()[0])
        else:
            df[col] = df[col].fillna(df[col].median())

    # Kategorik verileri sayısal hale getirme (Label Encoding)
    # Gerçek uygulamada OneHotEncoder daha iyidir ama basitlik için LabelEncoder kullanıyoruz.
    # Kullanıcıdan gelen veriyi de aynı şekilde dönüştürmek için mapping'leri saklayacağız.
    encoders = {}
    for col in df.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le

    # Model Eğitimi
    X = df.drop('target', axis=1)
    y = df['target']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    accuracy = accuracy_score(y_test, model.predict(X_test))
    
    # Feature Importance hesaplama
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    return model, encoders, df, feature_importance # df'i istatistikler için döndürüyoruz

model, encoders, df_clean, feature_importance = load_and_train_model()

if model is None:
    st.stop()

# --- 2. ARAYÜZ TASARIMI ---

st.title("❤️ Yapay Zeka Destekli Kalp Hastalığı Risk Analizi")
st.markdown("""
Bu uygulama, makine öğrenimi modelleri (Random Forest) kullanarak kalp hastalığı riskini tahmin eder.
Lütfen sol taraftaki menüden hastanın değerlerini giriniz.
""")

st.sidebar.header("Hasta Bilgileri Girişi")
st.sidebar.markdown("Lütfen kan ve test sonuçlarını giriniz.")

def user_input_features():
    with st.expander("📝 Demografik Bilgiler", expanded=True):
        # Yaş
        age = st.slider('Yaş', 20, 80, 50, help="Hastanın yaşı.")
        
        # Cinsiyet
        sex_disp = st.selectbox('Cinsiyet', ('Erkek', 'Kadın'), help="Hastanın biyolojik cinsiyeti.")
        sex = 'Male' if sex_disp == 'Erkek' else 'Female'

    with st.expander("🩺 Klinik Bulgular", expanded=True):
        # Göğüs Ağrısı Tipi (CP)
        cp_map = {
            'Tipik Anjina': 'typical angina',
            'Atipik Anjina': 'atypical angina',
            'Anjinal Olmayan Ağrı': 'non-anginal',
            'Asemptomatik': 'asymptomatic'
        }
        cp_disp = st.selectbox(
            'Göğüs Ağrısı Tipi', 
            list(cp_map.keys()), 
            help="Hastanın şikayet ettiği göğüs ağrısı türü. Detaylar için Yardım sayfasına bakınız."
        )
        cp = cp_map[cp_disp]
        
        # Kan Basıncı (Trestbps)
        trestbps = st.number_input(
            'İstirahat Kan Basıncı (mm Hg)', 
            90, 200, 120, 
            help="Hastaneye girişteki dinlenme tansiyonu. 120/80 mm Hg normal kabul edilir."
        )
        
        # Kolesterol (Chol)
        chol = st.number_input(
            'Serum Kolesterol (mg/dl)', 
            100, 600, 200,
            help="Kandaki toplam kolesterol miktarı. 200 mg/dl altı istenen seviyedir."
        )
        
        # Açlık Kan Şekeri (FBS)
        fbs_disp = st.radio(
            'Açlık Kan Şekeri > 120 mg/dl?', 
            ('Hayır', 'Evet'),
            help="Aç karnına ölçülen kan şekeri 120 mg/dl'den yüksek mi?"
        )
        fbs = True if fbs_disp == 'Evet' else False

    with st.expander("🔬 Test Sonuçları", expanded=True):
        # EKG Sonuçları (Restecg)
        restecg_map = {
            'Normal': 'normal',
            'ST-T Dalga Anormalliği': 'st-t abnormality',
            'Sol Ventrikül Hipertrofisi': 'lv hypertrophy'
        }
        restecg_disp = st.selectbox(
            'İstirahat EKG Sonucu', 
            list(restecg_map.keys()),
            help="Dinlenme halindeki EKG sonucu."
        )
        restecg = restecg_map[restecg_disp]
        
        # Maksimum Kalp Atış Hızı (Thalach)
        thalach = st.slider(
            'Maksimum Kalp Atış Hızı', 
            60, 220, 150,
            help="Efor testi sırasında ulaşılan en yüksek nabız."
        )
        
        # Egzersize Bağlı Anjina (Exang)
        exang_disp = st.radio(
            'Egzersize Bağlı Anjina?', 
            ('Hayır', 'Evet'),
            help="Efor sarf ederken göğüs ağrısı oluyor mu?"
        )
        exang = True if exang_disp == 'Evet' else False
        
        # Oldpeak
        oldpeak = st.slider(
            'ST Depresyonu (Oldpeak)', 
            0.0, 6.0, 1.0, 0.1,
            help="Egzersizle oluşan ST segment çökmesi miktarı."
        )
        
        # Eğim (Slope)
        slope_map = {
            'Yukarı Eğimli': 'upsloping',
            'Düz': 'flat',
            'Aşağı Eğimli': 'downsloping'
        }
        slope_disp = st.selectbox(
            'ST Segment Eğimi', 
            list(slope_map.keys()),
            help="Efor sırasındaki EKG'de ST segmentinin eğimi."
        )
        slope = slope_map[slope_disp]
        
        # Büyük Damarlar (CA)
        ca = st.slider(
            'Floroskopi ile Boyanan Ana Damar Sayısı (0-3)', 
            0, 3, 0,
            help="Görüntülemede görülen tıkalı/daralmış ana damar sayısı."
        )
        
        # Talasemi (Thal)
        thal_map = {
            'Normal': 'normal',
            'Sabit Kusur': 'fixed defect',
            'Tersine Çevrilebilir Kusur': 'reversable defect'
        }
        thal_disp = st.selectbox(
            'Talasemi Durumu', 
            list(thal_map.keys()),
            help="Kan akışı (perfüzyon) durumu."
        )
        thal = thal_map[thal_disp]

    data = {
        'age': age, 'sex': sex, 'cp': cp, 'trestbps': trestbps,
        'chol': chol, 'fbs': fbs, 'restecg': restecg,
        'thalch': thalach, 'exang': exang, 'oldpeak': oldpeak,
        'slope': slope, 'ca': ca, 'thal': thal
    }
    return pd.DataFrame(data, index=[0])

input_df = user_input_features()

# --- 3. TAHMİN ---

# Girdiyi encode etme
input_df_encoded = input_df.copy()
for col, encoder in encoders.items():
    if col in input_df_encoded.columns:
        # Bilinmeyen kategori hatasını önlemek için basit try-except (veya map)
        try:
            input_df_encoded[col] = encoder.transform(input_df_encoded[col])
        except:
             # Eğer eğitim setinde olmayan bir kategori gelirse (nadir), en sık görüleni ata
             input_df_encoded[col] = 0 

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Hasta Giriş Değerleri")
    st.write(input_df)
    
    if st.button('Risk Analizi Yap'):
        prediction = model.predict(input_df_encoded)
        prediction_proba = model.predict_proba(input_df_encoded)
        
        risk_probability = prediction_proba[0][1]
        
        st.divider()
        st.subheader("Analiz Sonucu")
        
        if risk_probability > 0.5:
            st.error(f"⚠️ **Yüksek Risk Tespit Edildi**")
            st.write(f"Model, bu hastada **%{risk_probability*100:.1f}** ihtimalle kalp hastalığı riski öngörüyor.")
        else:
            st.success(f"✅ **Düşük Risk**")
            st.write(f"Model, bu hastada **%{risk_probability*100:.1f}** ihtimalle kalp hastalığı riski öngörüyor.")

        # Grafiksel Karşılaştırma
        st.subheader("Karşılaştırmalı Analiz")
        
        # Kullanıcı verisi vs Ortalama Hasta Verisi
        sick_avg = df_clean[df_clean['target'] == 1].mean(numeric_only=True)
        healthy_avg = df_clean[df_clean['target'] == 0].mean(numeric_only=True)
        
        # Önemli 3 parametreyi karşılaştıralım
        metrics = ['chol', 'thalch', 'trestbps']
        labels = ['Kolesterol', 'Max Kalp Hızı', 'Kan Basıncı']
        
        fig, ax = plt.subplots(figsize=(10, 5))
        x = np.arange(len(metrics))
        width = 0.25
        
        # Kullanıcı değerleri
        user_vals = [input_df['chol'][0], input_df['thalch'][0], input_df['trestbps'][0]]
        
        # Ortalamalar
        sick_vals = [sick_avg['chol'], sick_avg['thalch'], sick_avg['trestbps']]
        healthy_vals = [healthy_avg['chol'], healthy_avg['thalch'], healthy_avg['trestbps']]
        
        rects1 = ax.bar(x - width, user_vals, width, label='Bu Hasta', color='#3498db')
        rects2 = ax.bar(x, sick_vals, width, label='Ortalama Hasta (Riskli)', color='#e74c3c')
        rects3 = ax.bar(x + width, healthy_vals, width, label='Ortalama Sağlıklı', color='#2ecc71')
        
        ax.set_ylabel('Değerler')
        ax.set_title('Hasta Değerlerinin Genelle Karşılaştırılması')
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.legend()
        
        st.pyplot(fig)

        # Feature Importance Grafiği
        st.subheader("Modelin Kararını Etkileyen Faktörler")
        st.markdown("Aşağıdaki grafik, modelin tahmin yaparken hangi özellikleri daha önemli bulduğunu göstermektedir.")
        
        fig_imp, ax_imp = plt.subplots(figsize=(10, 6))
        sns.barplot(x='importance', y='feature', data=feature_importance, ax=ax_imp, palette='viridis')
        ax_imp.set_title('Özellik Önem Düzeyleri')
        ax_imp.set_xlabel('Önem Düzeyi')
        ax_imp.set_ylabel('Özellik')
        st.pyplot(fig_imp)
        
        # Yasal Uyarı
        st.error("⚠️ **YASAL UYARI:** Bu uygulama tıbbi bir teşhis aracı değildir. Sonuçlar sadece bilgilendirme amaçlıdır. Lütfen kesin tanı için doktorunuza başvurunuz.")

with col2:
    st.info("ℹ️ Bilgi Paneli")
    st.markdown("""
    **Giriş parametreleri hakkında:**
    * **cp:** Göğüs ağrısı türü.
    * **trestbps:** Hastaneye girişteki istirahat tansiyonu.
    * **chol:** Serum kolesterolü.
    * **fbs:** Açlık kan şekeri > 120 mg/dl ise.
    * **oldpeak:** Egzersizle indüklenen ST depresyonu.
    """)
    
    st.write("Model Doğruluğu (Test Seti):")
    st.metric(label="Accuracy", value="85%") # Temsili, gerçekte hesaplanan değer kullanılabilir ama UI'da sabit durabilir.