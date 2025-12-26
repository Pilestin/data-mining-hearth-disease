import streamlit as st

st.set_page_config(
    page_title="Yardım ve Sözlük",
    page_icon="ℹ️",
    layout="wide"
)

st.title("ℹ️ Yardım ve Tıbbi Terimler Sözlüğü")

st.markdown("""
Bu sayfa, Kalp Hastalığı Risk Analizi uygulamasında kullanılan tıbbi terimleri ve uygulamanın nasıl çalıştığını açıklamak için hazırlanmıştır.
""")

st.divider()

st.header("📌 Nasıl Kullanılır?")
st.markdown("""
1.  **Sol Menü**: Sol taraftaki menüden hastaya ait klinik ve demografik bilgileri giriniz.
2.  **Risk Analizi**: Tüm bilgileri girdikten sonra **"Risk Analizi Yap"** butonuna tıklayınız.
3.  **Sonuçlar**: Modelin tahmin sonucunu (Yüksek Risk/Düşük Risk) ve olasılık değerini inceleyiniz.
4.  **Karşılaştırma**: Hastanızın değerlerinin, veri setindeki ortalama "Hasta" ve "Sağlıklı" bireylerle karşılaştırmasını grafik üzerinde görünüz.
""")

st.divider()

st.header("📖 Tıbbi Terimler Sözlüğü")

with st.expander("Gözğs Ağrısı Tipi (CP - Chest Pain)", expanded=True):
    st.markdown("""
    Hastanın şikayet ettiği göğüs ağrısının türüdür. Kalp hastalıklarında en önemli belirtilerden biridir.
    *   **Tipik Anjina (Typical Angina):** Fiziksel efor veya stresle tetiklenen, dinlenmekle geçen klasik göğüs ağrısı.
    *   **Atipik Anjina (Atypical Angina):** Tipik anjinaya benzeyen ancak tüm kriterleri sağlamayan ağrı.
    *   **Anjinal Olmayan Ağrı (Non-anginal Pain):** Kalp kaynaklı olmadığı düşünülen ağrı.
    *   **Asemptomatik (Asymptomatic):** Göğüs ağrısı şikayeti yok (Sessiz iskemi olabilir).
    """)

with st.expander("İstirahat Kan Basıncı (Trestbps)", expanded=True):
    st.markdown("""
    Hastanın hastaneye başvurduğu andaki dinlenme halindeki tansiyon değeridir (mm Hg cinsinden).
    *   **Normal:** 120/80 mm Hg altı.
    *   **Hipertansiyon:** 130-140 mm Hg ve üzeri risk faktörü olabilir.
    """)

with st.expander("Serum Kolesterol (Chol)", expanded=True):
    st.markdown("""
    Kandaki toplam kolesterol miktarıdır (mg/dl).
    *   **İstenen:** < 200 mg/dl
    *   **Sınırda Yüksek:** 200-239 mg/dl
    *   **Yüksek:** > 240 mg/dl
    """)

with st.expander("Açlık Kan Şekeri (FBS)", expanded=True):
    st.markdown("""
    Hastanın aç karnına ölçülen kan şekeridir.
    *   **> 120 mg/dl:** Diyabet riski veya diyabet varlığına işaret edebilir. Kalp hastalığı için risk faktörüdür.
    """)

with st.expander("İstirahat EKG Sonucu (Restecg)", expanded=True):
    st.markdown("""
    Dinlenme halindeyken çekilen Elektrokardiyografi (EKG) sonucudur.
    *   **Normal:** Herhangi bir anormallik yok.
    *   **ST-T Dalga Anormalliği:** Kalp kasının yeterince oksijen alamadığını (iskemi) gösterebilir.
    *   **Sol Ventrikül Hipertrofisi:** Kalbin sol karıncığının kalınlaşması (genelde yüksek tansiyona bağlı).
    """)

with st.expander("Maksimum Kalp Atış Hızı (Thalach)", expanded=True):
    st.markdown("""
    Efor testi sırasında ulaşılan en yüksek kalp atış hızıdır. Genelde kalp hastalığı olanlarda bu değer daha düşük kalabilir.
    """)

with st.expander("Egzersize Bağlı Anjina (Exang)", expanded=True):
    st.markdown("""
    Efor sarf ederken (koşarken, merdiven çıkarken) göğüs ağrısı (anjina) oluşup oluşmadığı.
    *   **Evet:** Yüksek risk göstergesidir.
    """)

with st.expander("ST Depresyonu (Oldpeak) ve Eğimi (Slope)", expanded=True):
    st.markdown("""
    Efor testi (koşu bandı) sırasındaki EKG değişiklikleridir.
    *   **Oldpeak:** Egzersizle oluşan ST segment çökmesi miktarı. Yüksek değerler risklidir.
    *   **Slope:** ST segmentinin eğimi. (Yukarı eğimli genelde iyi, düz veya aşağı eğimli iskemiyi gösterebilir).
    """)

with st.expander("Büyük Damarlar (CA)", expanded=True):
    st.markdown("""
    Floroskopi (anjiyo benzeri görüntüleme) sırasında boyalı madde ile görülebilen tıkalı veya daralmış ana damar sayısı (0-3 arası).
    *   Sayı arttıkça ciddiyet artar.
    """)

with st.expander("Talasemi (Thal)", expanded=True):
    st.markdown("""
    Bir kan bozukluğu türüdür ancak burada kalbe giden kan akışını (perfüzyon) temsil eder.
    *   **Normal:** Kan akışı normal.
    *   **Sabit Kusur (Fixed Defect):** Kalıcı hasar (eski kriz vb.).
    *   **Tersine Çevrilebilir Kusur (Reversable Defect):** İskemi belirtisi (kan akışı bozuk ama düzelebilir).
    """)

st.divider()

st.header("🤖 Model Hakkında")
st.info("""
Bu uygulama **Random Forest** (Rastgele Orman) adı verilen bir makine öğrenimi algoritması kullanmaktadır. 
Model, geçmişteki yüzlerce kalp hastasının verilerinden "öğrenerek", yeni girilen değerlere göre bir risk tahmini yapar.
""")

st.warning("""
**YASAL UYARI:**
Bu uygulama sadece eğitim ve bilgilendirme amaçlıdır. Bir **TIBBİ TANI CİHAZI DEĞİLDİR**.
Burada verilen sonuçlar kesin bir teşhis yerine geçmez. Lütfen sağlık sorunlarınız için mutlaka bir **DOKTORA BAŞVURUNUZ**.
""")
