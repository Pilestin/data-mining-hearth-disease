
# Kalp Hastalığı Risk Analizi

Yapay zeka destekli kalp hastalığı risk tahmin sistemi. Bu proje, makine öğrenimi yöntemleri kullanarak hastaların klinik verilerine dayanarak kalp hastalığı riskini tahmin eder.

## Proje Yapısı

```
data-mining-hearth-disease/
├── data/
│   └── heart_disease_uci.csv
├── results/
│   └── advanced_optimization.png
│   └── feature_importance.png
│   └── model_comparison.png
│   └── threshold_optimization.png
├── pages/
│   └── 1_Yardım_ve_Sözlük.py
├── Home.py
├── advanced_optimization.py
├── model_analysis.py
└── requirements.txt
```

## Kullanılan Yöntemler

Uygulama, her başlatıldığında üç farklı makine öğrenimi modelini eğitip bunları bir topluluk modeli (ensemble) olarak birleştirir:

### 1. Random Forest Classifier
- 200 karar ağacından oluşan ensemble model
- Maksimum derinlik: 10
- Sınıf dengesizliği için ağırlıklandırma aktif

### 2. Gradient Boosting Classifier
- 200 iterasyon ile kademeli öğrenme
- Öğrenme oranı: 0.01
- Maksimum derinlik: 3
- Erken durdurma mekanizması aktif

### 3. Logistic Regression
- Maksimum iterasyon: 1000
- Sınıf ağırlıkları dengelenmiş
- Ölçeklendirilmiş özellikler kullanılır

### 4. Voting Ensemble
Üç modelin tahminlerini soft voting (olasılık tabanlı) yöntemiyle birleştirir ve nihai tahmini üretir.

## Model Eğitim Süreci

`load_and_train_optimized_model()` fonksiyonu `@st.cache_resource` dekoratörü ile işaretlenmiştir. Bu, modelin önbellekte tutulmasını sağlar:

**Eğitim Zamanlaması:**
- İlk çalıştırma: Uygulama başlatıldığında modeller sıfırdan eğitilir
- Sonraki istekler: Önbellekteki model kullanılır, yeniden eğitim yapılmaz
- Yeniden başlatma: Uygulama yeniden başlatıldığında önbellek temizlenir ve modeller tekrar eğitilir

**Veri Hazırlama:**
- Eksik değerler: Kategorik değişkenler için mod, sayısal değişkenler için medyan kullanılır
- Kodlama: Kategorik değişkenler Label Encoding ile sayısala dönüştürülür
- Bölme: %80 eğitim, %20 test verisi (stratified sampling)
- Ölçeklendirme: Logistic Regression için StandardScaler uygulanır

## Model Performansı

Uygulama aşağıdaki metrikleri raporlar:
- Test doğruluğu (Accuracy)
- ROC-AUC skoru
- 5-katlı çapraz doğrulama sonuçları (ortalama ve standart sapma)

## Kurulum

```bash
pip install -r requirements.txt
```

## Kullanım

Streamlit uygulamasını başlatmak için:

```bash
streamlit run Home.py
```

## Özellikler

- Türkçe arayüz
- Detaylı parametre açıklamaları ve yardım metinleri
- Görsel karşılaştırmalı analiz
- Risk seviyesi değerlendirmesi (Düşük/Orta/Yüksek)
- Özellik önem sıralaması
- Tıbbi uyarılar ve öneriler

## Not

Bu uygulama eğitim ve araştırma amaçlıdır. Tıbbi teşhis aracı olarak kullanılmamalıdır. Sağlık sorunları için mutlaka uzman bir hekime başvurunuz.