# UCI Kalp Hastalığı Veri Seti - Makine Öğrenmesi Projesi

## 📋 Proje Açıklaması

Bu proje, UCI Machine Learning Repository'deki kalp hastalığı veri setini kullanarak kapsamlı bir istatistiksel analiz ve makine öğrenmesi modelleme çalışmasıdır. Yüksek lisans dersi projesi kapsamında geliştirilmiştir.

**Yazarlar:** Yasin Ünal ve Serhat Kahraman  
**Tarih:** Aralık 2025

## 🎯 Proje Hedefleri

1. **İstatistiksel Analiz**: Veri setinin detaylı istatistiksel analizi
2. **Model Testi - Senaryo 1**: Temel makine öğrenmesi modellerinin test edilmesi
3. **Model Testi - Senaryo 2**: Gelişmiş veri işleme teknikleri ile optimizasyon

## 📊 İki Senaryo Yaklaşımı

### Senaryo 1: Temel Model Testleri
- **Amaç**: Modellerin ham veri üzerinde doğrudan performansını ölçmek
- **Veri İşleme**: Sadece temel hazırlık (encoding, eksik değer)
- **Modeller**:
  - Logistic Regression
  - Decision Tree
  - Random Forest
  - Support Vector Machine (SVM)
  - K-Nearest Neighbors (KNN)
  - Naive Bayes
  - Gradient Boosting
  - AdaBoost
  - XGBoost (varsa)

### Senaryo 2: Gelişmiş Veri İşleme ve Optimizasyon
- **Amaç**: Veri bilimi tekniklerinin model performansına etkisini görmek
- **Teknikler**:
  - **SMOTE**: Veri dengesizliğini giderme
  - **Feature Selection**: En önemli özellikleri seçme
  - **Optuna**: Hiperparametre optimizasyonu
- **Modeller**: Random Forest, Logistic Regression, Gradient Boosting, SVM, XGBoost

## 📈 Değerlendirme Metrikleri

Her model şu metriklerle değerlendirilir:
- **Accuracy**: Genel doğruluk
- **Precision**: Pozitif tahminlerin doğruluğu
- **Recall (Sensitivity)**: Gerçek pozitifleri yakalama oranı
- **F1-Score**: Precision ve Recall'un harmonik ortalaması
- **ROC-AUC**: Sınıflandırıcının ayırt etme gücü
- **Confusion Matrix**: Detaylı hata analizi

## 📁 Proje Yapısı

```
data_mining/
│
├── main.py                      # Ana analiz ve model test scripti
├── requirements.txt             # Gerekli Python kütüphaneleri
├── README.md                    # Proje dokümantasyonu
│
├── data/
│   └── heart_disease_uci.csv   # Veri seti
│
├── img/                         # Görselleştirme çıktıları
│   ├── 01_data_quality_analysis.png
│   ├── 02_descriptive_statistics_distributions.png
│   ├── ...
│   └── 14_feature_importance.png
│
└── results/                     # Sonuç dosyaları
    └── detailed_results.csv     # Detaylı model sonuçları
```

## 🚀 Kurulum ve Çalıştırma

### 1. Gerekli Kütüphaneleri Yükleme

```powershell
pip install -r requirements.txt
```

### 2. Projeyi Çalıştırma

```powershell
python main.py
```

## 📊 Çıktılar

### İstatistiksel Analiz Görselleri
1. Veri kalitesi ve eksik değer analizi
2. Tanımlayıcı istatistikler ve dağılımlar
3. Outlier (aykırı değer) analizi
4. Korelasyon matrisi
5. Kategorik değişken analizleri
6. Pair plot ve violin plot'lar
7. Yaş bazlı detaylı analizler

### Makine Öğrenmesi Görselleri
1. Senaryo karşılaştırması
2. Confusion matrix'ler (her model için)
3. ROC eğrileri
4. Precision-Recall eğrileri
5. Feature importance grafiği

### Sonuç Dosyaları
- `results/detailed_results.csv`: Tüm modellerin detaylı performans metrikleri

## 🔬 Kullanılan Kütüphaneler

- **Veri İşleme**: pandas, numpy
- **Görselleştirme**: matplotlib, seaborn
- **İstatistiksel Analiz**: scipy
- **Makine Öğrenmesi**: scikit-learn
- **Veri Dengeleme**: imbalanced-learn (SMOTE)
- **Hiperparametre Optimizasyonu**: optuna
- **Gradient Boosting**: xgboost

## 📖 Veri Seti Hakkında

**Kaynak**: UCI Machine Learning Repository  
**Veri Seti**: Heart Disease (Cleveland)  
**Özellik Sayısı**: 14  
**Hedef Değişken**: Kalp hastalığı varlığı (binary: 0=sağlıklı, 1=hastalıklı)

## 💡 Önemli Notlar

1. **Klinik Önemi**: Kalp hastalığı teşhisinde **Recall (Sensitivity)** metriği kritiktir. False Negative (hastalığı kaçırma) riski minimize edilmelidir.

2. **Veri Dengesizliği**: Senaryo 2'de SMOTE kullanarak sınıf dengesizliği giderilir.

3. **Overfitting Kontrolü**: Cross-validation skorları train-test farklarını gösterir.

4. **Feature Selection**: En önemli özelliklerin belirlenmesi model yorumlanabilirliğini artırır.

## 🏆 Beklenen Sonuçlar

- Senaryo 1: Baseline performans değerlendirmesi
- Senaryo 2: SMOTE ve optimizasyon ile geliştirilmiş performans
- Karşılaştırmalı analiz ile en uygun yaklaşımın belirlenmesi

## 📝 Lisans

Bu proje eğitim amaçlıdır ve akademik kullanım içindir.

## 👥 Katkıda Bulunanlar

- **Yasin Ünal**: Proje geliştirme ve analiz
- **Serhat Kahraman**: Proje geliştirme ve analiz

## 📧 İletişim

Sorularınız için proje ekibiyle iletişime geçebilirsiniz.

---

**Not**: Proje çalıştırıldığında tüm analizler otomatik olarak gerçekleştirilir ve sonuçlar ilgili klasörlere kaydedilir.
