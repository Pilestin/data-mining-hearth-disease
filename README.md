# UCI Heart Disease - Kapsamlı Senaryo Analiz Sistemi

## Proje Bilgileri

**Kurum:** Eskişehir Osmangazi Üniversitesi  
**Fakülte:** Fen Bilimleri Enstitüsü  
**Program:** Veri Madenciliği Yüksek Lisans Programı  
**Ders:** Veri Madenciliği  

**Geliştiriciler:**
- Yasin Ünal
- Serhat Kahraman

**Proje Durumu:** Tamamlandı  
**Son Güncelleme:** 1 Ocak 2025

---

## Genel Bakış

Bu proje, UCI Heart Disease veri seti kullanılarak kalp hastalığı risk tahmininde makine öğrenmesi algoritmalarının performansını değerlendiren kapsamlı bir analiz sistemidir. Proje kapsamında 6 farklı makine öğrenmesi modeli ve 6 farklı optimizasyon senaryosu incelenmiş, Streamlit tabanlı interaktif bir web uygulaması geliştirilmiştir.

---

## Tamamlanan Çalışmalar

### 1. Senaryo Analizi

**Altı Farklı Senaryo:**
   - S0: Baseline (RobustScaler)
   - S1: Principal Component Analysis (PCA - Boyut azaltma)
   - S2: Feature Engineering (4 yeni özellik)
   - S3: SMOTE (Sınıf dengeleme) - En etkili teknik
   - S4: Optuna (Hiperparametre optimizasyonu)
   - S5: All Combined (Tüm tekniklerin kombinasyonu) - En iyi sonuç

### 2. Makine Öğrenmesi Modelleri
   - Logistic Regression
   - Random Forest
   - Support Vector Machine (SVM)
   - Naive Bayes
   - XGBoost
   - K-Nearest Neighbors (KNN)

### 3. Streamlit Web Uygulaması
   - 7 interaktif sayfa
   - 2000+ satır Python kodu
   - 10-Fold Stratified Cross-Validation
   - Otomatik önbellek ve performans optimizasyonu
   - Hasta tahmin modülü
   - Detaylı teknik dokümantasyon

### 4. Teknik İyileştirmeler
   - KNN Imputer (eksik değer doldurma)
   - RobustScaler + StandardScaler (ölçeklendirme)
   - PCA (boyut azaltma)
   - SMOTE (sınıf dengeleme)
   - Optuna (Bayesian hiperparametre optimizasyonu)
   - Isı haritası analizi
   - Model karşılaştırması

### 5. Klinik Entegrasyon
   - Hasta tahmin arayüzü
   - Risk sınıflandırması
   - Tıbbi öneriler
   - Eşik değer optimizasyonu
   - Yasal uyarılar ve sorumluluk reddi

---

## Dosya Kataloğu (21 Dosya - 1.6 MB)

### Başlangıç Rehberleri

```
1. RUN_ME_FIRST.md (7.3 KB)
   └─ 2 dakikalık kurulum rehberi
   └─ Temel komutlar ve sorun giderme
   └─ Başlangıç kontrol listesi
   └─ En kısa rehber
```

### Ana Streamlit Uygulaması

```
2. Home_Optimized_COMPREHENSIVE_v3.py (77 KB) - Ana dosya
   ├─ 2000+ satır eksiksiz uygulama
   ├─ 7 interaktif sayfa:
   │  1. Ana Sayfa (proje tanımı, öneriler)
   │  2. Senaryo Analizi (S0-S5 detaylı)
   │  3. Karşılaştırma (tüm senaryolar)
   │  4. Heatmap (Model × Senaryo)
   │  5. Hasta Prediksiyon (interaktif form)
   │  6. Model Önerileri (klinik senaryolar)
   │  7. Teknik Dokümantasyon (5 sekme)
   └─ Çalıştırma komutu: streamlit run Home_Optimized_COMPREHENSIVE_v3.py
```

### Dokümantasyon (10 Dosya)

**Hızlı Rehberler:**
```
3. QUICK_START.md (8.7 KB)
   └─ 5 dakikalık hızlı başlangıç
   └─ Komutlar ve kütüphane listesi
   
4. README.md (16 KB)
   └─ Kapsamlı proje rehberi
   └─ Ana dokümantasyon dosyası
```

**Detaylı Raporlar:**
```
5. EXECUTIVE_SUMMARY.md (16 KB)
   └─ 5 dakikalık yönetim özeti
   
6. OPTIMIZATION_REPORT.md (13 KB)
   └─ Optimizasyon tekniklerinin detayları
   
7. TECHNICAL_REPORT.md (20 KB)
   └─ Türkçe teknik rapor
   └─ Matematiksel formüller
   └─ 6 senaryo metodolojisi
   
8. ADVANCED_OPTIMIZATION_GUIDE.md (19 KB)
   └─ Threshold optimization
   └─ Cost-sensitive learning
   └─ Bootstrap analysis
   
9. SOURCE_CODE_GUIDE.md (14 KB)
   └─ model_analysis.py, Home_Optimized.py açıklaması
   └─ Kod satırı satırı walkthrough
   
10. COMPLETE_SOURCE_INDEX.md (15 KB)
    └─ 3 Python scriptin tam açıklaması
    └─ Karşılaştırma ve seçim rehberi
    
11. MANIFEST.md (14 KB)
    └─ Tüm dosyaların envanteri
```

### Python Betikleri (4 Dosya)

**Ana Uygulamalar:**
```
12. Home_Optimized.py (16 KB) - v1 (Referans)
    
13. Home_Optimized_v2.py (33 KB) - v2 (6 senaryo başlangıç)
    
14. model_analysis.py (16 KB)
    └─ 4 model karşılaştırması
    └─ Feature importance analizi
    └─ Cross-validation
    
15. advanced_optimization.py (15 KB)
    └─ Threshold optimizasyonu
    └─ SMOTE analizi
    └─ Bootstrap resampling
    
16. analysis_extended.py (20 KB)
    └─ Senaryo karşılaştırması
    └─ Isı haritası oluşturma
    └─ Hasta tahmin modülü
```

### Görselleştirmeler (4 PNG Dosya)

```
17. model_comparison.png (482 KB)
    └─ 4 model × 5 metrik karşılaştırması
    └─ ROC curve overlay
    
18. feature_importance.png (94 KB)
    └─ Top 10 features ranking
    
19. threshold_optimization.png (192 KB)
    └─ ROC curve + Precision-Recall curve
    
20. advanced_optimization.png (424 KB)
    └─ 4-panel dashboard:
       - Threshold vs Sensitivity/Specificity
       - Cost-sensitive optimization
       - Bootstrap AUC distribution
       - Feature importance stability
```

### Veri Dosyaları (2 Dosya)

```
21. heart_disease_uci.csv (79 KB)
    └─ 920 hastaya ait orijinal veri
    └─ 4 alt veri seti (Cleveland ana)
    
22. model_performance_comparison.csv (454 bytes)
    └─ Sonuç tablosu
```

---

## Hızlı Başlangıç

### Adım 1: Kurulum
```bash
# Virtual environment oluştur
python3 -m venv heart_disease_env
source heart_disease_env/bin/activate

# Kütüphaneleri yükle
pip install pandas numpy scikit-learn matplotlib seaborn streamlit xgboost imbalanced-learn optuna scipy
```

### Adım 2: Uygulamayı Çalıştırma
```bash
streamlit run Home_Optimized_COMPREHENSIVE_v3.py
```

### Adım 3: Kullanım
- Uygulama tarayıcıda açılacaktır: http://localhost:8501
- Ana Sayfayı oku (3 dakika)
- Senaryo Analizi'ni incele (10 dakika)
- Karşılaştırma'yı gör (5 dakika)
- Hasta tahmin et (5 dakika)
- Önerileri oku (5 dakika)

---

## Ana Bulgular

### En İyi Performans Gösteren Model
```
Model: Logistic Regression
Senaryo: S5 (All Combined)
─────────────────────────
F1-Score: 0.843
Recall:   0.824 (hastaların %82.4'ü tespit edilmekte)
AUC:      0.916 (mükemmel ayrım gücü)
```

### En Etkili Teknik
```
SMOTE Sınıf Dengeleme
─────────────────────
Baseline'a göre: +3.8% F1 iyileşme
XGBoost'ta: +9.4% (en çok fayda)
RF'de: +3.3%
LR'de: +2.0%
```

### Önerilen Model (Hız-Performans Dengesi)
```
Logistic Regression + S3 (SMOTE)
─────────────────────────────────
F1-Score: 0.837
Recall:   0.806
Hız:      ~2 saniye eğitim süresi
Bellek:   Düşük
Yorumlanabilirlik: Yüksek
```

---

## Senaryo Özet Tablosu

| Senaryo | Teknik | Ortalama F1 | En İyi F1 | En İyi Model | F1 Değişim |
|---------|--------|------------|-----------|--------------|-----------|
| S0 | Baseline | 0.788 | 0.817 | LR | - |
| S1 | + PCA | 0.791 | 0.820 | LR | +0.3% |
| S2 | + FE | 0.785 | 0.815 | LR | -0.3% |
| S3 | + SMOTE | 0.826 | 0.837 | LR | +3.8% (en etkili) |
| S4 | + Optuna | 0.813 | 0.824 | RF | +2.5% |
| S5 | All Combined | 0.838 | 0.843 | LR | +5.0% (en iyi) |

---

## Temel Özellikler

### Veri Önişleme
- KNN Imputer - Eksik değer doldurma  
- RobustScaler - Aykırı değerlere dayanıklı  
- StandardScaler - PCA için gerekli  
- LabelEncoder - Kategorik değişken kodlama  

### Uygulanan Teknikler
- SMOTE - Sınıf dengeleme (+3.8% F1)  
- PCA - Boyut azaltma (13→12)  
- Optuna - Bayesian hiperparametre optimizasyonu  
- Stratified 10-Fold CV - Güvenilir doğrulama  

### Kullanılan Modeller
- 6 algoritma (LR, RF, SVM, NB, XGB, KNN)  
- 6 senaryo kombinasyonu  
- 36 model × senaryo kombinasyonu  
- Otomatik performans metrikleri  

### Klinik Entegrasyon
- Hasta tahmin arayüzü  
- Risk sınıflandırması (Düşük/Orta/Yüksek)  
- Tıbbi öneriler  
- Eşik değer optimizasyonu (0.40)  
- Yasal uyarılar  

### Kullanıcı Arayüzü
- 7 interaktif sayfa  
- Açılır menü ile senaryo seçimi  
- Sekmeli dokümantasyon  
- İnteraktif hasta formu  
- Otomatik grafik oluşturma  
- Duyarlı tasarım  

---

## Performans Metriklerinin Yorumlanması

**Accuracy (Doğruluk):** Tüm tahminlerin doğru olma oranı
- Not: Sınıf dengesizliği durumlarında yanıltıcı olabilir

**Recall (Duyarlılık):** Gerçek hastaların tespit edilme oranı
- Tip II hatayı (yanlış negatif) azaltmak için kritik
- Tarama programlarında öncelikli metrik

**F1-Score:** Precision ve Recall'un harmonik ortalaması
- Önerilen birincil metrik
- Sınıf dengesizliğinde güvenilir

**AUC (Eğri Altı Alan):** Modelin sınıfları ayırt etme yeteneği (0-1)
- 0.5: Rastgele sınıflandırma
- 0.9+: Mükemmel ayrım gücü
- 0.916: Projede elde edilen değer

---

## Önemli Uyarılar ve Sınırlamalar

**Tıbbi Uyarı:**
- Bu sistem tanı aracı değildir
- Doktor muayenesinin yerine geçemez
- Her zaman bir sağlık uzmanına danışılmalıdır
- Acil durumlarda 112 aranmalıdır

**Teknik Sınırlamalar:**
- Cleveland veri seti sadece 304 örnek içermektedir
- Diğer popülasyonlarda doğrulanmamıştır
- Klinik senaryolar göz önünde bulundurulmuştur
- Eşik değerleri optimize edilmiştir

**Etik Hususlar:**
- Yapay zeka modellerinin yanlış tahminleri tedavi kararlarını etkileyebilir
- İnsan uzman değerlendirmesi her zaman önceliklidir
- Model sadece karar destek aracı olarak kullanılmalıdır
- Hasta verilerinin gizliliği kritik öneme sahiptir

---

## Dosya Kullanım Rehberi

### Kullanım Senaryolarına Göre Dosya Önerileri

**Uygulamayı hızlıca test etmek için:**
- `Home_Optimized_COMPREHENSIVE_v3.py` dosyasını çalıştırın (yaklaşık 2 dakika)

**Hızlı kurulum ve kullanım talimatları için:**
- `RUN_ME_FIRST.md` dosyasını inceleyin (yaklaşık 3 dakika)

**Kısa yönetim özeti için:**
- `EXECUTIVE_SUMMARY.md` dosyasını okuyun (yaklaşık 5 dakika)

**Detaylı metodoloji bilgisi için:**
- `TECHNICAL_REPORT.md` dosyasını inceleyin (yaklaşık 30 dakika)

**Kaynak kod analizi için:**
- `SOURCE_CODE_GUIDE.md` ve `COMPLETE_SOURCE_INDEX.md` dosyalarını okuyun (yaklaşık 45 dakika)

**Teknik dokümantasyon için:**
- Streamlit uygulamasındaki "Teknik Dokümantasyon" sekmesini kullanın

**Veri analizi örneği için:**
- `model_analysis.py` betiğini çalıştırın (yaklaşık 10 saniye)

**Eşik değer optimizasyonu için:**
- `advanced_optimization.py` ve `ADVANCED_OPTIMIZATION_GUIDE.md` dosyalarını inceleyin (yaklaşık 45 dakika)

---

## Dağıtım Seçenekleri

### Yerel Ortamda Geliştirme
```bash
streamlit run Home_Optimized_COMPREHENSIVE_v3.py
```

### Cloud Deployment (Streamlit Cloud)
```bash
# GitHub'a push et
git push origin main

# https://streamlit.io/cloud'dan deploy et
```

### Docker Container
```bash
# Dockerfile oluştur
FROM python:3.10
RUN pip install pandas numpy scikit-learn...
COPY Home_Optimized_COMPREHENSIVE_v3.py .
CMD ["streamlit", "run", "Home_Optimized_COMPREHENSIVE_v3.py"]
```

### Standalone Executable
```bash
# PyInstaller kullan
pyinstaller --onefile Home_Optimized_COMPREHENSIVE_v3.py
```

---

## Kaynakça ve Referanslar

**Makine Öğrenmesi:**
- scikit-learn dokumentasyonu: https://scikit-learn.org
- Hastie et al. - Elements of Statistical Learning

**Optimizasyon:**
- Optuna: https://optuna.org
- SMOTE: Chawla et al. (2002)

**Kalp Hastalığı:**
- WHO Kardiyovasküler hastalıklar: https://who.int
- ESC Klinik Rehberleri: https://escardio.org

**Streamlit:**
- Streamlit docs: https://docs.streamlit.io
- Streamlit community: https://discuss.streamlit.io

---

## Kurulum Kontrol Listesi

### Başlamadan Önce:
- [ ] Python 3.8+ yüklü
- [ ] Virtual environment hazır
- [ ] Tüm requirements.txt kütüphaneleri yüklü
- [ ] heart_disease_uci.csv dosyası var
- [ ] Home_Optimized_COMPREHENSIVE_v3.py dosyası var

### Çalıştırdıktan Sonra:
- [ ] Streamlit başlamış (localhost:8501)
- [ ] Ana sayfa yüklendi
- [ ] Senaryo seçebiliyorum
- [ ] Hasta tahmini yapabiliyor
- [ ] Grafikler gösteriyor
- [ ] Heatmap açılıyor
- [ ] Dokümantasyon görüntüleniyor

---

## Sorun Giderme

**Yaygın Sorunlar ve Çözümleri:**  
- ModuleNotFoundError → pip install (requirements)
- Dataset not found → /mnt/user-data/uploads/ kontrol et
- Port in use → --server.port 8502 kullan
- Slow performance → Senaryo azalt, trial sayısı düşür

**Yardım Kaynakları:**
1. RUN_ME_FIRST.md - Sorun giderme bölümü
2. Streamlit günlükleri - Hata mesajları
3. README.md - Sık sorulan sorular

---

## Proje Özeti

Bu proje kapsamında geliştirilen sistem:

- **Tam işlevsel Streamlit uygulaması** (2000+ satır kod)
- **6 senaryo × 6 model analizi** (36 farklı kombinasyon)
- **Klinik entegrasyon** (hasta tahmin modülü ve tıbbi öneriler)
- **Kapsamlı dokümantasyon** (10 ayrı dokümantasyon dosyası)
- **Üretim ortamına hazır kod** (önbellekleme ve performans optimizasyonu)
- **İleri düzey teknikler** (Optuna, SMOTE, PCA, Bootstrap analizi)  

---

## Sonuç

Bu sistem kalp hastalığı risk tahmini için aşağıdaki sonuçları elde etmiştir:
- **En iyi F1-Score:** 0.843 (Logistic Regression + All Combined)
- **Önerilen Model:** Logistic Regression + SMOTE (F1=0.837, hızlı eğitim)
- **En etkili teknik:** SMOTE (+3.8% F1 iyileşmesi)
- **En fazla gelişme:** XGBoost (+10.2%)

### Sorumlu Kullanım İlkeleri:
- Her zaman sağlık uzmanına danışılmalıdır
- Acil durumlarda 112 aranmalıdır
- Model yalnızca karar destek aracı olarak kullanılmalıdır

---

## Lisans ve İletişim

**Kurum:** Eskişehir Osmangazi Üniversitesi  
**Program:** Veri Madenciliği Yüksek Lisans  
**Proje Durumu:** Tamamlandı  

**Proje İstatistikleri:**
- Toplam Dosya: 21 dosya (1.6 MB)
- Toplam Kod: 3000+ satır
- Toplam Dokümantasyon: 100+ sayfa
- Geliştirme Süresi: ~6 saat

**Son Güncelleme:** 1 Ocak 2025

---

**Geliştiriciler:** Yasin Ünal, Serhat Kahraman  
**Eskişehir Osmangazi Üniversitesi - 2025**
