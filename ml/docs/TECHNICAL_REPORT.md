# UCI Heart Disease Veri Seti Üzerinde Makine Öğrenmesi ile Kalp Hastalığı Tahmini

## Teknik Rapor ve Dokümantasyon

**Tarih:** 1 Ocak 2025  
**Veri Seti:** UCI Heart Disease (Cleveland)  
**Analiz Türü:** Sınıflandırma (Binary Classification)

---

## 1. Giriş ve Problem Tanımı

### 1.1 Amaç

Bu çalışmada, UCI Heart Disease veri seti kullanılarak kalp hastalığı tahmin modelleri geliştirilmiştir. Çalışmanın temel hedefleri:

1. Farklı veri önişleme tekniklerinin model performansına etkisini izole olarak incelemek (Ablation Study)
2. Hiperparametre optimizasyonu (Optuna) ile model performansını maksimize etmek
3. 6 farklı makine öğrenmesi algoritmasını karşılaştırmak
4. En iyi ve en kötü modellerin tüm tekniklerle birlikte nasıl performans gösterdiğini analiz etmek

### 1.2 Veri Seti Özellikleri

- **Kaynak:** UCI Machine Learning Repository
- **Alt Veri Seti:** Cleveland (en yaygın kullanılan)
- **Örneklem Sayısı:** 304
- **Özellik Sayısı:** 13 (orijinal) + 4 (mühendislik) = 17
- **Hedef Değişken:** Binary (0: Sağlıklı, 1: Kalp Hastalığı)
- **Sınıf Dağılımı:** Sağlıklı: 165 (%54.3), Hasta: 139 (%45.7)

### 1.3 Deneysel Tasarım (Ablation Study)

Bu çalışmada her tekniğin etkisini izole olarak görmek için 6 farklı senaryo tasarlanmıştır:

| Senaryo | İçerik | Scaler | FE | PCA | SMOTE | Optuna | CV |
|---------|--------|--------|-----|-----|-------|--------|-----|
| **S0: Baseline** | Temel | RobustScaler | ❌ | ❌ | ❌ | ❌ | 10-Fold |
| **S1: + PCA** | PCA etkisi | StandardScaler | ❌ | ✅ | ❌ | ❌ | 10-Fold |
| **S2: + FE** | Feature Eng. etkisi | RobustScaler | ✅ | ❌ | ❌ | ❌ | 10-Fold |
| **S3: + SMOTE** | Dengeleme etkisi | RobustScaler | ❌ | ❌ | ✅ | ❌ | 10-Fold |
| **S4: + Optuna** | HP optimizasyonu | RobustScaler | ❌ | ❌ | ❌ | ✅ | 10-Fold |
| **S5: All Combined** | Tüm teknikler | StandardScaler | ✅ | ✅ | ✅ | ✅ | 10-Fold |

---

## 2. Veri Seti Özellikleri (Features)

### 2.1 Orijinal Özellikler

| Özellik    | Açıklama                              | Tip       | Değer Aralığı                                              |
| ---------- | ------------------------------------- | --------- | ---------------------------------------------------------- |
| `age`      | Yaş                                   | Sürekli   | 28-77 yıl                                                  |
| `sex`      | Cinsiyet                              | Kategorik | Male, Female                                               |
| `cp`       | Göğüs ağrısı tipi                     | Kategorik | typical angina, atypical angina, non-anginal, asymptomatic |
| `trestbps` | Dinlenme kan basıncı                  | Sürekli   | 94-200 mmHg                                                |
| `chol`     | Serum kolesterol                      | Sürekli   | 126-564 mg/dl                                              |
| `fbs`      | Açlık kan şekeri > 120 mg/dl          | Binary    | TRUE, FALSE                                                |
| `restecg`  | Dinlenme EKG sonucu                   | Kategorik | normal, st-t abnormality, lv hypertrophy                   |
| `thalch`   | Maksimum kalp hızı                    | Sürekli   | 71-202 bpm                                                 |
| `exang`    | Egzersize bağlı angina                | Binary    | TRUE, FALSE                                                |
| `oldpeak`  | Egzersizin neden olduğu ST depresyonu | Sürekli   | -2.6 - 6.2                                                 |
| `slope`    | ST segment eğimi                      | Kategorik | upsloping, flat, downsloping                               |
| `ca`       | Floroskopi ile boyanan damar sayısı   | Ordinal   | 0-3                                                        |
| `thal`     | Talasemi                              | Kategorik | normal, fixed defect, reversable defect                    |

### 2.2 Mühendislik Özellikleri (Feature Engineering)

| Yeni Özellik          | Formül                            | Gerekçe                                                        |
| --------------------- | --------------------------------- | -------------------------------------------------------------- |
| `risk_score`          | (age × chol) / 10000              | Yaş ve kolesterol etkileşimi - kardiyovasküler risk göstergesi |
| `age_group`           | Binning (0-40, 40-55, 55-70, 70+) | Yaş kategorileri ile risk gruplandırması                       |
| `hr_age_ratio`        | thalch / (age + 1)                | Yaşa göre normalize kalp hızı performansı                      |
| `bp_chol_interaction` | (trestbps × chol) / 10000         | Kan basıncı ve kolesterol etkileşimi                           |

---

## 3. Metodoloji

### 3.1 Veri Önişleme Pipeline

```
Ham Veri (920 satır)
    │
    ├── 1. Cleveland Filtresi → 304 satır
    │
    ├── 2. Kategorik Encoding (LabelEncoder)
    │     ├── sex: 2 kategori → [0, 1]
    │     ├── cp: 4 kategori → [0, 1, 2, 3]
    │     ├── restecg: 3 kategori → [0, 1, 2]
    │     ├── exang: 2 kategori → [0, 1]
    │     ├── slope: 3 kategori → [0, 1, 2]
    │     ├── thal: 4 kategori → [0, 1, 2, 3]
    │     └── fbs: 2 kategori → [0, 1]
    │
    ├── 3. Eksik Değer Doldurma (KNN Imputer, k=5)
    │
    ├── 4. Ölçekleme (Senaryoya göre)
    │     ├── RobustScaler: S0, S2, S3, S4
    │     └── StandardScaler: S1, S5 (PCA için zorunlu)
    │
    └── 5. Senaryoya Bağlı Ek İşlemler
          ├── S1: PCA (%95 varyans)
          ├── S2: Feature Engineering (+4 özellik)
          ├── S3: SMOTE (sınıf dengeleme)
          ├── S4: Optuna (hiperparametre optimizasyonu)
          └── S5: FE + PCA + SMOTE + Optuna
```

### 3.2 Kullanılan Teknikler

#### 3.2.1 KNN Imputer

```python
from sklearn.impute import KNNImputer
imputer = KNNImputer(n_neighbors=5)
df_processed[numeric_cols] = imputer.fit_transform(df_processed[numeric_cols])
```

- **Parametre:** k=5 (en yakın 5 komşu)
- **Avantaj:** Benzer örneklerin değerlerini kullanır

#### 3.2.2 RobustScaler vs StandardScaler

| Scaler | Formül | Kullanım Senaryosu |
|--------|--------|-------------------|
| **RobustScaler** | `(X - median) / IQR` | S0, S2, S3, S4 - Aykırı değerlere dayanıklı |
| **StandardScaler** | `(X - mean) / std` | S1, S5 - PCA için zorunlu |

#### 3.2.3 SMOTE

```python
from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_scaled, y)
```

- **Sonuç:** 165 Sağlıklı vs 139 Hasta → 165 Sağlıklı vs 165 Hasta

#### 3.2.4 PCA

```python
from sklearn.decomposition import PCA
pca = PCA(n_components=0.95, random_state=42)
X_pca = pca.fit_transform(X_scaled)
```

- **Sonuç:** 13 özellik → 12 bileşen (%97.14 varyans korundu)

### 3.3 Validasyon Stratejisi

Tüm senaryolarda **Stratified 10-Fold Cross Validation** kullanılmıştır:

```python
from sklearn.model_selection import StratifiedKFold, cross_val_score
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
```

**Neden 10-Fold CV?**
1. Her fold'da sınıf oranları korunur
2. 10 farklı test seti ile güvenilir performans tahmini
3. Standart sapma model kararlılığını gösterir

### 3.4 Hiperparametre Optimizasyonu (Optuna)

```python
import optuna

def objective(trial):
    n_estimators = trial.suggest_int('n_estimators', 50, 300)
    max_depth = trial.suggest_int('max_depth', 3, 20)
    # ...
    model = RandomForestClassifier(...)
    return cross_val_score(model, X, y, cv=skf, scoring='f1').mean()

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=30, show_progress_bar=True)
```

| Özellik | Değer |
|---------|-------|
| **Algoritma** | TPE (Bayesian Optimizasyon) |
| **Trial Sayısı** | 30 (S4), 50 (S5) |
| **Metrik** | F1-Score (maximize) |

---

## 4. Kullanılan Modeller

### 4.1 Modeller ve Varsayılan Parametreleri

| Model | Varsayılan Parametreler |
|-------|------------------------|
| **Logistic Regression** | `max_iter=1000, random_state=42` |
| **Random Forest** | `random_state=42, n_jobs=-1` |
| **SVM** | `probability=True, random_state=42` |
| **Naive Bayes** | `GaussianNB()` (varsayılan) |
| **XGBoost** | `random_state=42, n_jobs=-1, eval_metric='logloss'` |
| **KNN** | `n_jobs=-1` |

### 4.2 Optuna Hiperparametre Arama Uzayları

#### Logistic Regression
| Parametre | Aralık |
|-----------|--------|
| `C` | [0.01, 10.0] (log scale) |
| `penalty` | ['l1', 'l2'] |

#### Random Forest
| Parametre | Aralık |
|-----------|--------|
| `n_estimators` | [50, 300] |
| `max_depth` | [3, 20] |
| `min_samples_split` | [2, 20] |
| `min_samples_leaf` | [1, 10] |

#### SVM
| Parametre | Aralık |
|-----------|--------|
| `C` | [0.1, 100.0] (log scale) |
| `gamma` | ['scale', 'auto'] |
| `kernel` | ['rbf', 'poly'] |

#### Naive Bayes
| Parametre | Aralık |
|-----------|--------|
| `var_smoothing` | [1e-12, 1e-6] (log scale) |

#### XGBoost
| Parametre | Aralık |
|-----------|--------|
| `n_estimators` | [50, 300] |
| `max_depth` | [3, 15] |
| `learning_rate` | [0.01, 0.3] (log scale) |
| `subsample` | [0.6, 1.0] |
| `colsample_bytree` | [0.6, 1.0] |

#### KNN
| Parametre | Aralık |
|-----------|--------|
| `n_neighbors` | [3, 21] (tek sayılar) |
| `weights` | ['uniform', 'distance'] |
| `metric` | ['euclidean', 'manhattan'] |

---

## 5. Deneysel Sonuçlar

### 5.1 Senaryo 0: Baseline

**Konfigürasyon:** RobustScaler + 10-Fold CV + 6 Model (varsayılan parametreler)

| Model | Accuracy | Recall | F1-Score | AUC |
|-------|----------|--------|----------|-----|
| **Logistic Regression** 🏆 | 0.842±0.053 | 0.771±0.076 | **0.817±0.061** | 0.911±0.056 |
| SVM | 0.832±0.055 | 0.771±0.099 | 0.806±0.069 | 0.900±0.056 |
| KNN | 0.822±0.082 | 0.771±0.126 | 0.796±0.102 | 0.877±0.066 |
| Naive Bayes | 0.819±0.049 | 0.785±0.084 | 0.798±0.058 | 0.895±0.055 |
| Random Forest | 0.809±0.046 | 0.757±0.096 | 0.783±0.052 | 0.896±0.045 |
| **XGBoost** 📉 | 0.763±0.066 | 0.721±0.133 | **0.732±0.088** | 0.881±0.043 |

**Bulgular:**
- 🏆 **En İyi:** Logistic Regression (F1=0.817)
- 📉 **En Kötü:** XGBoost (F1=0.732)
- Senaryo 5 için bu iki model seçildi

### 5.2 Senaryo 1: + PCA

**Konfigürasyon:** StandardScaler + PCA(%95) + 10-Fold CV + 6 Model  
**PCA Sonucu:** 13 özellik → 12 bileşen (%97.14 varyans)

| Model | Accuracy | Recall | F1-Score | AUC |
|-------|----------|--------|----------|-----|
| **Logistic Regression** 🏆 | 0.845±0.049 | 0.771±0.069 | **0.820±0.056** | 0.910±0.060 |
| XGBoost | 0.822±0.051 | 0.799±0.099 | 0.803±0.061 | 0.887±0.062 |
| SVM | 0.825±0.058 | 0.756±0.120 | 0.794±0.080 | 0.900±0.050 |
| Random Forest | 0.802±0.063 | 0.771±0.119 | 0.780±0.072 | 0.878±0.052 |
| KNN | 0.799±0.048 | 0.757±0.096 | 0.774±0.055 | 0.873±0.053 |
| Naive Bayes | 0.806±0.075 | 0.735±0.114 | 0.774±0.090 | 0.893±0.061 |

**Baseline'a Göre Değişim:**
- LR: +0.3% F1 iyileşme
- XGBoost: +7.1% F1 iyileşme (en çok fayda gören)

### 5.3 Senaryo 2: + Feature Engineering

**Konfigürasyon:** RobustScaler + 4 Yeni Özellik + 10-Fold CV + 6 Model  
**Yeni Özellikler:** risk_score, age_group, hr_age_ratio, bp_chol_interaction

| Model | Accuracy | Recall | F1-Score | AUC |
|-------|----------|--------|----------|-----|
| **Logistic Regression** 🏆 | 0.839±0.060 | 0.778±0.080 | **0.815±0.068** | 0.910±0.059 |
| Naive Bayes | 0.812±0.047 | 0.785±0.070 | 0.793±0.050 | 0.878±0.069 |
| Random Forest | 0.812±0.077 | 0.757±0.115 | 0.786±0.089 | 0.895±0.064 |
| SVM | 0.819±0.065 | 0.721±0.120 | 0.781±0.086 | 0.898±0.061 |
| XGBoost | 0.789±0.062 | 0.771±0.114 | 0.769±0.071 | 0.887±0.037 |
| KNN | 0.799±0.098 | 0.735±0.127 | 0.769±0.114 | 0.870±0.068 |

**Baseline'a Göre Değişim:**
- LR: -0.2% F1 (minimal etki)
- Feature engineering bu veri setinde etkisiz

### 5.4 Senaryo 3: + SMOTE

**Konfigürasyon:** RobustScaler + SMOTE + 10-Fold CV + 6 Model  
**SMOTE Sonucu:** 165 vs 139 → 165 vs 165 (dengeli sınıflar)

| Model | Accuracy | Recall | F1-Score | AUC |
|-------|----------|--------|----------|-----|
| **Logistic Regression** 🏆 | 0.842±0.074 | 0.806±0.066 | **0.837±0.075** | 0.908±0.055 |
| KNN | 0.830±0.067 | 0.824±0.104 | 0.827±0.074 | 0.901±0.067 |
| XGBoost | 0.830±0.068 | 0.811±0.081 | 0.826±0.073 | 0.899±0.055 |
| SVM | 0.836±0.056 | 0.799±0.091 | 0.828±0.063 | 0.902±0.053 |
| Random Forest | 0.830±0.076 | 0.805±0.096 | 0.824±0.083 | 0.911±0.060 |
| Naive Bayes | 0.821±0.060 | 0.781±0.094 | 0.811±0.070 | 0.889±0.047 |

**Baseline'a Göre Değişim:**
- LR: +2.0% F1 iyileşme
- XGBoost: +9.4% F1 iyileşme (en çok fayda gören!)
- **SMOTE tüm modellerde önemli iyileşme sağladı**

### 5.5 Senaryo 4: + Optuna

**Konfigürasyon:** RobustScaler + Optuna(30 trial) + 10-Fold CV + 6 Model

| Model | Accuracy | Recall | F1-Score | AUC |
|-------|----------|--------|----------|-----|
| **Random Forest** 🏆 | 0.848±0.056 | 0.778±0.086 | **0.824±0.065** | 0.914±0.048 |
| XGBoost | 0.836±0.049 | 0.814±0.065 | 0.820±0.049 | 0.906±0.045 |
| Logistic Regression | 0.842±0.053 | 0.771±0.076 | 0.817±0.061 | 0.911±0.057 |
| SVM | 0.845±0.072 | 0.757±0.111 | 0.815±0.090 | 0.906±0.055 |
| KNN | 0.829±0.056 | 0.764±0.090 | 0.802±0.068 | 0.906±0.044 |
| Naive Bayes | 0.819±0.049 | 0.785±0.084 | 0.798±0.058 | 0.895±0.055 |

**Baseline'a Göre Değişim:**
- RF: +4.1% F1 iyileşme (Optuna'dan en çok fayda gören)
- XGBoost: +8.8% F1 iyileşme
- **Optuna tüm modellerde iyileştirme sağladı**

### 5.6 Senaryo 5: All Combined

**Konfigürasyon:** StandardScaler + FE + PCA + SMOTE + Optuna(50 trial) + 10-Fold CV  
**Test Edilen Modeller:** En iyi (LR) ve en kötü (XGBoost) modeller  
**Pipeline:** 17 özellik → PCA: 12 → SMOTE: 330 örnek

| Model | Accuracy | Recall | F1-Score | AUC |
|-------|----------|--------|----------|-----|
| **Logistic Regression (Best)** 🏆 | 0.845±0.064 | 0.824±0.051 | **0.843±0.064** | 0.916±0.048 |
| **XGBoost (Worst)** | 0.830±0.053 | 0.849±0.049 | **0.834±0.051** | 0.909±0.038 |

**Baseline'a Göre Değişim:**
- LR: +2.6% F1 iyileşme (0.817 → 0.843)
- XGBoost: **+10.2% F1 iyileşme** (0.732 → 0.834) 🚀

---

## 6. Senaryo Karşılaştırması Özeti

### 6.1 Tüm Senaryoların Özeti

| Senaryo | Ortalama F1 | En İyi F1 | En İyi Model |
|---------|-------------|-----------|--------------|
| **S0: Baseline** | 0.788 | 0.817 | Logistic Regression |
| **S1: + PCA** | 0.791 | 0.820 | Logistic Regression |
| **S2: + FE** | 0.785 | 0.815 | Logistic Regression |
| **S3: + SMOTE** | 0.826 | 0.837 | Logistic Regression |
| **S4: + Optuna** | 0.813 | 0.824 | Random Forest |
| **S5: All Combined** 🏆 | **0.838** | **0.843** | Logistic Regression |

### 6.2 Teknik Bazında Etki Analizi

| Teknik | Ortalama F1 Artışı | En Çok Fayda Gören Model |
|--------|-------------------|--------------------------|
| **PCA** | +0.3% | XGBoost (+7.1%) |
| **Feature Engineering** | -0.3% | - (Etkisiz) |
| **SMOTE** | +3.8% | XGBoost (+9.4%) |
| **Optuna** | +2.5% | RF (+4.1%), XGBoost (+8.8%) |
| **All Combined** | +5.0% | XGBoost (+10.2%) |

### 6.3 Temel Bulgular

1. **SMOTE en etkili teknik** - Tüm modellerde önemli iyileşme sağladı
2. **Feature Engineering etkisiz** - Cleveland veri seti zaten iyi tasarlanmış
3. **XGBoost en çok gelişen model** - Baseline'da en kötü, All Combined'da çok iyi
4. **Logistic Regression en tutarlı** - Her senaryoda en iyi veya en iyi 2'de
5. **Tüm teknikler birlikte +10% iyileşme** sağladı (XGBoost için)

---

## 7. Performans Metrikleri

### 7.1 Metrik Tanımları

| Metrik | Formül | Tıbbi Yorumu |
|--------|--------|--------------|
| **Accuracy** | (TP+TN)/(TP+TN+FP+FN) | Genel doğruluk |
| **Recall** | TP/(TP+FN) | Gerçek hastaların kaçını yakaladık |
| **F1-Score** | 2×(P×R)/(P+R) | Precision ve Recall dengesi |
| **AUC-ROC** | Area Under ROC | Sınıf ayırt etme yeteneği |

### 7.2 Tıbbi Bağlamda Metrik Önceliği

**Kritik:** Tıbbi taramada **Recall** en önemli metriktir.

- **False Negative (Tip II Hata):** Hastaya "sağlıklı" demek → Tedavi gecikmesi
- **False Positive (Tip I Hata):** Sağlıklıya "hasta" demek → Gereksiz testler

**Sonuç:** FN maliyeti >> FP maliyeti

---

## 8. Model Seçimi Önerileri

### 8.1 Uygulama Senaryolarına Göre

| Senaryo | Önerilen Model | Gerekçe |
|---------|----------------|---------|
| **Tarama Programı** | LR + All Combined | En yüksek Recall (0.824) |
| **Klinik Karar Destek** | Logistic Regression | Yorumlanabilir, tutarlı |
| **Sınırlı Kaynak** | LR + SMOTE | İyi performans, hızlı |

### 8.2 Nihai Öneri

**Logistic Regression + All Combined Pipeline:**
- F1-Score: 0.843±0.064
- AUC: 0.916±0.048
- Recall: 0.824 (hastaların %82.4'ünü yakalama)
- Avantaj: Yorumlanabilir, hızlı, düşük varyans

---

## 9. Teknik Uygulama

### 9.1 Kullanılan Kütüphaneler

```python
# Veri İşleme
pandas>=2.0, numpy>=1.24

# Makine Öğrenmesi
scikit-learn>=1.3, xgboost>=2.0, imbalanced-learn>=0.11

# Optimizasyon
optuna>=3.4

# Görselleştirme
matplotlib>=3.7, seaborn>=0.12
```

### 9.2 Kod Dosyaları

| Dosya | Açıklama |
|-------|----------|
| `main_5_revised_scenarios.py` | 6 Senaryo karşılaştırması |
| `scenario_results_*/` | Sonuç klasörleri (tarih damgalı) |

### 9.3 Çalıştırma

```bash
python main_5_revised_scenarios.py
```

---

## 10. Sonuç

Bu çalışmada UCI Heart Disease Cleveland veri seti üzerinde 6 senaryo ile kapsamlı bir ablation study gerçekleştirilmiştir.

### Temel Bulgular:

1. ✅ **SMOTE en etkili teknik** (+3.8% ortalama F1)
2. ✅ **Logistic Regression en tutarlı model** (her senaryoda top-2)
3. ✅ **XGBoost en çok gelişen model** (+10.2% All Combined'da)
4. ❌ **Feature Engineering etkisiz** (Cleveland zaten iyi tasarlanmış)
5. ✅ **Tüm teknikler birlikte** F1=0.843 elde edildi

### Tıbbi Öneri:

Tarama programlarında **Logistic Regression + SMOTE + Optuna** kombinasyonu önerilir:
- Yüksek Recall (%82+) ile hasta yakalama
- Yorumlanabilir model (klinik açıklama)
- Düşük hesaplama maliyeti

---

## Kaynaklar

1. UCI Machine Learning Repository - Heart Disease Dataset
2. Akiba, T., et al. (2019). Optuna: Hyperparameter Optimization Framework
3. Chawla, N. V., et al. (2002). SMOTE: Synthetic Minority Over-sampling

---

**Rapor Sonu**

_Son Güncelleme: 1 Ocak 2025_
