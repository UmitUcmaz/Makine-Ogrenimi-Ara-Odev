# Müşteri Churn (Terk) Tahmini - ML Ara Ödev

Bu proje, müşteri terk (**Churn**) durumunu tahmin etmeye yönelik temel makine öğrenimi akışını pratik etmek amacıyla geliştirilmiştir. Kirli ve yapay olarak üretilmiş bir veri seti üzerinde veri temizleme, öznitelik mühendisliği, korelasyon bazlı değişken seçimi ve model karşılaştırma adımları uygulanmıştır.

---

## 📌 Proje Akışı & Adımları

1. **Veri Ön İşleme (Data Cleaning):**
* **Eksik Değerler (Null):** Sayısal sütunlar medyan, kategorik sütunlar mod değerleri ile dolduruldu.
* **Mükerrer Kayıtlar:** Tekrar eden veriler veri setinden çıkarıldı.
* **Aykırı Değerler (Outliers):** IQR yöntemiyle tespit edilen aykırı değerler temizlendi.
* **Mantık Dışı Değerler:** Yaş, gelir, abonelik süresi ve destek talebi sayısındaki negatif/geçersiz değerler filtrelendi.


2. **Öznitelik Mühendisliği (Feature Engineering):**
* `aylik_destek_talebi`: Müşterinin aylık ortalama destek talebi oranı.
* `hayat_boyu_sadakat_orani`: Abonelik süresinin müşteri yaşına oranı.


3. **Korelasyon & Özellik Seçimi (Feature Selection):**
* Hedef değişken (`churn`) ile öznitelikler arasındaki korelasyonlar hesaplandı.
* Mutlak korelasyon değeri **0.10'un altında kalan** zayıf öznitelikler elendi.


4. **Encoding, Bölme ve Ölçeklendirme:**
* Kategorik değişkenler (`sehir`, `uyelik_tipi`) **One-Hot Encoding** yöntemiyle dönüştürüldü.
* Veri seti **%64 Train**, **%16 Validation** ve **%20 Test** olacak şekilde ayrıldı.
* Sayısal öznitelikler **StandardScaler** ile ölçeklendirildi.


5. **Model Eğitimi & Hiperparametre Seçimi (Validation Seti ile):**
* **Logistic Regression:** L1 ve L2 penalizasyonları doğrulama seti başarısına göre karşılaştırıldı.
* **KNN:** $k \in [3, 14]$ aralığında deneme yapılarak en yüksek validation başarımına sahip $k$ değeri seçildi.
* **Decision Tree:** `max_depth` $[1, 10]$ aralığında optimize edildi.



---

## 🛠️ Kurulum & Gereksinimler

Proje **Python 3** ortamında hazırlanmış olup aşağıdaki kütüphaneleri kullanmaktadır:

```bash
pip install pandas scikit-learn matplotlib

```

---

## 📊 Model Performans Karşılaştırması

Model değerlendirmeleri test seti üzerinde `Accuracy`, `Precision`, `Recall` ve `Confusion Matrix` metrikleri ile gerçekleştirilmiştir.

### Test Seti Sonuçları

| Model | Accuracy (Doğruluk) | Precision (Sınıf 1) | Recall (Sınıf 1) |
| --- | --- | --- | --- |
| **Logistic Regression** | %63.0 | %58.0 | - |
| **KNN** | %63.0 | %67.0 | - |
| **Decision Tree** | **%68.0** | **%75.0** | - |

---

## 📈 Karmaşıklık Matrisi (Confusion Matrix)

Modellerin test seti üzerindeki tahmin doğruluğunu görselleştirmek için üretilen karmaşıklık matrisleri:

```text
  Logistic Regression            KNN                Decision Tree
     [[16  5]                 [[18  3]                [[19  2]
      [ 9  7]]                 [11  5]]                [10  6]]

```

---

## 📝 Sonuç ve Değerlendirme

* Test seti üzerinde en yüksek genel doğruluk (**%68 Accuracy**) ve en yüksek pozitif sınıf hassasiyetini (**%75 Precision**) **Decision Tree** modeli sağlamıştır.
* Müşteri terkini tespit etmede (Churn = 1) en düşük yanlış pozitif oranına sahip model Decision Tree olmuştur.
* Küçük ölçekli ve gürültülü (kirli) veri setlerinde ağaç tabanlı modellerin, ölçeklendirilmiş veri üzerinde doğrusal veya mesafe tabanlı modellere göre daha kararlı sonuç ürettiği gözlemlenmiştir.
