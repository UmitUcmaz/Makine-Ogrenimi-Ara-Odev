# Makine-Ogrenimi-Ara-Odev

## Amaç
* ML modeli oluşturma akışını pratik etmek

## Veri Seti
* Yapay Zekaya üretilen 200 satırlık veri seti. ("musteri_churn_kirli_veri_seti.csv").
* Veri setinde eksik, tekrar eden, aykırı ve mantık dışı değerler mevcuttur.

## Adımlar
1. Veri okuma
2. Temel veri ön işleme (Eksik, Tekrar eden, Aykırı ve Mantık dışı Değerler)
3. Öznitelik üretme
4. Hedef değişken ile öznitelikler arasındaki korelasyonları inceleme
5. Mutlak korelasyon değerine göre yüksek olan özniteliklerin seçilmesi (Feature selection)
6. Kategorik değişkenleri One-Hot Encoding ile sayısal forma dönüştürme
7. Train-validation-test bölme
8. Veri Ölçeklendirme
9. Model eğitimi
10. Model Test ve Sınıflandırma metrikleriyle değerlendirme

## Kurulumlar
* **Adım 1-6 (Veri Okuma, Ön İşleme, Öznitelik Mühendisliği, Encoding):** Pandas
* **Adım 7-10 (Veri Bölme, Ölçeklendirme, Model Eğitimi, Test ve Değerlendirme):** scikit-learn

```bash
pip install -r requirements.txt
