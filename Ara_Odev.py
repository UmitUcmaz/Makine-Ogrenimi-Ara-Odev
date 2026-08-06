"""
Amaç:
    - ML modeli oluşturma akışını pratik etmek

Veri Seti:
    - Yapay Zekaya üretilen 200 satırlık veri seti. ("musteri_churn_kirli_veri_seti.csv")
    - Veri setinde eksik, tekrar eden, ayrkırı ve mantık dışı değerler mevcuttur.

Adımlar:
    1. Veri okuma
    2. Temel veri ön işleme (Eksik, Tekrar eden , Aykırı ve Mantık dışı Değerler)
    3. Öznitelik üretme
    4. Hedef değişken ile öznitelikler arasındaki korelasyonları inceleme
    5. Mutlak korelasyon değerine göre yüksek olan özniteliklerin seçilmesi (Feature selection)
    6. Kategorik değişkenleri One-Hot Encoding ile sayısal forma dönüştürme
    7. Train-validation-test bölme
    8. Veri Ölçeklendirme
    9. Model eğitimi 
    10. Model Test ve Sınıflandırma metrikleriyle değerlendirme 

Kurulumlar:
Adım 1-6 (Veri Okuma, Ön İşleme, Öznitelik Mühendisliği, Encoding): Pandas
Adım 7-10 (Veri Bölme, Ölçeklendirme, Model Eğitimi, Test ve Değerlendirme): scikit-learn

pip install pandas scikit-learn
"""

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report , ConfusionMatrixDisplay

# 1. Veri okuma
df = pd.read_csv("musteri_churn_kirli_veri_seti.csv")

print(f"df.shape : \n{df.shape}")
print(f"df.info() : \n{df.info()}")
print(f"df.describe() : \n{df.describe()}")
print(f"df.head() : \n{df.head()}")

# 2. Temel veri ön işleme
# 2.1 Null veri kontrolü
print(f"Null veri Sayısı : \n{df.isnull().sum()}")

# 2.2 Null veri doldurma
sayisal_sutunlar = ["yas", "gelir", "abonelik_suresi", "destek_talebi_sayisi"]
katagorik_sutunlar = ["sehir","uyelik_tipi"]

print(f"{df["sehir"].unique()}")
print(f"{df["uyelik_tipi"].unique()}")

print(f"sayisal_sutunlar : {sayisal_sutunlar}")
print(f"katagorik_sutunlar : {katagorik_sutunlar}")

for sutun in sayisal_sutunlar:
    medyan = df[sutun].median()
    df[sutun] = df[sutun].fillna(medyan)

for sutun in katagorik_sutunlar:
    medyan = df[sutun].mode()[0]
    df[sutun] = df[sutun].fillna(medyan)

print(f"Temizlik sonrası Null veri Sayısı : \n{df.isnull().sum()}")

# 2.3 Duplicate veri kontrol

tekrarlayan_kayitlar = df[df.duplicated()]
print(f"Tekrar Eden Veri Sayısı : {len(tekrarlayan_kayitlar)}" )

df_2 = df.drop_duplicates() 
print(f"df_ready.shape : \n{df_2.shape}")
"""
(195, 8)
"""

# 2.4 IQR yöntemiyle aykırı değerleri tespit etme

aykiri_deger_maskesi = pd.Series(False, index = df_2.index)

for sutun in sayisal_sutunlar:

    q1 = df_2[sutun].quantile(0.25)
    q3 = df_2[sutun].quantile(0.75)

    iqr = q3 - q1

    alt_sinir = q1 - 1.5 * iqr
    ust_sinir = q3 + 1.5 * iqr

    sutun_maskesi = (
        (df_2[sutun] < alt_sinir) | (df_2[sutun] > ust_sinir)
    )

    aykiri_deger_maskesi = aykiri_deger_maskesi | sutun_maskesi

    print(f"Aykırı değer sayısı: {sutun_maskesi.sum()}")

    if sutun_maskesi.any():
        print(f"Aykırı değerler: \n{df_2.loc[sutun_maskesi, sutun]}")

# aykırı değer içeren satırları veri setinden çıkartma
df_clean = df_2.loc[~aykiri_deger_maskesi].copy()

# 2.5 Matıksal hatalı değerleri tespit etme ve temizleme

hatali_kayitlar = df_clean[
    (df_clean["yas"] <= 0) |
    (df_clean["gelir"] < 0) |
    (df_clean["abonelik_suresi"] < 0) |
    (df_clean["destek_talebi_sayisi"] < 0) 
]
print(f"Mantılsak hatalı Kayıt sayısı : \n{len(hatali_kayitlar)}")
print(f"Mantılsak hatalı Kayıtlar : \n{hatali_kayitlar}")

df_clean = df_clean[
    (df_clean["yas"] > 0) &
    (df_clean["gelir"] >= 0) &
    (df_clean["abonelik_suresi"] >= 0) &
    (df_clean["destek_talebi_sayisi"] >= 0) 
]

print(f"Mantılsak hatalı Kayıtlar temizlendikten sonra satır kolon sayısı : \n{df_clean.shape}")

# 3. Öznitelik üretme

df_clean["aylik_destek_talebi"] = df_clean["destek_talebi_sayisi"] / (df_clean["abonelik_suresi"] + 1)
df_clean["hayat_boyu_sadakat_orani"] = df_clean["abonelik_suresi"] / df_clean["yas"]

# 4. Hedef değişken ile öznitelikler arasındaki korelasyonları inceleme
sayisal_df = df_clean[["yas", "gelir", "abonelik_suresi", "destek_talebi_sayisi", "aylik_destek_talebi", "hayat_boyu_sadakat_orani","churn"]]
korelasyonlar = sayisal_df.corr(numeric_only=True)["churn"].sort_values(ascending=False)
print(korelasyonlar)

# 5. Mutlak korelasyon değerine göre yüksek olan özniteliklerin seçilmesi (Feature selection)
elenen_ozniteliler = korelasyonlar[abs(korelasyonlar) < 0.1].index.tolist()

df_clean = df_clean.drop(elenen_ozniteliler, axis=1)

# 6. Kategorik değişkenleri One-Hot Encoding ile sayısal forma dönüştürme.( Churn verisi 0-1 olduğu için dönüşüme gerek yoktur )
df_clean.reset_index(drop=True, inplace=True)

y = df_clean["churn"]

X = df_clean.drop(columns=["churn","musteri_id"]).copy()

X = pd.get_dummies(X, columns=["sehir", "uyelik_tipi"], drop_first=True, dtype=int)

# 7. Train-validation-test bölme

X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y) # train_val = %80, test = %20

X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.2, random_state=42, stratify=y_train_val) # train = %64, val = %16, (test = %20)

print(f"X_train: {X_train.shape}")
print(f"X_val: {X_val.shape}")
print(f"X_test: {X_test.shape}")

# 8. Veri Ölçeklendirme
sayisal_sutunlar = X.select_dtypes(include=["int64", "float64"]).columns
standard_scaler = StandardScaler()

X_train_standard = X_train.copy()
X_val_standard = X_val.copy()
X_test_standard = X_test.copy()

X_train_standard[sayisal_sutunlar] = (
    standard_scaler.fit_transform(
        X_train[sayisal_sutunlar]
    )
)

X_val_standard[sayisal_sutunlar] = (
    standard_scaler.transform(
        X_val[sayisal_sutunlar]
    )
)

X_test_standard[sayisal_sutunlar] = (
    standard_scaler.transform(
        X_test[sayisal_sutunlar]
    )
)

print(f"Data Frame eğitim öncesi X_train_standard : \n{X_train_standard.head()}")
print(f"Data Frame eğitim öncesi X_val_standard : \n{X_val_standard.head()}")
print(f"Data Frame eğitim öncesi X_test_standard : \n{X_test_standard.head()}")

# 9. Model eğitimi , Validation ve Hiperparametre ayarmalaması
# 9.1 Model eğitimi , Validation ve Hiperparametre ayarmalaması - Logistic Regression

log_reg_l2 = LogisticRegression(penalty="l2", C=1, max_iter=200)
log_reg_l2.fit(X_train_standard, y_train)
acc_l2 = log_reg_l2.score(X_val_standard, y_val)

log_reg_l1 = LogisticRegression(penalty="l1", solver="liblinear", C=1, max_iter=200)
log_reg_l1.fit(X_train_standard, y_train)
acc_l1 = log_reg_l1.score(X_val_standard, y_val)

if acc_l2 >= acc_l1:
    print("\n ***LogisticRegression için penalty='l2' seçildi.")
    log_reg = LogisticRegression(penalty="l2", C=1, max_iter=200)
    log_reg.fit(X_train_standard, y_train)
else:
    print("\n ***LogisticRegression için penalty='l1' seçildi.")
    log_reg = LogisticRegression(penalty="l1", solver="liblinear", C=1, max_iter=200)
    log_reg.fit(X_train_standard, y_train)

# 9.2 Model eğitimi , Validation ve Hiperparametre ayarmalaması - KNN

best_k = 3
best_val_score = 0

for k in range(3, 15):
    knn_temp = KNeighborsClassifier(n_neighbors=k)
    knn_temp.fit(X_train_standard, y_train)
    score = accuracy_score(y_val, knn_temp.predict(X_val_standard))
    if score > best_val_score:
        best_val_score = score
        best_k = k
print(f"KNN best k : {best_k}")
knn = KNeighborsClassifier(n_neighbors=best_k)
knn.fit(X_train_standard, y_train)


# 9.3 Model eğitimi , Validation ve Hiperparametre ayarmalaması - Decision Tree
best_depth = 1
best_depth_score = 0

for depth in range(1,11):
    tree_clf = DecisionTreeClassifier(criterion="gini", max_depth=depth, random_state=42)
    tree_clf.fit(X_train_standard, y_train)
    
    val_pred = tree_clf.predict(X_val_standard)
    val_score = accuracy_score(y_val, val_pred)
    if val_score > best_depth_score:
        best_depth_score = val_score
        best_depth = depth

print(f"Decision Tree best depth : {best_depth}, score : {best_depth_score}")
tree_clf = DecisionTreeClassifier(criterion="gini", max_depth=best_depth, random_state=42)
tree_clf.fit(X_train_standard, y_train)


# 10. Model Test ve Sınıflandırma metrikleriyle değerlendirme 
# 10.1 Model Test ve Sınıflandırma metrikleriyle değerlendirme - Logistic Regression

y_pred_log_reg = log_reg.predict(X_test_standard)

print("=== Logistic Regression ===")
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_log_reg))
print("\nSınıflandırma Raporu:")
print(classification_report(y_test, y_pred_log_reg))

# 10.2 Model Test ve Sınıflandırma metrikleriyle değerlendirme - KNN
y_test_pred_knn = knn.predict(X_test_standard)

print("=== KNN ===")
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_test_pred_knn))
print("\nSınıflandırma Raporu:")
print(classification_report(y_test, y_test_pred_knn))

# 10.3 Model Test ve Sınıflandırma metrikleriyle değerlendirme - Logistic Regression
tree_y_test_pred = tree_clf.predict(X_test_standard)

print("=== Decision Tree  ===")
print("Confusion Matrix:")
print(confusion_matrix(y_test, tree_y_test_pred))
print("\nSınıflandırma Raporu:")
print(classification_report(y_test, tree_y_test_pred))

# 10.4 Model Test ve Sınıflandırma metrikleriyle değerlendirme - Tüm modellerin Confusion matrix çizimleri
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

ConfusionMatrixDisplay.from_predictions(y_test, y_pred_log_reg, cmap="Blues", values_format="d", ax=axes[0])
axes[0].set_title("Logistic Regression")

ConfusionMatrixDisplay.from_predictions(y_test, y_test_pred_knn, cmap="Blues", values_format="d", ax=axes[1])
axes[1].set_title("KNN")

ConfusionMatrixDisplay.from_predictions(y_test, tree_y_test_pred, cmap="Blues", values_format="d", ax=axes[2])
axes[2].set_title("Decision Tree")

plt.tight_layout()  
plt.show()

# 11. Yorumlama
"""
    1. ML modellerin accuray değerleri;
    - Logistic Regression : 0.63
    - KNN : 0.63
    - Decision Tree : 0.68

    2. ML modellerin (1) precision değerleri;
    - Logistic Regression : 0.58
    - KNN : 0.67
    - Decision Tree : 0.75

    Yorum : Modellerden be

"""