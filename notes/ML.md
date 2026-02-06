# 🤖 Python Machine Learning Tutorial (Data Science) oleh <br>[Programming with Mosh](https://youtu.be/7eh4d6sabA0?si=Ip7zM44rVUyPozQ-)

## ![ML Tutorial](https://i.ytimg.com/vi/7eh4d6sabA0/hq720.jpg?sqp=-oaymwEhCK4FEIIDSFryq4qpAxMIARUAAAAAGAElAADIQj0AgKJD&rs=AOn4CLBEQOvQCwmBsz_SweM6v_p_Z6OAUg "Machine Learning Tutorial")

## 1) 📌 Apa itu Machine Learning?

Machine Learning (ML) adalah cabang dari artificial intelligence (AI) yang memungkinkan komputer belajar dari data tanpa diprogram secara eksplisit dengan aturan tetap. Model ML menemukan pola dalam data dan kemudian menggunakannya untuk meng-prediksi data baru.

**Contoh masalah ML:**

- Klasifikasi gambar (kucing vs anjing)

- Prediksi harga rumah

- Rekomendasi musik berdasarkan preferensi pengguna

- Segmentasi pelanggan

---

## 2) 🚀 Langkah-langkah Umum pada Proyek Machine Learning

1. **Ambil/Import Data →** biasanya dari file seperti `CSV`.

2. **Bersihkan & Pre-proses →** hilangkan duplikat, perlakukan nilai kosong, konversi kategori jadi angka.

3. **Pisahkan Data →** menjadi training dan testing set.

4. **Pilih Model →** misalnya Decision Tree, Logistic Regression, Random Forest, dll.

5. **Training (latih) Model →** biarkan model belajar dari training set.

6. **Evaluasi Model →** nilai performa dengan testing set.

7. **Prediksi →** gunakan model yang sudah dilatih untuk buat prediksi baru.

---

## 3) 🛠️ Alat & Library yang Digunakan

- **Pandas:** Untuk analisis data dan manipulasi Data Frame.

- **NumPy:** Untuk pengiraan tatasusunan (array) pelbagai dimensi.

- **Scikit-Learn (sklearn):** Library utama yang menyediakan pelbagai algoritma machine learning.

- **Jupyter Notebook:** Persekitaran pembangunan yang memudahkan visualisasi data.

---

## 4) ⚙️ Instalasi & Setup Lingkungan

**4.1 📖 install library**

```bash
pip install numpy pandas scikit-learn matplotlib seaborn
```

---

**4.2 📓 Install Jupyter Notebook**
Jupyter Notebook digunakan untuk eksperimen, eksplorasi data (EDA), dan visualisasi Machine Learning secara interaktif.

**🐍 Install via pip**
menggunakan jupyter notebook:

```bash
pip install notebook ipykernel
```

atau menggunakan jupyter lab:

```bash
pip install jupyterlab ipykernel
```

---

## 5) 🖥️ Import Library – Dasar Project

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
```

---

## 6) 📊 Load Data

Misalnya kita punya dataset sederhana:

```python
df = pd.read_csv('music_preferences.csv')
print(df.head())
```

Contoh struktur dataset:

| Age | Gender | Genre  |
| --- | ------ | ------ |
| 23  | 1      | hiphop |
| 29  | 0      | jazz   |

> **Catatan:** Untuk dataset kalian bisa cari di website **kaggle**. Di **kaggle** ada banyak dataset yang bisa kalian coba untuk eksperimen ataupun membuat project nya dan **kaggle** selalu update lewat komunitasnya yang aktif.

---

## 7) 📉 Pre-processing Data

**7.1 Handle Missing Values**

```python
df = df.dropna()  # buang baris yang nilai kosong
```

**7.2 Encode Label**
Kalau kolom target berupa teks:

```python
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['genre'] = le.fit_transform(df['genre'])
```

**7.3 Split Train/Test**

```python
X = df[['age', 'gender']]
y = df['genre']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

---

## 8) ⚖️ Scaling / Normalisasi (Opsional)

Sering penting ketika nilai fitur memiliki rentang berbeda:

```python
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

---

## 9) 🌳 Pilih Model — Contoh: Decision Tree

```python
from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier()
model.fit(X_train_scaled, y_train)
```

---

## 10) 🧪 Evaluasi Model

```python
y_pred = model.predict(X_test_scaled)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
```

📍 **Accuracy** memberi tahu seberapa sering model prediksi benar.

---

## 11) 📈 Visualisasi Model (Opsional)

```python
from sklearn import tree

plt.figure(figsize=(12,8))
tree.plot_tree(model, feature_names=X.columns, class_names=le.classes_, filled=True)
plt.show()
```

Ini akan menunjukkan diagram _decision tree_ secara grafis.

---

## 12) 💾 Simpan & 🔃 Load Model (Pickle)

**Simpan model**

```python
import joblib
joblib.dump(model, 'model_ml.pkl')
```

**Load model**

```python
model_loaded = joblib.load('model_ml.pkl')
prediksi = model_loaded.predict(scaler.transform([[30, 1]]))
print(prediksi)
```

---

### 📌 Contoh Kasus Penerapan Sederhana: Prediksi Genre Musik

Kita asumsikan dataset:
| Age | Gender | Genre |
| --- | ------ | ----- |
| 22 | 1 | hihop |
| 29 | 0 | acoustic |
| ... | ... | ... |

Kode lengkapnya:

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Read CSV
df = pd.read_csv('music_preferences.csv')

# Encode target
le = LabelEncoder()
df['genre'] = le.fit_transform(df['genre'])

X = df[['age', 'gender']]
y = df['genre']

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
model = DecisionTreeClassifier()
model.fit(X_train_scaled, y_train)

# Evaluate
y_pred = model.predict(X_test_scaled)
print("Accuracy:", accuracy_score(y_test, y_pred))
```

---

### 🧠 Analisis Kritis

Asumsi yang sering muncul di video tutorial:

- Dataset bersih dan telah lengkap – padahal data nyata sering berantakan.

- Performa model cukup tinggi – padahal metrik lain seperti confusion matrix atau cross validation perlu dilihat.

- Hanya satu algoritma – padahal bisa dibandingkan beberapa model.

➡️ _Benchmarking_ model lain (Random Forest, SVM) sering memberi insight lebih baik.
