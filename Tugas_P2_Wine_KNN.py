import pandas as pd
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 1. Memuat dataset Wine dari UCI Machine Learning Repository
wine = fetch_ucirepo(id=109)

X = wine.data.features.copy()
y = wine.data.targets.iloc[:, 0].copy()

# Gabungkan fitur dan target agar pemeriksaan awal mudah dibaca
df = X.copy()
df["class"] = y

print("=" * 60)
print("1. INFORMASI DATASET")
print("=" * 60)
print("Ukuran dataset (baris, kolom):", df.shape)
print("Jumlah calon fitur:", X.shape[1])
print("Nama kolom:", df.columns.tolist())
print("\nTipe data setiap kolom:")
print(df.dtypes)
print("\nRingkasan info dataset:")
df.info()

# 2. Pemeriksaan missing values
print("\n" + "=" * 60)
print("2. PEMERIKSAAN MISSING VALUES")
print("=" * 60)
missing_per_column = df.isnull().sum()
print(missing_per_column)
print("Total missing values:", int(missing_per_column.sum()))

# Dataset Wine dari UCI umumnya tidak memiliki missing values.
# Jika program menemukan nilai kosong, hentikan program agar nilai tersebut
# tidak ditangani secara diam-diam tanpa keputusan prapemrosesan yang jelas.
if missing_per_column.sum() > 0:
    raise ValueError(
        "Ditemukan missing values. Tentukan metode penanganannya "
        "terlebih dahulu sebelum melanjutkan."
    )

# 3. Pemeriksaan duplikasi
print("\n" + "=" * 60)
print("3. PEMERIKSAAN DUPLIKASI")
print("=" * 60)
duplicate_count = int(df.duplicated().sum())
print("Jumlah baris duplikat:", duplicate_count)

# Tidak menghapus baris secara otomatis. Jika ada duplikat, jumlahnya
# dilaporkan terlebih dahulu agar keputusan penghapusan dapat dijelaskan.
if duplicate_count > 0:
    print("Catatan: baris duplikat ditemukan; tinjau sebelum memutuskan penghapusan.")
else:
    print("Tidak ditemukan baris duplikat.")

# 4. Pemeriksaan konsistensi kategori/target
print("\n" + "=" * 60)
print("4. PEMERIKSAAN TARGET (CLASS)")
print("=" * 60)
print("Nilai unik pada target:", sorted(df["class"].dropna().unique()))
print("Jumlah nilai target yang kosong:", int(df["class"].isnull().sum()))
print(
    "Target Wine berupa label kelas, bukan fitur numerik untuk distandarisasi."
)

# 5. Pembagian data dan standarisasi fitur
# Target tidak ikut distandarisasi. Scaler hanya fit pada data latih,
# kemudian parameter yang sama diterapkan pada data uji.
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\n" + "=" * 60)
print("5. STANDARISASI")
print("=" * 60)
print("Jumlah data latih:", X_train.shape[0])
print("Jumlah data uji:", X_test.shape[0])
print("Jumlah fitur yang distandarisasi:", X.shape[1])
print("Metode: StandardScaler (z-score)")
print("Scaler di-fit hanya pada data latih, lalu diterapkan pada data uji.")

# Buat tabel hasil prapemrosesan seluruh data untuk memenuhi permintaan
# tampilan head dan tail. Nilai transformasi menggunakan scaler yang hanya
# dipelajari dari data latih; kolom target tetap seperti aslinya.
X_all_scaled = scaler.transform(X)
processed_df = pd.DataFrame(
    X_all_scaled,
    columns=X.columns,
    index=X.index
)
processed_df["class"] = y.to_numpy()

print("\n=== 5 DATA AWAL SEBELUM PRAPEMROSESAN ===")
print(df.head(5))
print("\n=== 5 DATA AKHIR SEBELUM PRAPEMROSESAN ===")
print(df.tail(5))

print("\n" + "=" * 60)
print("6. LIMA BARIS TERATAS HASIL PRAPEMROSESAN")
print("=" * 60)
print(processed_df.head(5).round(4).to_string())

print("\n" + "=" * 60)
print("7. LIMA BARIS TERBAWAH HASIL PRAPEMROSESAN")
print("=" * 60)
print(processed_df.tail(5).round(4).to_string())

# 8. PELATIHAN DAN PENGUJIAN MODEL KNN
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

print("\n" + "=" * 60)
print("8. PENGUJIAN MODEL KNN")
print("=" * 60)

# Membuat model KNN dengan K = 13
knn = KNeighborsClassifier(n_neighbors=13)

# Melatih model menggunakan data latih yang sudah distandarisasi
knn.fit(X_train_scaled, y_train)

# Melakukan prediksi terhadap data uji
y_pred = knn.predict(X_test_scaled)

# Menghitung akurasi
akurasi = accuracy_score(y_test, y_pred)

print("=== HASIL PENGUJIAN KNN ===")
print("Nilai K:", knn.n_neighbors)
print("Akurasi data uji:", round(akurasi, 4))
print("Akurasi data uji (%):", round(akurasi * 100, 2), "%")

# 9. CONFUSION MATRIX
print("\n" + "=" * 60)
print("9. CONFUSION MATRIX")
print("=" * 60)

cm = confusion_matrix(y_test, y_pred)
print("=== CONFUSION MATRIX ===")
print(cm)

# 10. CLASSIFICATION REPORT
print("\n" + "=" * 60)
print("10. CLASSIFICATION REPORT")
print("=" * 60)

print("=== CLASSIFICATION REPORT ===")
print(
    classification_report(
        y_test,
        y_pred,
        zero_division=0
    )
)

print("\nPengujian model KNN selesai.")
