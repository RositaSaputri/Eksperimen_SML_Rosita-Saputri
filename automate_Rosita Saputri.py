import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# ------------------------------
# 1. Load dataset mentah
# ------------------------------
input_path = "stroke-data.csv"   # pastikan ada di folder yang sama

print("Memuat dataset...")
df = pd.read_csv(input_path)
print(f"Dataset berhasil dimuat dengan {len(df)} baris.\n")

# ------------------------------
# 2. Copy untuk mulai preprocessing
# ------------------------------
df_prep = df.copy()

# ------------------------------
# 3. Hapus duplikasi
# ------------------------------
before = len(df_prep)
df_prep = df_prep.drop_duplicates()
after = len(df_prep)

print(f"Duplikasi dihapus: {before - after} baris")

# ------------------------------
# 4. Tangani missing values
# ------------------------------
if df_prep['bmi'].isna().sum() > 0:
    df_prep['bmi'] = df_prep['bmi'].fillna(df_prep['bmi'].median())

print(f"Missing value tersisa: {df_prep.isna().sum().sum()}\n")

# ------------------------------
# 5. Pisahkan fitur dan target
# ------------------------------
y = df_prep['stroke']
X = df_prep.drop('stroke', axis=1)

# ------------------------------
# 6. Tentukan kolom numerik & kategorikal
# ------------------------------
numeric_features = ['age', 'avg_glucose_level', 'bmi']
categorical_features = [col for col in X.columns if col not in numeric_features]

print("Fitur numerik :", numeric_features)
print("Fitur kategorikal :", categorical_features)

# ------------------------------
# 7. Buat pipeline preprocessing
# ------------------------------
numeric_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ],
    remainder='drop'
)

# ------------------------------
# 8. Jalankan preprocessing
# ------------------------------
print("\nMenjalankan proses encoding & scaling...")
X_processed = preprocessor.fit_transform(X)

# ------------------------------
# 9. Buat dataframe hasil transformasi
# ------------------------------
encoded_cat_features = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features)
new_feature_names = numeric_features + list(encoded_cat_features)

df_processed = pd.DataFrame(X_processed, columns=new_feature_names)

# gabungkan dengan target
df_final = pd.concat([df_processed, y.reset_index(drop=True)], axis=1)

# ------------------------------
# 10. Simpan hasil preprocessing
# ------------------------------
output_path = "stroke_preprocessed.csv"
df_final.to_csv(output_path, index=False)

print(f"\nPreprocessing selesai!")
print(f"File disimpan sebagai: {output_path}")
print("\nContoh data setelah preprocessing:")
print(df_final.head())
