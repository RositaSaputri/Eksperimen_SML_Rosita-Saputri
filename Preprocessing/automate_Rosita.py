import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import joblib
import os
import argparse

# --- Variabel Global Disesuaikan untuk Data Bank ---
RAW_DATA_PATH = "Churn_Modelling.csv" # Nama file input disesuaikan
OUTPUT_FOLDER = "preprocessing"
PROCESSED_DATA_PATH = os.path.join(OUTPUT_FOLDER, "bank_churn_preprocessed.csv") # Nama output disesuaikan
PREPROCESSOR_PATH = os.path.join(OUTPUT_FOLDER, "preprocessor_bank_churn.joblib") # Nama preprocessor disesuaikan

def load_data(path):
    """Memuat data mentah dari file CSV."""
    print(f"Memuat data mentah dari {path}...")
    return pd.read_csv(path)

def clean_data(df):
    """Melakukan pembersihan data awal (Disesuaikan untuk Data Bank)."""
    print("Memulai pembersihan data...")
    df_clean = df.copy()
    
    # 1. Hapus kolom yang tidak relevan (RowNumber, CustomerId, Surname)
    cols_to_drop = ['RowNumber', 'CustomerId', 'Surname']
    df_clean = df_clean.drop(cols_to_drop, axis=1)
    
    # 2. Tangani missing values (Data bank ini bersih, jadi cukup cek)
    # Langkah konversi TotalCharges dihilangkan karena tidak ada di dataset ini dan data bersih
    if df_clean.isnull().sum().sum() > 0:
        print(f"Peringatan: Ada {df_clean.isnull().sum().sum()} missing values. Menghapus baris NaN...")
        df_clean.dropna(inplace=True)
    else:
        print("Data tidak memiliki missing values.")
    
    print(f"Data bersih, jumlah baris: {len(df_clean)}")
    return df_clean

def define_preprocessor(df):
    """Mendefinisikan fitur, target, dan pipeline preprocessor (Disesuaikan untuk Data Bank)."""
    
    df_prep = df.copy()
    
    # Target adalah 'Exited' (Sudah 0/1, tidak perlu mapping)
    X = df_prep.drop('Exited', axis=1)
    y = df_prep['Exited']
    
    # Identifikasi tipe kolom (Disesuaikan untuk fitur bank)
    # Kolom biner (HasCrCard, IsActiveMember) dimasukkan ke numerik untuk di-scaling
    numerical_features = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 
                          'EstimatedSalary', 'HasCrCard', 'IsActiveMember']

    # Kolom kategorikal yang perlu One-Hot Encoding
    categorical_features = ['Geography', 'Gender']
    
    print(f"\nFitur Numerik ({len(numerical_features)}): {numerical_features}")
    print(f"Fitur Kategorikal ({len(categorical_features)}): {categorical_features}")

    # Buat pipeline transformer
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='passthrough'
    )
    
    return preprocessor, X, y

def process_and_save(preprocessor, X, y, output_data_path, output_preprocessor_path):
    """Menjalankan preprocessing dan menyimpan hasilnya."""
    
    print("Menjalankan fit_transform pada data...")
    # 'fit_transform' untuk data pelatihan
    X_processed = preprocessor.fit_transform(X)
    
    # Dapatkan nama fitur baru setelah One-Hot Encoding
    # Kita hanya perlu nama dari transformer 'cat'
    cat_transformer = preprocessor.named_transformers_['cat']
    encoded_cat_features = cat_transformer.get_feature_names_out(preprocessor.transformers_[1][2])

    # Gabungkan nama fitur
    numerical_features = preprocessor.transformers_[0][2]
    new_feature_names = list(numerical_features) + list(encoded_cat_features)
    
    # Buat DataFrame hasil proses
    df_processed = pd.DataFrame(X_processed, columns=new_feature_names)
    
    # Gabungkan kembali dengan target
    df_final = pd.concat([df_processed, y.reset_index(drop=True)], axis=1)
    
    # Simpan data yang sudah diproses
    df_final.to_csv(output_data_path, index=False)
    print(f"Data yang sudah diproses disimpan di: {output_data_path}")
    
    # Simpan preprocessor yang sudah di-fit
    joblib.dump(preprocessor, output_preprocessor_path)
    print(f"Preprocessor yang sudah di-fit disimpan di: {output_preprocessor_path}")

def main():
    """Fungsi utama untuk menjalankan seluruh pipeline preprocessing."""
    
    # Pastikan folder output ada
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    
    # 1. Load
    df_raw = load_data(RAW_DATA_PATH)
    
    # 2. Clean
    df_clean = clean_data(df_raw)
    
    # 3. Define
    preprocessor, X, y = define_preprocessor(df_clean)
    
    # 4. Process & Save
    process_and_save(preprocessor, X, y, PROCESSED_DATA_PATH, PREPROCESSOR_PATH)
    
    print("\nOtomatisasi preprocessing selesai.")

if __name__ == "__main__":
    main()