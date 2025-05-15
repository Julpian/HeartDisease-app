import streamlit as st
import pandas as pd
import joblib

# Konfigurasi halaman
st.set_page_config(page_title="Prediksi Penyakit Jantung", layout="wide")

# Mengatur tema warna menggunakan CSS
custom_theme = """
<style>
/* Mengatur warna latar belakang utama */
.stApp {
    background-color: #b1e6d0;
}

/* Mengatur warna latar belakang sekunder (seperti sidebar) */
.css-1v3fvcr, .css-17eq0hr {
    background-color: #b1e6d0 !important;
}

/* Mengatur warna teks utama */
.stApp, .stMarkdown, .stTextInput, .stSelectbox, .stNumberInput, .stButton button, .css-1v3fvcr, .css-17eq0hr {
    color: #000000 !important;
}

/* Memastikan teks pada elemen lain juga hitam */
p, h1, h2, h3, h4, h5, h6, label, div {
    color: #000000 !important;
}
</style>
"""
st.markdown(custom_theme, unsafe_allow_html=True)

# Sidebar Navigasi
st.sidebar.image("https://cdn-icons-png.flaticon.com/128/17858/17858467.png", width=100)
st.sidebar.title("📌 Navigasi")
menu = st.sidebar.radio("Pilih Halaman", ["🏠 Home", "🔍 Prediksi", "ℹ️ Tentang"])

# Load model prediksi
try:
    model = joblib.load("best_random_forest_model.pkl")
    model_loaded = True
except Exception as e:
    model_loaded = False
    model_error = e

# Mapping variabel kategorikal
categorical_mappings = {
    'Sex': {'Laki-laki': 1, 'Perempuan': 0},
    'ChestPainType': {'ATA': 0, 'NAP': 1, 'ASY': 2, 'TA': 3},
    'RestingECG': {'Normal': 0, 'ST': 1, 'LVH': 2},
    'ExerciseAngina': {'Tidak': 0, 'Ya': 1},
    'ST_Slope': {'Naik': 0, 'Datar': 1, 'Turun': 2}
}

# Halaman Home
if menu == "🏠 Home":
    st.title("❤️ Aplikasi Prediksi Penyakit Jantung")
    st.markdown("""
    Aplikasi ini menggunakan model pembelajaran mesin untuk membantu memprediksi risiko seseorang terkena penyakit jantung berdasarkan data medis umum.

    > Cocok untuk edukasi, demonstrasi, dan eksplorasi data sains di bidang kesehatan.

    Silakan lanjut ke tab **Prediksi** untuk mencoba.
    """)
    st.image("https://i.pinimg.com/736x/d2/61/d4/d261d480026ca1bb86fd96390014463e.jpg", use_column_width=True)

# Halaman Prediksi
elif menu == "🔍 Prediksi":
    st.title("🔍 Prediksi Risiko Penyakit Jantung")

    if not model_loaded:
        st.error("❌ Model tidak berhasil dimuat. Pastikan file `best_random_forest_model.pkl` tersedia di direktori aplikasi.")
        st.exception(model_error)
    else:
        st.markdown("Silakan isi informasi berikut:")

        st.header("🩺 Data Pasien")
        col1, col2 = st.columns(2)

        with col1:
            age = st.number_input("Usia", 0, 120, 40)
            resting_bp = st.number_input("Tekanan Darah Istirahat (mm Hg)", 0, 300, 120)
            cholesterol = st.number_input("Kolesterol (mg/dl)", 0, 600, 200)
            max_hr = st.number_input("Detak Jantung Maksimum", 0, 300, 150)

        with col2:
            sex = st.selectbox("Jenis Kelamin", ["Laki-laki", "Perempuan"])
            chest_pain_type = st.selectbox("Tipe Nyeri Dada", ["ATA", "NAP", "ASY", "TA"])
            resting_ecg = st.selectbox("Hasil EKG Saat Istirahat", ["Normal", "ST", "LVH"])
            exercise_angina = st.selectbox("Nyeri Dada Saat Latihan?", ["Tidak", "Ya"])
            fasting_bs = st.selectbox("Gula Darah Saat Puasa > 120 mg/dl?", [0, 1], format_func=lambda x: "Ya" if x == 1 else "Tidak")
            oldpeak = st.number_input("Oldpeak (Depresi ST)", -5.0, 10.0, 0.0, step=0.1)
            st_slope = st.selectbox("Kemiringan ST", ["Naik", "Datar", "Turun"])

        if st.button("🧠 Jalankan Prediksi"):
            input_data = pd.DataFrame({
                'Age': [age],
                'Sex': [sex],
                'ChestPainType': [chest_pain_type],
                'RestingBP': [resting_bp],
                'Cholesterol': [cholesterol],
                'FastingBS': [fasting_bs],
                'RestingECG': [resting_ecg],
                'MaxHR': [max_hr],
                'ExerciseAngina': [exercise_angina],
                'Oldpeak': [oldpeak],
                'ST_Slope': [st_slope]
            })

            for col in ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope']:
                input_data[col] = input_data[col].map(categorical_mappings[col])

            prediction = model.predict(input_data)[0]
            probability = model.predict_proba(input_data)[0][1]

            st.subheader("📊 Hasil Prediksi")
            if prediction == 1:
                st.error(f"⚠️ Pasien **berisiko tinggi** terkena penyakit jantung. (Probabilitas: {probability:.2%})")
            else:
                st.success(f"✅ Pasien **kemungkinan besar tidak** terkena penyakit jantung. (Probabilitas: {probability:.2%})")

# Halaman Tentang
elif menu == "ℹ️ Tentang":
    st.title("ℹ️ Tentang Aplikasi")
    st.markdown("""
    Aplikasi ini dikembangkan untuk edukasi dan eksplorasi dalam bidang *Data Science* khususnya prediksi penyakit jantung.

    **Teknologi yang digunakan:**
    - Python
    - Streamlit
    - Scikit-learn
    - Pandas

    **Pengembang:**  
    👨‍💻 Lutfi Julpian  
    💡 Mahasiswa & Praktisi Data Pemula

    ---
    ⚠️ *Catatan: Aplikasi ini tidak dimintended untuk diagnosis medis. Gunakan hanya untuk keperluan pembelajaran.*
    """)

    st.image("https://cdn-icons-png.flaticon.com/512/2965/2965567.png", width=100)