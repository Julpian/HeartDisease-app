import streamlit as st
import pandas as pd
import joblib
from streamlit_option_menu import option_menu
from sklearn.preprocessing import StandardScaler

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

# Sidebar Navigasi dengan streamlit-option-menu
with st.sidebar:
    st.image("https://img.icons8.com/?size=100&id=35583&format=png", width=100)
    st.title("📌 Navigasi")

    menu = option_menu(
        menu_title=None,
        options=["🏠 Home", "🔍 Prediksi", "ℹ️ About"],
        icons=["house", "search", "info-circle"],
        menu_icon="cast",
        default_index=0,
        orientation="vertical",
        styles={
            "container": {"padding": "5px", "background-color": "#b1e6d0"},
            "icon": {"color": "#000000", "font-size": "20px"},
            "nav-link": {
                "font-size": "18px",
                "text-align": "left",
                "margin": "0px",
                "--hover-color": "#4CAF50",
                "color": "#000000",
            },
            "nav-link-selected": {
                "background-color": "#4CAF50",
                "color": "white",
                "font-weight": "bold",
            },
        }
    )

# Load model dan scaler
try:
    model = joblib.load("best_random_forest_model.pkl")
    scaler = joblib.load("scaler.pkl")  # Pastikan file scaler.pkl tersedia
    model_loaded = True
except Exception as e:
    model_loaded = False
    model_error = e

# Mapping variabel kategorikal
categorical_mappings = {
    'Sex': {'M': 1, 'F': 0},
    'ChestPainType': {'ATA': 0, 'NAP': 1, 'ASY': 2, 'TA': 3},
    'RestingECG': {'LVH': 0, 'Normal': 1, 'ST': 2},
    'ExerciseAngina': {'N': 0, 'Y': 1},
    'ST_Slope': {'Down': 0, 'Flat': 1, 'Up': 2}
}

# Halaman Home
if menu == "🏠 Home":
    st.title("❤️ Aplikasi Prediksi Penyakit Jantung")

    # Tabel Standar Input
    st.subheader("📥 Standar Input Fitur")
    st.markdown("""
    Berikut adalah rentang nilai dan jenis data yang digunakan sebagai input untuk model prediksi:

    | Fitur           | Nilai Minimum     | Nilai Maksimum     | Keterangan |
    |------------------|-------------------|---------------------|------------|
    | **Age**          | 28 tahun          | 77 tahun            | Rentang usia pasien berdasarkan statistik deskriptif (sub-bab 4.1). |
    | **Sex**          | F (Perempuan)     | M (Laki-laki)       | Kategorikal, hanya dua nilai yang valid berdasarkan dataset. |
    | **ChestPainType**| ASY               | TA                  | Kategorikal: ASY (Asimptomatik), ATA (Angina Atipikal), NAP (Non-Angina), TA (Angina Tipikal). |
    | **RestingBP**    | 80 mm Hg          | 200 mm Hg           | Rentang tekanan darah saat istirahat, nilai 0 dianggap anomali (sub-bab 4.1). |
    | **Cholesterol**  | 100 mg/dl         | 603 mg/dl           | Kadar kolesterol serum, nilai 0 tidak realistis secara klinis (sub-bab 4.1). |
    | **FastingBS**    | 0 (≤120 mg/dl)    | 1 (>120 mg/dl)      | Kategorikal biner, berdasarkan ambang batas gula darah puasa. |
    | **RestingECG**   | LVH               | ST                  | Kategorikal: Normal, ST (kelainan ST-T), LVH (hipertrofi ventrikel kiri). |
    | **MaxHR**        | 60                | 202                 | Denyut jantung maksimum, nilai >202 dianggap anomali (sub-bab 4.1). |
    | **ExerciseAngina**| N (Tidak)        | Y (Ya)              | Kategorikal biner, menunjukkan ada/tidaknya angina akibat olahraga. |
    | **Oldpeak**      | -2.6              | 6.2                 | Nilai depresi segmen ST, rentang berdasarkan dataset. |
    | **ST_Slope**     | Down              | Up                  | Kategorikal: Up (Naik), Flat (Datar), Down (Menurun). |
    """, unsafe_allow_html=True)

    # Informasi Aplikasi
    st.markdown("""
    ---
    Aplikasi ini dikembangkan sebagai bagian dari **pembelajaran dan penelitian** dalam bidang data sains dan kesehatan, 
    khususnya untuk memprediksi risiko penyakit jantung menggunakan model pembelajaran mesin.

    ⚠️ **Disclaimer:**  
    Aplikasi ini **belum layak** untuk digunakan sebagai alat diagnosis medis atau penentu kondisi kesehatan seseorang. 
    Semua hasil prediksi hanya bersifat edukatif dan **tidak menggantikan** saran atau diagnosis dari tenaga medis profesional.

    #### 📊 Tentang Dataset:
    Dataset yang digunakan dalam aplikasi ini diambil dari platform [Kaggle](https://www.kaggle.com/), 
    dan tersedia secara terbuka di bawah lisensi **Open Database License (OdbL)**. 
    Artinya, penggunaannya dalam penelitian ini **sah dan legal**, tanpa memerlukan izin tambahan.  
    Dataset ini dianggap layak digunakan sebagai bahan penelitian akademik maupun eksplorasi data sains.

    > Silakan lanjut ke tab **Prediksi** untuk mencoba model prediksi penyakit jantung berdasarkan input data medis Anda.
    """)

    # Deskripsi Fitur
    st.markdown("""
    #### 🧬 Deskripsi Fitur Dataset:

    | Nama Fitur       | Deskripsi |
    |------------------|-----------|
    | **Age**          | Usia pasien (dalam tahun) |
    | **Sex**          | Jenis kelamin pasien (`M` = Laki-laki, `F` = Perempuan) |
    | **ChestPainType**| Jenis nyeri dada: <br> • `TA`: Angina tipikal <br> • `ATA`: Angina atipikal <br> • `NAP`: Nyeri non-angina <br> • `ASY`: Asimptomatik (tanpa gejala) |
    | **RestingBP**    | Tekanan darah saat istirahat (mm Hg) |
    | **Cholesterol**  | Kadar kolesterol serum (mg/dl) |
    | **FastingBS**    | Gula darah puasa (`1` jika > 120 mg/dl, `0` jika ≤ 120 mg/dl) |
    | **RestingECG**   | Hasil elektrokardiogram saat istirahat: <br> • `Normal` <br> • `ST`: Kelainan gelombang ST-T <br> • `LVH`: Hipertrofi ventrikel kiri |
    | **MaxHR**        | Denyut jantung maksimum saat uji beban (60–202) |
    | **ExerciseAngina**| Angina akibat olahraga (`Y` = Ya, `N` = Tidak) |
    | **Oldpeak**      | Nilai depresi segmen ST setelah latihan fisik |
    | **ST_Slope**     | Kemiringan segmen ST saat puncak latihan: <br> • `Up`: Menaik <br> • `Flat`: Datar <br> • `Down`: Menurun |
    | **HeartDisease** | Target/output: `1` = Mengidap penyakit jantung, `0` = Normal |
    """, unsafe_allow_html=True)

# Halaman Prediksi
elif menu == "🔍 Prediksi":
    st.title("🔍 Prediksi Risiko Penyakit Jantung")

    # Tambahkan style untuk tombol prediksi
    st.markdown("""
        <style>
            div.stButton > button:first-child {
                background-color: #4CAF50;
                color: white;
                padding: 0.75em 2em;
                font-size: 1.1em;
                border-radius: 8px;
                transition: 0.3s;
                box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
            }
            div.stButton > button:hover {
                background-color: #45a049;
                transform: scale(1.02);
            }
        </style>
    """, unsafe_allow_html=True)

    if not model_loaded:
        st.error("❌ Model tidak berhasil dimuat. Pastikan file `best_random_forest_model.pkl` dan `scaler.pkl` tersedia di direktori aplikasi.")
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
            sex = st.selectbox("Jenis Kelamin", ["F", "M"])
            chest_pain_type = st.selectbox("Tipe Nyeri Dada", ["ATA", "NAP", "ASY", "TA"])
            resting_ecg = st.selectbox("Hasil EKG Saat Istirahat", ["Normal", "ST", "LVH"])
            exercise_angina = st.selectbox("Nyeri Dada Saat Latihan?", ["N", "Y"])
            fasting_bs = st.selectbox(
                "Gula Darah Saat Puasa > 120 mg/dl?",
                [0, 1],
                format_func=lambda x: "1" if x == 1 else "0"
            )
            oldpeak = st.number_input("Oldpeak (Depresi ST)", -5.0, 10.0, 0.0, step=0.1)
            st_slope = st.selectbox("Kemiringan ST", ["Up", "Flat", "Down"])

        if st.button("🧠 Jalankan Prediksi"):
            with st.spinner("🔄 Menjalankan model prediksi..."):
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

                # Petakan fitur kategorikal
                for col in ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope']:
                    input_data[col] = input_data[col].map(categorical_mappings[col])

                # Pastikan urutan kolom sama dengan saat pelatihan
                input_data = input_data[['Age', 'Sex', 'ChestPainType', 'RestingBP', 'Cholesterol', 
                                         'FastingBS', 'RestingECG', 'MaxHR', 'ExerciseAngina', 
                                         'Oldpeak', 'ST_Slope']]

                # Scaling fitur numerik
                numeric_features = ['Age', 'RestingBP', 'FastingBS', 'Cholesterol', 'MaxHR', 'Oldpeak']
                input_data[numeric_features] = scaler.transform(input_data[numeric_features])

                # Prediksi
                prediction = model.predict(input_data)[0]
                probability_class_1 = model.predict_proba(input_data)[0][1]

            # Hasil prediksi berdasarkan probabilitas
            st.subheader("📊 Hasil Prediksi")
            if probability_class_1 < 0.25:
                st.success(f"✅ Pasien **tidak berisiko** terkena penyakit jantung.\n\n**Probabilitas Penyakit Jantung:** {probability_class_1:.2%}")
            elif 0.25 <= probability_class_1 <= 0.50:
                st.warning(f"⚠️ Pasien **kemungkinan memiliki** penyakit jantung.\n\n**Probabilitas Penyakit Jantung:** {probability_class_1:.2%}")
            else:  # probability_class_1 > 0.50
                st.error(f"🚨 Pasien **berisiko tinggi** terkena penyakit jantung.\n\n**Probabilitas Penyakit Jantung:** {probability_class_1:.2%}")
                st.markdown("Silakan konsultasikan hasil ini dengan dokter atau tenaga medis profesional untuk penanganan lebih lanjut.")

# Halaman Tentang
elif menu == "ℹ️ About":
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
    """)

    # Menampilkan foto pengembang
    st.image("gambar/lutfi.jpg", width=150, caption="Lutfi Julpian")

    st.markdown("""
    ---
    ⚠️ *Catatan: Aplikasi ini tidak dimaksudkan untuk diagnosis medis. Gunakan hanya untuk keperluan pembelajaran.*
    """)

    # Ikon ilustrasi tambahan
    st.image("https://cdn-icons-png.flaticon.com/512/2965/2965567.png", width=100)