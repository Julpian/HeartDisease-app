# app.py

import streamlit as st
import pandas as pd
import joblib
from streamlit_option_menu import option_menu
import time
import lime
import lime.lime_tabular
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import plotly.graph_objects as go
import plotly.express as px

# ==============================================================================
# 1. KONFIGURASI, DATA STATIS, DAN GAYA (CSS)
# ==============================================================================

# Konfigurasi halaman utama Streamlit
st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ========================================================================================
# [PERBAIKAN] - Memasukkan kembali semua kunci yang diperlukan
# ========================================================================================
TRANSLATIONS = {
    'id': {
        'banner_title': "Sistem Prediksi Penyakit Jantung Canggih",
        'banner_subtitle': "Didukung oleh Teknologi Machine Learning & AI",
        'nav_home': "🏠 Home", 'nav_predict': "🔍 Prediksi", 'nav_about': "ℹ️ Tentang",
        'nav_title': "📌 Navigasi", 'lang_select_title': "Pilih Bahasa",
        'model_info_title': "🎯 Info Model",
        'model_info_algo': "Algoritma: Random Forest",
        'model_info_recall': "Recall: 92%",
        'model_info_accuracy': "Akurasi: 90%",
        'model_info_features': "Fitur: 11 Indikator Medis",
        'home_intro_new': "Manfaatkan kekuatan kecerdasan buatan untuk menilai risiko penyakit jantung dengan presisi dan kepercayaan diri.",
        'dataset_overview_title': "📊 Tinjauan Dataset Real-time",
        'total_samples_title': "Total Sampel", 'total_samples_desc': "Rekam Pasien",
        'features_title': "Fitur", 'features_desc': "Indikator Kesehatan",
        'positive_cases_title': "Kasus Sakit Jantung", 'positive_cases_desc': "Kasus Positif",
        'healthy_cases_title': "Kasus Sehat", 'healthy_cases_desc': "Kasus Negatif",
        'input_guide_title': "📘 Panduan Input Fitur",
        'age_title': "🎂 Usia (Age)", 'age_range': "Rentang data pada model: **28 - 77 tahun**.",
        'resting_bp_title': "🩸 Tekanan Darah (RestingBP)", 'resting_bp_range': "Rentang data: **0 - 200 mm Hg**.",
        'cholesterol_title': "🧪 Kolesterol (Cholesterol)", 'cholesterol_range': "Rentang data: **0 - 603 mg/dl**.",
        'cholesterol_note': "💡 **Info**: Nilai 0 menandakan data hilang. Nilai normal klinis umumnya **< 200 mg/dl**.",
        'max_hr_title': "❤️‍🔥 Detak Jantung Maks. (MaxHR)", 'max_hr_range': "Rentang data: **60 - 202 bpm**.",
        'oldpeak_title': "📈 Oldpeak", 'oldpeak_range': "Rentang data: **-2.6 - 6.2**.",
        'categorical_title': "🗂️ Fitur Pilihan",
        'categorical_desc': "Untuk fitur seperti *Jenis Kelamin* atau *Tipe Nyeri Dada*, Anda hanya perlu memilih dari opsi yang tersedia.",
        'home_disclaimer_title': "⚠️ Disclaimer Penting",
        'home_disclaimer_md': "Aplikasi ini adalah prototipe untuk tujuan edukasi dan **tidak boleh digunakan untuk diagnosis medis nyata**. Hasil prediksi tidak menggantikan konsultasi dengan tenaga medis profesional.",
        'form_intro': "Silakan isi data pasien di bawah ini secara akurat untuk mendapatkan prediksi.",
        'patient_data_header': "🩺 Data Pasien",
        'age_label': "Usia", 'sex_label': "Jenis Kelamin", 'chest_pain_type_label': "Tipe Nyeri Dada",
        'resting_bp_label': "Tekanan Darah Istirahat", 'cholesterol_label': "Kolesterol",
        'fasting_bs_label': "Gula Darah Puasa > 120 mg/dl?", 'resting_ecg_label': "Hasil EKG Istirahat",
        'max_hr_label': "Detak Jantung Maksimum", 'exercise_angina_label': "Angina Akibat Olahraga?",
        'oldpeak_label': "Oldpeak", 'st_slope_label': "Kemiringan Puncak Latihan ST",
        'predict_button': "🧠 Jalankan Prediksi & Analisis", 'spinner_text': "🔄 Menganalisis data...",
        'result_header': "📊 Hasil Analisis AI",
        'risk_score': "Skor Risiko", 'risk_level': "Tingkat Risiko", 'recommendation': "Rekomendasi",
        'result_low_risk': "RISIKO RENDAH", 'result_medium_risk': "RISIKO SEDANG", 'result_high_risk': "RISIKO TINGGI",
        'recommendation_low': "Lanjutkan gaya hidup sehat.",
        'recommendation_medium': "Disarankan untuk konsultasi dengan dokter untuk pemantauan.",
        'recommendation_high': "SEGERA konsultasi dengan dokter spesialis jantung.",
        'probability_breakdown': "🔗 Rincian Probabilitas",
        'risk_distribution': "Distribusi Probabilitas Risiko",
        'no_disease': "Tidak Sakit Jantung", 'disease': "Sakit Jantung",
        'model_confidence_title': "🎯 Keyakinan Model",
        'prediction_confidence_label': "Keyakinan Prediksi",
        'no_disease_prob': "Probabilitas Tidak Sakit Jantung", 'disease_prob': "Probabilitas Sakit Jantung",
        'lime_analysis_title': "🔬 Penjelasan Faktor Risiko (Analisis LIME)",
        'lime_explanation': "Grafik di bawah ini menunjukkan faktor-faktor yang paling berpengaruh pada prediksi untuk pasien ini. Faktor dengan bar merah mendukung prediksi 'Sakit Jantung', sedangkan bar hijau menentangnya.",
        'lime_plot_title_prefix': "Penjelasan lokal untuk kelas", 'class_disease': "Penyakit",
        
        # KUNCI YANG DIPERBAIKI DAN DITAMBAHKAN KEMBALI
        'prob_explanation_title': "💡 Interpretasi Probabilitas",
        'prob_explanation_high': "Skor probabilitas pasien **({score:.1f}%)** berada di atas ambang batas risiko tinggi **(46%)**, sehingga diklasifikasikan sebagai **RISIKO TINGGI**.",
        'prob_explanation_medium': "Skor probabilitas pasien **({score:.1f}%)** berada di antara ambang batas risiko sedang **(25% - 45.9%)**, sehingga diklasifikasikan sebagai **RISIKO SEDANG**.",
        'prob_explanation_low': "Skor probabilitas pasien **({score:.1f}%)** berada di bawah ambang batas risiko sedang **(25%)**, sehingga diklasifikasikan sebagai **RISIKO RENDAH**.",
        
        'about_what_is_it_title': "🎯 Apa itu aplikasi ini?",
        'about_what_is_it_desc': "Aplikasi prediksi penyakit jantung ini menggunakan algoritma machine learning canggih untuk menilai risiko penyakit jantung berdasarkan berbagai indikator kesehatan pasien.",
        'about_main_features_title': "✨ Fitur Utama",
        'about_feature_1': "**Prediksi Bertenaga AI:** Menggunakan algoritma Random Forest untuk akurasi prediksi yang tinggi.",
        'about_feature_2': "**Analisis Interaktif:** Menjelajahi data medis dengan visualisasi yang mudah dipahami.",
        'about_feature_3': "**Penilaian Risiko Personal:** Dapatkan evaluasi risiko penyakit jantung yang dipersonalisasi.",
        'about_feature_4': "**Ramah Pengguna:** Antarmuka yang sederhana untuk skrining kesehatan yang mudah.",
        'about_dataset_info_title': "📋 Informasi Dataset",
        'about_datasource_title': "📊 Sumber Data",
        'about_datasource_1': "**Sumber:** Kaggle - Heart Failure Prediction",
        'about_datasource_2': "**Rekaman:** 918 pasien",
        'about_datasource_3': "**Fitur:** 11 indikator medis",
        'about_datasource_4': "**Akurasi Model:** ~90%",
        'about_health_indicators_title': "📈 Indikator Kesehatan",
        'about_indicators_list': ["Usia", "Jenis Kelamin", "Tipe Nyeri Dada", "Tekanan Darah Istirahat", "Kolesterol", "Gula Darah Puasa", "Hasil EKG Istirahat", "Detak Jantung Maksimum", "Angina Akibat Olahraga", "Oldpeak", "Kemiringan ST"],
        'about_dev_team_title': "👥 Tim Pengembang & Pembimbing",
        'about_developer': "Pengembang",
        'about_advisor_1': "Dosen Pembimbing 1: Irani Hoeronis S.Si, M.T., CRP., CIISA.",
        'about_advisor_2': "Dosen Pembimbing 2: Siti Yulianti S.T., M.Kom.",
    },
    'en': {
        'banner_title': "Advanced Heart Disease Prediction System",
        'banner_subtitle': "Powered by Machine Learning & AI Technology",
        'nav_home': "🏠 Home", 'nav_predict': "🔍 Prediction", 'nav_about': "ℹ️ About",
        'nav_title': "📌 Navigation", 'lang_select_title': "Select Language",
        'model_info_title': "🎯 Model Info",
        'model_info_algo': "Algorithm: Random Forest",
        'model_info_recall': "Recall: 92%",
        'model_info_accuracy': "Accuracy: 90%",
        'model_info_features': "Features: 11 Medical Indicators",
        'home_intro_new': "Leverage the power of artificial intelligence to assess heart disease risk with precision and confidence.",
        'dataset_overview_title': "📊 Real-time Dataset Overview",
        'total_samples_title': "Total Samples", 'total_samples_desc': "Patient Records",
        'features_title': "Features", 'features_desc': "Health Indicators",
        'positive_cases_title': "Heart Disease Cases", 'positive_cases_desc': "Positive Cases",
        'healthy_cases_title': "Healthy Cases", 'healthy_cases_desc': "Negative Cases",
        'input_guide_title': "📘 Feature Input Guide",
        'age_title': "🎂 Age", 'age_range': "Data range in the model: **28 - 77 years**.",
        'resting_bp_title': "🩸 Resting BP", 'resting_bp_range': "Data range: **0 - 200 mm Hg**.",
        'cholesterol_title': "🧪 Cholesterol", 'cholesterol_range': "Data range: **0 - 603 mg/dl**.",
        'cholesterol_note': "💡 **Info**: A value of 0 likely indicates missing data. A clinically normal value is generally **< 200 mg/dl**.",
        'max_hr_title': "❤️‍🔥 Max HR", 'max_hr_range': "Data range: **60 - 202 bpm**.",
        'oldpeak_title': "📈 Oldpeak", 'oldpeak_range': "Data range: **-2.6 - 6.2**.",
        'categorical_title': "🗂️ Categorical Features",
        'categorical_desc': "For features like *Sex* or *Chest Pain Type*, you just need to select from the available options.",
        'home_disclaimer_title': "⚠️ Important Disclaimer",
        'home_disclaimer_md': "This application is a prototype for educational purposes and **must not be used for real medical diagnosis**. The prediction results do not replace consultation with a professional healthcare provider.",
        'form_intro': "Please fill in the patient's data below accurately to get a prediction.",
        'patient_data_header': "🩺 Patient Data",
        'age_label': "Age", 'sex_label': "Sex", 'chest_pain_type_label': "Chest Pain Type",
        'resting_bp_label': "Resting BP", 'cholesterol_label': "Cholesterol",
        'fasting_bs_label': "Fasting BS > 120 mg/dl?", 'resting_ecg_label': "Resting ECG",
        'max_hr_label': "Max HR", 'exercise_angina_label': "Exercise Angina?",
        'oldpeak_label': "Oldpeak", 'st_slope_label': "ST Slope",
        'predict_button': "🧠 Run Prediction & Analysis", 'spinner_text': "🔄 Analyzing data...",
        'result_header': "📊 AI Analysis Results",
        'risk_score': "Risk Score", 'risk_level': "Risk Level", 'recommendation': "Recommendation",
        'result_low_risk': "LOW RISK", 'result_medium_risk': "MEDIUM RISK", 'result_high_risk': "HIGH RISK",
        'recommendation_low': "Continue healthy lifestyle practices.",
        'recommendation_medium': "It is advisable to consult a doctor for monitoring.",
        'recommendation_high': "IMMEDIATELY consult a cardiologist.",
        'probability_breakdown': "🔗 Probability Breakdown",
        'risk_distribution': "Risk Probability Distribution",
        'no_disease': "No Heart Disease", 'disease': "Heart Disease",
        'model_confidence_title': "🎯 Model Confidence",
        'prediction_confidence_label': "Prediction Confidence",
        'no_disease_prob': "Probability of No Heart Disease", 'disease_prob': "Probability of Heart Disease",
        'lime_analysis_title': "🔬 Risk Factor Explanation (LIME Analysis)",
        'lime_explanation': "The plot below shows the factors that most influenced this prediction. Features with red bars support the 'Heart Disease' prediction, while green bars oppose it.",
        'lime_plot_title_prefix': "Local explanation for class", 'class_disease': "Disease",

        # KEYS FIXED AND RE-ADDED
        'prob_explanation_title': "💡 Probability Interpretation",
        'prob_explanation_high': "The patient's probability score **({score:.1f}%)** is above the high-risk threshold **(46%)**, thus classified as **HIGH RISK**.",
        'prob_explanation_medium': "The patient's probability score **({score:.1f}%)** is within the medium-risk threshold **(25% - 45.9%)**, thus classified as **MEDIUM RISK**.",
        'prob_explanation_low': "The patient's probability score **({score:.1f}%)** is below the medium-risk threshold **(25%)**, thus classified as **LOW RISK**.",
        
        'about_what_is_it_title': "🎯 What is this application?",
        'about_what_is_it_desc': "This heart disease prediction application uses an advanced machine learning algorithm to assess heart disease risk based on various patient health indicators.",
        'about_main_features_title': "✨ Main Features",
        'about_feature_1': "**AI-Powered Prediction:** Utilizes a Random Forest machine learning algorithm for high prediction accuracy.",
        'about_feature_2': "**Interactive Analysis:** Explore medical data with easy-to-understand visualizations.",
        'about_feature_3': "**Personalized Risk Assessment:** Get a personalized evaluation of your heart disease risk.",
        'about_feature_4': "**User-Friendly:** A simple and intuitive interface for easy health screening.",
        'about_dataset_info_title': "📋 Dataset Information",
        'about_datasource_title': "📊 Data Source",
        'about_datasource_1': "**Source:** Kaggle - Heart Failure Prediction",
        'about_datasource_2': "**Records:** 918 patients",
        'about_datasource_3': "**Features:** 11 medical indicators",
        'about_datasource_4': "**Model Accuracy:** ~90%",
        'about_health_indicators_title': "📈 Health Indicators",
        'about_indicators_list': ["Age", "Sex", "Chest Pain Type", "Resting Blood Pressure", "Cholesterol", "Fasting Blood Sugar", "Resting ECG", "Maximum Heart Rate", "Exercise Angina", "Oldpeak", "ST Slope"],
        'about_dev_team_title': "👥 Development & Advisory Team",
        'about_developer': "Developer",
        'about_advisor_1': "Advisor 1: Irani Hoeronis S.Si, M.T., CRP., CIISA.",
        'about_advisor_2': "Advisor 2: Siti Yulianti S.T., M.Kom.",
    }
}

# Fungsi untuk mengatur gaya CSS
def atur_gaya():
    st.markdown("""
    <style>
    .stApp { background-color: #ffffff; } 
    [data-testid="stSidebar"] { background-color: #1ff0b3; }
    .stButton button { width: 100%; border-radius: 8px; background-color: #4CAF50; color: white; } 
    .stSpinner > div > div { border-top-color: #4CAF50; }
    .banner { background: linear-gradient(135deg, #0CE8BC 0%, #764ba2 100%); color: white; padding: 30px 20px; border-radius: 15px; text-align: center; margin-bottom: 20px; }
    .banner-title { font-size: 36px; font-weight: bold; display: flex; align-items: center; justify-content: center; }
    .banner-icon { font-size: 40px; margin-right: 15px; }
    .banner-subtitle { font-size: 16px; margin-top: 10px; opacity: 0.9; }
    .result-box { border-radius: 10px; padding: 25px; margin-bottom: 20px; color: #31333F; }
    .result-box-low { background-color: #E9F5E9; border-left: 6px solid #4CAF50; }
    .result-box-medium { background-color: #FFF8E1; border-left: 6px solid #FFC107; }
    .result-box-high { background-color: #FFEBEE; border-left: 6px solid #F44336; }
    .result-title { font-size: 24px; font-weight: bold; display: flex; align-items: center; }
    .result-icon { font-size: 28px; margin-right: 10px; }
    .result-score { font-size: 18px; font-weight: bold; margin-top: 10px; color: #555; }
    .result-recommendation { font-size: 14px; margin-top: 10px; color: #666; }
    </style>
    """, unsafe_allow_html=True)

# ==============================================================================
# 2. FUNGSI-FUNGSI UTAMA (LOGIKA & HALAMAN)
# ==============================================================================

def tampilkan_header_banner(T):
    st.markdown(f"""
    <div class="banner">
        <div class="banner-title">
            <span class="banner-icon">🫀</span>
            <span>{T['banner_title']}</span>
        </div>
        <p class="banner-subtitle">{T['banner_subtitle']}</p>
    </div>
    """, unsafe_allow_html=True)

@st.cache_resource
def setup_resources():
    try:
        model = joblib.load("model_pipeline_terbaik.pkl")
        df = pd.read_csv('heart.csv')
        X = df.drop('HeartDisease', axis=1)
        y = df['HeartDisease']
        X_train, _, _, _ = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        preprocessor = model.named_steps['preprocessor']
        X_train_processed = preprocessor.fit_transform(X_train)
        explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data=X_train_processed,
            feature_names=preprocessor.get_feature_names_out(),
            class_names=['No Disease', 'Disease'],
            mode='classification'
        )
        return model, explainer, preprocessor, df
    except FileNotFoundError:
        st.error("Gagal memuat file 'model_pipeline_terbaik.pkl' atau 'heart.csv'. Pastikan file-file tersebut ada di direktori yang sama.")
        return None, None, None, None

def create_gauge_chart(probability_score, T):
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta", value = probability_score, title = {'text': T['risk_level'], 'font': {'size': 20}},
        number = {'suffix': "%", 'font': {'size': 16}},
        gauge = {'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"}, 'bar': {'color': "rgba(0,0,0,0)"},
                 'steps': [{'range': [0, 46], 'color': 'green'}, {'range': [46, 75], 'color': 'yellow'}, {'range': [75, 100], 'color': 'red'}],
                 'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.9, 'value': probability_score}},
        domain = {'x': [0, 1], 'y': [0, 1]}))
    fig.update_layout(height=250, margin=dict(l=10, r=10, t=50, b=10))
    return fig

def create_pie_chart(probability_score, T):
    labels = [T['no_disease'], T['disease']]; values = [100 - probability_score, probability_score]; colors = ['green', 'red']
    fig = px.pie(values=values, names=labels, color_discrete_sequence=colors, hole=.4)
    fig.update_layout(title_text=T['risk_distribution'], showlegend=True, height=300, margin=dict(l=10, r=10, t=40, b=10))
    return fig

def atur_navigasi(T):
    with st.sidebar:
        st.markdown(f"### {T['lang_select_title']}")
        selected_lang_display = option_menu(menu_title=None, options=["Indonesia", "English"], icons=["globe2", "globe-americas"], menu_icon="translate", default_index=0 if st.session_state.lang == 'id' else 1, orientation="horizontal", key="lang_menu", styles={"container": {"padding": "0!important", "background-color": "#e0e0e0", "border-radius": "8px"}, "nav-link-selected": {"background-color": "#2c3e50", "color": "white"}})
        new_lang = 'id' if selected_lang_display == 'Indonesia' else 'en'
        if st.session_state.lang != new_lang: st.session_state.lang = new_lang; st.rerun()
        st.markdown(f"## {T['nav_title']}")
        selected_page = option_menu(menu_title=None, options=[T['nav_home'], T['nav_predict'], T['nav_about']], icons=["house-heart-fill", "search-heart", "info-circle-fill"], menu_icon="cast", default_index=0, key="main_menu", styles={"container": {"padding": "5px", "background-color": "#ffffff", "border-radius": "10px"}, "nav-link-selected": {"background-color": "#4CAF50"}})
        st.markdown("<br>", unsafe_allow_html=True)
        with st.container():
            st.markdown(f"**{T['model_info_title']}**")
            st.markdown(f"""
            - {T['model_info_algo']}
            - {T['model_info_recall']}
            - {T['model_info_accuracy']}
            - {T['model_info_features']}
            """)
    return selected_page

def tampilkan_halaman_home(T, df):
    st.markdown(f"#### {T['home_intro_new']}")
    st.markdown("---")
    if df is None: st.warning("Data `heart.csv` tidak ditemukan."); return
    total_sampel = df.shape[0]; total_fitur = df.shape[1] - 1; kasus_positif = df['HeartDisease'].sum(); kasus_negatif = total_sampel - kasus_positif
    st.subheader(T['dataset_overview_title'])
    st.markdown("""<style>div[data-testid="metric-container"] {background-color: #667eea;background-image: linear-gradient(135deg, #667eea 0%, #764ba2 100%);border: 1px solid rgba(255, 255, 255, 0.2);border-radius: 10px;padding: 20px;color: white;}div[data-testid="metric-container"] > div, div[data-testid="metric-container"] label {color: white;}</style>""", unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    with col1: st.metric(label=f"🗂️ {T['total_samples_title']}", value=total_sampel, help=T['total_samples_desc'])
    with col2: st.metric(label=f"⚙️ {T['features_title']}", value=total_fitur, help=T['features_desc'])
    with col3: st.metric(label=f"⚠️ {T['positive_cases_title']}", value=kasus_positif, help=T['positive_cases_desc'])
    with col4: st.metric(label=f"✅ {T['healthy_cases_title']}", value=kasus_negatif, help=T['healthy_cases_desc'])
    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader(T['input_guide_title'])
    col1, col2 = st.columns(2)
    with col1:
        with st.container(border=True): st.markdown(f"**{T['age_title']}**"); st.write(T['age_range'])
        with st.container(border=True): st.markdown(f"**{T['resting_bp_title']}**"); st.write(T['resting_bp_range'])
        with st.container(border=True): st.markdown(f"**{T['cholesterol_title']}**"); st.write(T['cholesterol_range']); st.write(T['cholesterol_note'])
    with col2:
        with st.container(border=True): st.markdown(f"**{T['max_hr_title']}**"); st.write(T['max_hr_range'])
        with st.container(border=True): st.markdown(f"**{T['oldpeak_title']}**"); st.write(T['oldpeak_range'])
        with st.container(border=True): st.markdown(f"**{T['categorical_title']}**"); st.write(T['categorical_desc'])
    st.markdown("<br>", unsafe_allow_html=True)
    st.error(f"**{T['home_disclaimer_title']}**\n{T['home_disclaimer_md']}")

def tampilkan_halaman_prediksi(T, model, explainer, preprocessor):
    st.info(T['form_intro'])
    with st.container(border=True):
        with st.form("prediction_form"):
            st.header(T['patient_data_header'])
            col1, col2, col3 = st.columns(3)
            with col1:
                age = st.number_input(T['age_label'], min_value=15, max_value=80, value=58, step=1); sex = st.selectbox(T['sex_label'], ['M', 'F']); chest_pain_type = st.selectbox(T['chest_pain_type_label'], ['ATA', 'NAP', 'ASY', 'TA'])
            with col2:
                resting_bp = st.number_input(T['resting_bp_label'], min_value=50, max_value=250, value=130, step=1); cholesterol = st.number_input(T['cholesterol_label'], min_value=0, max_value=610, value=240, step=1); fasting_bs = st.selectbox(T['fasting_bs_label'], [0, 1], format_func=lambda x: "Ya" if x == 1 else "Tidak")
            with col3:
                max_hr = st.number_input(T['max_hr_label'], min_value=50, max_value=220, value=150, step=1); exercise_angina = st.selectbox(T['exercise_angina_label'], ['Y', 'N']); oldpeak = st.number_input(T['oldpeak_label'], min_value=-3.0, max_value=7.0, value=1.0, step=0.1, format="%.1f"); st_slope = st.selectbox(T['st_slope_label'], ['Up', 'Flat', 'Down']); resting_ecg = st.selectbox(T['resting_ecg_label'], ['Normal', 'ST', 'LVH'])
            submitted = st.form_submit_button(T['predict_button'], use_container_width=True, type="primary")
    if submitted:
        with st.spinner(T['spinner_text']):
            input_data = pd.DataFrame({'Age': [age], 'Sex': [sex], 'ChestPainType': [chest_pain_type], 'RestingBP': [resting_bp], 'Cholesterol': [cholesterol], 'FastingBS': [fasting_bs], 'RestingECG': [resting_ecg], 'MaxHR': [max_hr], 'ExerciseAngina': [exercise_angina], 'Oldpeak': [oldpeak], 'ST_Slope': [st_slope]})
            probabilitas = model.predict_proba(input_data); prob_sakit = probabilitas[0][1]; input_processed = preprocessor.transform(input_data)
            explanation = explainer.explain_instance(input_processed[0], model.named_steps['classifier'].predict_proba, num_features=11, labels=(1,))
        st.divider()
        st.header(T['result_header'])
        prob_sakit_percent = prob_sakit * 100
        if prob_sakit_percent >= 46: risk_text, risk_class, risk_icon, recommendation_text = T['result_high_risk'], "high", "🔴", T['recommendation_high']
        elif prob_sakit_percent >= 25: risk_text, risk_class, risk_icon, recommendation_text = T['result_medium_risk'], "medium", "🟡", T['recommendation_medium']
        else: risk_text, risk_class, risk_icon, recommendation_text = T['result_low_risk'], "low", "🟢", T['recommendation_low']
        col1, col2 = st.columns([0.6, 0.4])
        with col1: st.markdown(f"""<div class="result-box result-box-{risk_class}"><div class="result-title"><span class="result-icon">{risk_icon}</span><span>{risk_text}</span></div><div class="result-score">{T['risk_score']}: {prob_sakit_percent:.1f}%</div><div class="result-recommendation">{recommendation_text}</div></div>""", unsafe_allow_html=True)
        with col2: st.plotly_chart(create_gauge_chart(prob_sakit_percent, T), use_container_width=True)
        st.divider()
        st.subheader(T['probability_breakdown'])
        col1, col2 = st.columns([0.4, 0.6])
        with col1:
            with st.container(border=True): st.plotly_chart(create_pie_chart(prob_sakit_percent, T), use_container_width=True)
        with col2:
            with st.container(border=True):
                st.markdown(f"**{T['model_confidence_title']}**")
                st.markdown(f"**{T['prediction_confidence_label']}:** `{max(prob_sakit, 1-prob_sakit):.1%}`")
                st.markdown(f"**{T['no_disease_prob']}:** `{1-prob_sakit:.1%}`")
                st.markdown(f"**{T['disease_prob']}:** `{prob_sakit:.1%}`")
                st.markdown("---"); st.info(T['prob_explanation_low'].format(score=prob_sakit_percent) if risk_class == 'low' else T['prob_explanation_medium'].format(score=prob_sakit_percent) if risk_class == 'medium' else T['prob_explanation_high'].format(score=prob_sakit_percent))
        with st.expander(T['lime_analysis_title']):
            st.markdown(T['lime_explanation']); fig = explanation.as_pyplot_figure(label=1); ax = plt.gca(); bars = ax.patches; exp_list = explanation.as_list(label=1); exp_list.reverse()
            for i, patch in enumerate(bars):
                if i < len(exp_list):
                    weight = exp_list[i][1]
                    if weight > 0: patch.set_color('red')
                    else: patch.set_color('green')
            ax.set_title(f"{T['lime_plot_title_prefix']} {T['class_disease']}", fontsize=15); st.pyplot(fig, use_container_width=True); plt.clf()

def tampilkan_halaman_about(T):
    # --- Bagian 1: Apa itu aplikasi ini? ---
    st.markdown(f"### {T['about_what_is_it_title']}")
    with st.container(border=True):
        st.write(T['about_what_is_it_desc'])

    # --- Bagian 2: Fitur Utama ---
    st.markdown(f"### {T['about_main_features_title']}")
    with st.container(border=True):
        st.markdown(f"- {T['about_feature_1']}")
        st.markdown(f"- {T['about_feature_2']}")
        st.markdown(f"- {T['about_feature_3']}")
        st.markdown(f"- {T['about_feature_4']}")

    # --- Bagian 3: Informasi Dataset ---
    st.markdown(f"### {T['about_dataset_info_title']}")
    col1, col2 = st.columns(2)
    with col1:
        with st.container(border=True, height=280):
            st.markdown(f"#### {T['about_datasource_title']}")
            st.markdown(f"- {T['about_datasource_1']}")
            st.markdown(f"- {T['about_datasource_2']}")
            st.markdown(f"- {T['about_datasource_3']}")
            st.markdown(f"- {T['about_datasource_4']}")
    with col2:
        with st.container(border=True, height=280):
            st.markdown(f"#### {T['about_health_indicators_title']}")
            indicators = T['about_indicators_list']
            sub_col1, sub_col2 = st.columns(2)
            with sub_col1:
                for item in indicators[:6]:
                    st.markdown(f"- {item}")
            with sub_col2:
                for item in indicators[6:]:
                    st.markdown(f"- {item}")
    
    # --- Bagian 4: Tim Pengembang & Pembimbing ---
    st.markdown(f"### {T['about_dev_team_title']}")
    with st.container(border=True):
        col1, col2 = st.columns([0.3, 0.7])
        with col1:
            try:
                st.image("gambar/lutfi.jpg", use_column_width=True)
            except Exception:
                st.warning("Gambar lutfi.jpg tidak ditemukan.")
        with col2:
            st.markdown(f"**Lutfi Julpian**")
            st.markdown(f"_{T['about_developer']}_")
            st.markdown("---")
            st.markdown(f"**{T['about_advisor_1']}**")
            st.markdown(f"**{T['about_advisor_2']}**")


# ==============================================================================
# 3. FUNGSI UTAMA (MAIN) UNTUK MENJALANKAN APLIKASI
# ==============================================================================

def main():
    atur_gaya()
    if 'lang' not in st.session_state: st.session_state.lang = 'id'
    
    T = TRANSLATIONS[st.session_state.lang]
    model, explainer, preprocessor, df = setup_resources()
    selected_page = atur_navigasi(T)
    
    tampilkan_header_banner(T)

    if selected_page == T['nav_home']:
        tampilkan_halaman_home(T, df)
    elif selected_page == T['nav_predict']:
        if model is None: st.error(T['model_load_error'])
        else: tampilkan_halaman_prediksi(T, model, explainer, preprocessor)
    elif selected_page == T['nav_about']:
        tampilkan_halaman_about(T)

if __name__ == "__main__":
    main()