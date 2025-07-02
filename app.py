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
import seaborn as sns

# ==============================================================================
# 1. KONFIGURASI, DATA STATIS, DAN GAYA (CSS)
# ==============================================================================

st.set_page_config(
    page_title="Prediksi Penyakit Jantung",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Kamus Terjemahan
TRANSLATIONS = {
    'id': {
        'banner_title': "Sistem Prediksi Penyakit Jantung",
        'banner_subtitle': "Didukung oleh Teknologi Machine Learning & AI",
        'nav_home': "🏠 Home", 'nav_predict': "🔍 Prediksi", 'nav_about': "ℹ️ Tentang",
        'nav_title': "📌 Navigasi", 'lang_select_title': "Pilih Bahasa",
        'model_info_title': "🎯 Info Model",
        'model_info_algo': "Algoritma: Random Forest",
        'model_info_recall': "Recall: 92%",
        'model_info_accuracy': "Akurasi: 90%",
        'model_info_features': "Fitur: 11 Indikator Medis",
        'home_intro_new': "Manfaatkan kekuatan kecerdasan buatan untuk menilai risiko penyakit jantung dengan presisi dan kepercayaan diri.",
        'dataset_overview_title': "📊 Tinjauan Dataset",
        'total_samples_title': "Total Sampel", 'total_samples_desc': "Rekam Pasien",
        'features_title': "Fitur", 'features_desc': "Indikator Kesehatan",
        'positive_cases_title': "Kasus Sakit Jantung", 'positive_cases_desc': "Kasus Positif",
        'healthy_cases_title': "Kasus Sehat", 'healthy_cases_desc': "Kasus Negatif",
        'input_guide_title': "📘 Panduan Input Fitur",
        'age_title': "🎂 Usia (Age)", 'age_range': "Rentang data pada model: **15 - 80 tahun**.",
        'resting_bp_title': "🩸 Tekanan Darah (RestingBP)", 'resting_bp_range': "Rentang data: **80 - 200 mm Hg**.",
        'cholesterol_title': "🧪 Kolesterol (Cholesterol)", 'cholesterol_range': "Rentang data: **100 - 600 mg/dl**.",
        'cholesterol_note': "💡 **Info** Nilai normal klinis umumnya **< 200 mg/dl**.",
        'max_hr_title': "❤️‍🔥 Detak Jantung Maks. (MaxHR)", 'max_hr_range': "Rentang data: **70 - 205 bpm**.",
        'oldpeak_title': "📈 Oldpeak", 'oldpeak_range': "Rentang data: **-2.6 - 6.5**.",
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
        'about_datasource_1': "**Sumber:** Kaggle - Heart Failure Prediction (https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction)",
        'about_datasource_2': "**Rekaman:** 918 pasien",
        'about_datasource_3': "**Fitur:** 11 indikator medis",
        'about_datasource_4': "**Recall:** ~92%",
        'about_datasource_5': "**Akurasi Model:** ~90%",
        'about_health_indicators_title': "📈 Indikator Kesehatan",
        'about_indicators_list': ["Usia", "Jenis Kelamin", "Tipe Nyeri Dada", "Tekanan Darah Istirahat", "Kolesterol", "Gula Darah Puasa", "Hasil EKG Istirahat", "Detak Jantung Maksimum", "Angina Akibat Olahraga", "Oldpeak", "Kemiringan ST"],
        'about_dev_team_title': "👥 Tim Pengembang & Pembimbing",
        'about_developer': "Pengembang",
        'about_advisor_1': "Dosen Pembimbing 1: Irani Hoeronis S.Si, M.T., CRP., CIISA.",
        'about_advisor_2': "Dosen Pembimbing 2: Siti Yulianti S.T., M.Kom.",
        'risk_key_title': "🔑 Kunci Kategori Risiko",
        'risk_key_low_title': "🟢 Risiko Rendah",
        'risk_key_low_desc': "Probabilitas < 25%",
        'risk_key_medium_title': "🟡 Risiko Sedang",
        'risk_key_medium_desc': "Probabilitas 25% - 45.9%",
        'risk_key_high_title': "🔴 Risiko Tinggi",
        'risk_key_high_desc': "Probabilitas ≥ 46%",
        # ... (kunci-kunci yang sudah ada)
        'data_analysis_title': "Visualisasi Analisis Data",
        'class_disease': "Sakit Jantung",
        'class_healthy': "Sehat",

        'tab_target_dist': "Distribusi Target",
        'tab_age_dist': "Distribusi Usia",
        'tab_feature_relation': "Hubungan Fitur",
        'tab_correlation': "Korelasi Fitur",

        'target_dist_title': "Distribusi Kelas Target",
        'target_dist_desc': "Grafik ini menunjukkan proporsi jumlah pasien yang didiagnosis menderita penyakit jantung dibandingkan dengan yang sehat dalam dataset.",
        'target_dist_pie_title': "Proporsi Pasien: Sakit Jantung vs. Sehat",

        'age_dist_title': "Distribusi Usia Pasien",
        'age_dist_desc': "Histogram ini menunjukkan sebaran usia pasien. Anda dapat melihat kelompok usia mana yang paling banyak terwakili dalam data, dipisahkan berdasarkan status kesehatan mereka.",
        'age_dist_hist_title': "Distribusi Usia Berdasarkan Status Penyakit Jantung",

        'feature_relation_title': "Hubungan Antara Fitur dan Diagnosis",
        'feature_relation_desc': "Grafik di bawah ini mengeksplorasi bagaimana fitur-fitur tertentu seperti kolesterol dan jenis kelamin berhubungan dengan kemungkinan diagnosis penyakit jantung.",
        'chol_vs_disease_title': "Distribusi Kolesterol vs. Penyakit Jantung",
        'sex_vs_disease_title': "Jumlah Kasus Berdasarkan Jenis Kelamin",

        'correlation_title': "Matriks Korelasi Fitur Numerik",
        'correlation_desc': "Heatmap ini menunjukkan korelasi antara berbagai fitur numerik. Nilai mendekati 1 (merah tua) atau -1 (biru tua) menunjukkan hubungan yang kuat, sedangkan nilai mendekati 0 (putih) menunjukkan hubungan yang lemah.",
        'correlation_heatmap_title': "Heatmap Korelasi Fitur",

        # <<< MULAI KODE BARU >>>
        'tab_feature_importance': "Pentingnya Fitur",
        'feature_importance_title': "Tingkat Kepentingan Fitur (Feature Importance)",
        'feature_importance_desc': "Grafik ini menunjukkan fitur mana yang paling berpengaruh secara global terhadap prediksi model Random Forest. Semakin tinggi nilainya, semakin penting fitur tersebut bagi model.",
        'feature_importance_chart_title': "Fitur Paling Berpengaruh (Random Forest Feature Importance)",
        'feature_importance_xaxis': "Tingkat Kepentingan (Importance)",
        'feature_importance_yaxis': "Fitur"
        # <<< AKHIR KODE BARU >>>
    },
    'en': { # English translations need to be fully populated for a real app
        'banner_title': "Heart Disease Prediction System",
        'banner_subtitle': "Powered by Machine Learning & AI",
        'nav_home': "🏠 Home", 'nav_predict': "🔍 Predict", 'nav_about': "ℹ️ About",
        'nav_title': "📌 Navigation", 'lang_select_title': "Select Language",
        'model_info_title': "🎯 Model Info",
        'model_info_algo': "Algorithm: Random Forest",
        'model_info_recall': "Recall: 92%",
        'model_info_accuracy': "Accuracy: 90%",
        'model_info_features': "Features: 11 Medical Indicators",
        'home_intro_new': "Harness the power of artificial intelligence to assess heart disease risk with precision and confidence.",
        'dataset_overview_title': "📊 Dataset Overview",
        'total_samples_title': "Total Samples", 'total_samples_desc': "Patient Records",
        'features_title': "Features", 'features_desc': "Health Indicators",
        'positive_cases_title': "Heart Disease Cases", 'positive_cases_desc': "Positive Cases",
        'healthy_cases_title': "Healthy Cases", 'healthy_cases_desc': "Negative Cases",
        'input_guide_title': "📘 Feature Input Guide",
        'age_title': "🎂 Age", 'age_range': "Model data range: **15 - 80 years**.",
        'resting_bp_title': "🩸 Resting Blood Pressure", 'resting_bp_range': "Data range: **80 - 200 mm Hg**.",
        'cholesterol_title': "🧪 Cholesterol", 'cholesterol_range': "Data range: **100 - 600 mg/dl**.",
        'cholesterol_note': "💡 **Note** Clinical normal value is usually **< 200 mg/dl**.",
        'max_hr_title': "❤️‍🔥 Max Heart Rate", 'max_hr_range': "Data range: **70 - 205 bpm**.",
        'oldpeak_title': "📈 Oldpeak", 'oldpeak_range': "Data range: **-2.6 - 6.5**.",
        'categorical_title': "🗂️ Categorical Features",
        'categorical_desc': "For features like *Sex* or *Chest Pain Type*, just select from the available options.",
        'home_disclaimer_title': "⚠️ Important Disclaimer",
        'home_disclaimer_md': "This application is a prototype for educational purposes and **should not be used for real medical diagnosis**. Predictions do not replace consultation with a healthcare professional.",
        'form_intro': "Please fill in the patient's data below accurately to receive a prediction.",
        'patient_data_header': "🩺 Patient Data",
        'age_label': "Age", 'sex_label': "Sex", 'chest_pain_type_label': "Chest Pain Type",
        'resting_bp_label': "Resting Blood Pressure", 'cholesterol_label': "Cholesterol",
        'fasting_bs_label': "Fasting Blood Sugar > 120 mg/dl?", 'resting_ecg_label': "Resting ECG Results",
        'max_hr_label': "Maximum Heart Rate", 'exercise_angina_label': "Exercise-Induced Angina?",
        'oldpeak_label': "Oldpeak", 'st_slope_label': "ST Slope",
        'predict_button': "🧠 Run Prediction & Analysis", 'spinner_text': "🔄 Analyzing data...",
        'result_header': "📊 AI Analysis Result",
        'risk_score': "Risk Score", 'risk_level': "Risk Level", 'recommendation': "Recommendation",
        'result_low_risk': "LOW RISK", 'result_medium_risk': "MEDIUM RISK", 'result_high_risk': "HIGH RISK",
        'recommendation_low': "Continue a healthy lifestyle.",
        'recommendation_medium': "It is recommended to consult a doctor for monitoring.",
        'recommendation_high': "IMMEDIATELY consult a heart specialist.",
        'probability_breakdown': "🔗 Probability Breakdown",
        'risk_distribution': "Risk Probability Distribution",
        'no_disease': "No Heart Disease", 'disease': "Heart Disease",
        'model_confidence_title': "🎯 Model Confidence",
        'prediction_confidence_label': "Prediction Confidence",
        'no_disease_prob': "Probability of No Heart Disease", 'disease_prob': "Probability of Heart Disease",
        'lime_analysis_title': "🔬 Risk Factor Explanation (LIME Analysis)",
        'lime_explanation': "The chart below shows the most influential factors for this patient's prediction. Red bars support the 'Heart Disease' prediction, while green bars oppose it.",
        'lime_plot_title_prefix': "Local explanation for class", 'class_disease': "Disease",
        'prob_explanation_title': "💡 Probability Interpretation",
        'prob_explanation_high': "The patient's probability score **({score:.1f}%)** is above the high-risk threshold **(46%)**, hence classified as **HIGH RISK**.",
        'prob_explanation_medium': "The patient's probability score **({score:.1f}%)** is between the moderate-risk threshold **(25% - 45.9%)**, hence classified as **MEDIUM RISK**.",
        'prob_explanation_low': "The patient's probability score **({score:.1f}%)** is below the moderate-risk threshold **(25%)**, hence classified as **LOW RISK**.",
        'about_what_is_it_title': "🎯 What is this app?",
        'about_what_is_it_desc': "This heart disease prediction app uses advanced machine learning algorithms to assess heart disease risk based on various patient health indicators.",
        'about_main_features_title': "✨ Main Features",
        'about_feature_1': "**AI-Powered Predictions:** Utilizes Random Forest algorithm for high prediction accuracy.",
        'about_feature_2': "**Interactive Analysis:** Explore medical data with easy-to-understand visualizations.",
        'about_feature_3': "**Personalized Risk Assessment:** Get individualized heart disease risk evaluation.",
        'about_feature_4': "**User-Friendly:** Simple interface for easy health screening.",
        'about_dataset_info_title': "📋 Dataset Information",
        'about_datasource_title': "📊 Data Source",
        'about_datasource_1': "**Source:** Kaggle - Heart Failure Prediction (https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction)",
        'about_datasource_2': "**Records:** 918 patients",
        'about_datasource_3': "**Features:** 11 medical indicators",
        'about_datasource_4': "**Recall:** ~92%",
        'about_datasource_5': "**Model Accuracy:** ~90%",
        'about_health_indicators_title': "📈 Health Indicators",
        'about_indicators_list': ["Age", "Sex", "Chest Pain Type", "Resting Blood Pressure", "Cholesterol", "Fasting Blood Sugar", "Resting ECG", "Max Heart Rate", "Exercise-Induced Angina", "Oldpeak", "ST Slope"],
        'about_dev_team_title': "👥 Developer & Supervisors",
        'about_developer': "Developer",
        'about_advisor_1': "Advisor 1: Irani Hoeronis S.Si, M.T., CRP., CIISA.",
        'about_advisor_2': "Advisor 2: Siti Yulianti S.T., M.Kom.",
        'risk_key_title': "🔑 Risk Category Key",
        'risk_key_low_title': "🟢 Low Risk",
        'risk_key_low_desc': "Probability < 25%",
        'risk_key_medium_title': "🟡 Medium Risk",
        'risk_key_medium_desc': "Probability 25% - 45.9%",
        'risk_key_high_title': "🔴 High Risk",
        'risk_key_high_desc': "Probability ≥ 46%",
        'data_analysis_title': "Data Analysis Visualizations",
        'class_disease': "Heart Disease",
        'class_healthy': "Healthy",

        'tab_target_dist': "Target Distribution",
        'tab_age_dist': "Age Distribution",
        'tab_feature_relation': "Feature Relationship",
        'tab_correlation': "Feature Correlation",

        'target_dist_title': "Target Class Distribution",
        'target_dist_desc': "This chart shows the proportion of patients diagnosed with heart disease versus those who are healthy in the dataset.",
        'target_dist_pie_title': "Patient Proportion: Heart Disease vs. Healthy",

        'age_dist_title': "Patient Age Distribution",
        'age_dist_desc': "This histogram shows the age distribution of patients. You can see which age groups are most represented in the data, separated by health status.",
        'age_dist_hist_title': "Age Distribution by Heart Disease Status",

        'feature_relation_title': "Feature-Outcome Relationship",
        'feature_relation_desc': "The charts below explore how features like cholesterol and sex are related to the likelihood of a heart disease diagnosis.",
        'chol_vs_disease_title': "Cholesterol vs. Heart Disease Distribution",
        'sex_vs_disease_title': "Cases by Sex",

        'correlation_title': "Numerical Feature Correlation Matrix",
        'correlation_desc': "This heatmap shows correlations between various numerical features. Values close to 1 (dark red) or -1 (dark blue) indicate strong relationships, while values close to 0 (white) suggest weak ones.",
        'correlation_heatmap_title': "Feature Correlation Heatmap",

        # <<< MULAI KODE BARU >>>
        'tab_feature_importance': "Feature Importance",
        'feature_importance_title': "Feature Importance Level",
        'feature_importance_desc': "This chart shows which features are most influential globally for the Random Forest model's predictions. A higher value means the feature is more important to the model.",
        'feature_importance_chart_title': "Most Influential Features (Random Forest Feature Importance)",
        'feature_importance_xaxis': "Importance Level",
        'feature_importance_yaxis': "Feature"
        # <<< AKHIR KODE BARU >>>
    }
}

# Fungsi untuk mengatur gaya CSS
def atur_gaya():
    st.markdown("""
    <style>
    .stApp { background-color: #ffffff; } 
    [data-testid="stSidebar"] { background-color: #F0F2F6; }
    .stButton button { 
        width: 100%; border-radius: 8px; border: 1px solid #0d6efd;
        background-color: #0d6efd; color: white; 
    }
    .stButton button:hover { background-color: #0b5ed7; border: 1px solid #0b5ed7; color: white; }
    .stSpinner > div > div { border-top-color: #0d6efd; }
    .banner { background: #FFFFFF; color: #31333F; padding: 2rem; border-radius: 15px;
              border-left: 8px solid #0d6efd; box-shadow: 0 4px 8px rgba(0,0,0,0.1); margin-bottom: 2rem; }
    .banner-title { font-size: 2.5rem; font-weight: bold; display: flex; align-items: center; }
    .banner-icon { font-size: 2.8rem; margin-right: 1.5rem; color: #0d6efd; }
    .banner-subtitle { font-size: 1.1rem; margin-top: 0.5rem; opacity: 0.8; }
    .result-box { border-radius: 10px; padding: 25px; margin-bottom: 20px; color: #31333F; }
    .result-box-low { background-color: #E9F5E9; border-left: 6px solid #28a745; }
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
            <div>
                <span>{T['banner_title']}</span>
                <p class="banner-subtitle">{T['banner_subtitle']}</p>
            </div>
        </div>
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
        st.error("Gagal memuat file 'model_pipeline_terbaik.pkl' atau 'heart.csv'.")
        return None, None, None, None

def create_gauge_chart(probability_score, T):
    # ... (fungsi ini tidak berubah)
    fig = go.Figure(go.Indicator(
        mode = "gauge+number", value = probability_score,
        title = {'text': T['risk_level'], 'font': {'size': 20}},
        number = {'suffix': "%", 'font': {'size': 24}},
        gauge = {
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "gray"},
            'bar': {'color': "#0d6efd"}, 
            'steps': [
                {'range': [0, 25], 'color': '#E9F5E9'},
                {'range': [25, 46], 'color': '#FFF8E1'},
                {'range': [46, 100], 'color': '#FFEBEE'}],
        }))
    fig.update_layout(height=250, margin=dict(l=20, r=20, t=50, b=20))
    return fig

# ==========================================================
# [MODIFIKASI] - Mengubah pemetaan warna pada pie chart
# ==========================================================
def create_pie_chart(probability_score, T):
    labels = [T['no_disease'], T['disease']]
    values = [100 - probability_score, probability_score]
    
    # Pemetaan warna yang eksplisit dan benar
    color_map = {
        T['no_disease']: '#0d6efd',  # Hijau untuk "Tidak Sakit Jantung"
        T['disease']: '#FF4B4B'      # Merah untuk "Sakit Jantung"
    }

    fig = px.pie(
        values=values, 
        names=labels, 
        color=labels,
        color_discrete_map=color_map, # Menggunakan pemetaan warna eksplisit
        hole=.4
    )
    
    fig.update_layout(
        title_text=T['risk_distribution'], 
        showlegend=True, 
        height=300, 
        margin=dict(l=10, r=10, t=40, b=10)
    )
    return fig


def atur_navigasi(T):
    with st.sidebar:
        st.markdown(f"### {T['lang_select_title']}")
        selected_lang_display = option_menu(menu_title=None, options=["Indonesia", "English"], icons=["globe2", "globe-americas"], menu_icon="translate", default_index=0 if st.session_state.lang == 'id' else 1, orientation="horizontal", key="lang_menu", 
        styles={
            "container": {"background-color": "#eee", "border-radius": "8px"},
            "nav-link-selected": {"background-color": "#0d6efd"}
        })
        new_lang = 'id' if selected_lang_display == 'Indonesia' else 'en'
        if st.session_state.lang != new_lang: st.session_state.lang = new_lang; st.rerun()
        st.markdown(f"### {T['nav_title']}")
        selected_page = option_menu(menu_title=None, options=[T['nav_home'], T['nav_predict'], T['nav_about']], icons=["house-heart-fill", "search-heart", "info-circle-fill"], menu_icon="cast", default_index=0, key="main_menu", 
        styles={
            "container": {"padding": "5px !important", "background-color": "#F0F2F6"},
            "nav-link-selected": {"background-color": "#0d6efd"}
        })
        st.markdown("<br>", unsafe_allow_html=True)
        with st.container(border=True):
            st.markdown(f"**{T['model_info_title']}**")
            st.markdown(f"""
            - {T['model_info_algo']}
            - {T['model_info_recall']}
            - {T['model_info_accuracy']}
            - {T['model_info_features']}
            """)
    return selected_page

# <<< UBAH DEFINISI FUNGSI INI >>>
def tampilkan_halaman_home(T, df, model): # Tambahkan 'model' sebagai parameter
    st.markdown(f"#### {T['home_intro_new']}")
    st.markdown("---")
    if df is None: st.warning("Data `heart.csv` tidak ditemukan."); return

    # Tinjauan Dataset
    st.subheader(T['dataset_overview_title'])
    col1, col2, col3, col4 = st.columns(4)
    with col1: st.metric(label=f"🗂️ {T['total_samples_title']}", value=df.shape[0], help=T['total_samples_desc'])
    with col2: st.metric(label=f"⚙️ {T['features_title']}", value=df.shape[1] - 1, help=T['features_desc'])
    with col3: st.metric(label=f"⚠️ {T['positive_cases_title']}", value=df['HeartDisease'].sum(), help=T['positive_cases_desc'])
    with col4: st.metric(label=f"✅ {T['healthy_cases_title']}", value=df.shape[0] - df['HeartDisease'].sum(), help=T['healthy_cases_desc'])
    
    st.subheader(T['input_guide_title'])

    col1, col2 = st.columns(2)

    with col1:

        with st.container(border=True): st.markdown(f"**{T['age_title']}**"); st.write(T['age_range'])

        with st.container(border=True): st.markdown(f"**{T['resting_bp_title']}**"); st.write(T['resting_bp_range'])

        with st.container(border=True): st.markdown(f"**{T['cholesterol_title']}**"); st.write(T['cholesterol_range']); st.info(T['cholesterol_note'])

    with col2:

        with st.container(border=True): st.markdown(f"**{T['max_hr_title']}**"); st.write(T['max_hr_range'])

        with st.container(border=True): st.markdown(f"**{T['oldpeak_title']}**"); st.write(T['oldpeak_range'])

        with st.container(border=True): st.markdown(f"**{T['categorical_title']}**"); st.write(T['categorical_desc'])
    # ... (Anda bisa meletakkan kode panduan input di sini jika ada)
    
    st.markdown("---")
    st.subheader(T['data_analysis_title']) 

    df_display = df.copy()
    df_display['HeartDisease_Label'] = df_display['HeartDisease'].map({1: T['class_disease'], 0: T['class_healthy']})

    # Definisi Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        f"📊 {T['tab_target_dist']}",
        f"⏳ {T['tab_age_dist']}",
        f"📈 {T['tab_feature_relation']}",
        f"🔗 {T['tab_correlation']}",
        f"🏆 {T['tab_feature_importance']}"
    ])

    # --- KONTEN UNTUK SETIAP TAB ---

    with tab1:
        st.markdown(f"**{T['target_dist_title']}**")
        st.write(T['target_dist_desc'])
        pie_data = df_display['HeartDisease_Label'].value_counts()
        fig_pie = px.pie(
            values=pie_data.values,
            names=pie_data.index,
            title=T['target_dist_pie_title'],
            color=pie_data.index,
            color_discrete_map={
                T['class_disease']: '#FF4B4B', 
                T['class_healthy']: '#1E88E5'
            }
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    with tab2:
        st.markdown(f"**{T['age_dist_title']}**")
        st.write(T['age_dist_desc'])
        fig_hist = px.histogram(
            df_display, x='Age', color='HeartDisease_Label',
            marginal="box", nbins=30, title=T['age_dist_hist_title'],
            color_discrete_map={
                T['class_disease']: '#FF4B4B',
                T['class_healthy']: '#1E88E5'
            }
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    with tab3:
        st.markdown(f"**{T['feature_relation_title']}**")
        st.write(T['feature_relation_desc'])
        fig_box = px.box(
            df_display, x='HeartDisease_Label', y='Cholesterol', color='HeartDisease_Label',
            title=T['chol_vs_disease_title'],
            color_discrete_map={
                T['class_disease']: '#FF4B4B',
                T['class_healthy']: '#1E88E5'
            }
        )
        st.plotly_chart(fig_box, use_container_width=True)
        fig_bar = px.histogram(
            df_display, x='Sex', color='HeartDisease_Label', barmode='group',
            title=T['sex_vs_disease_title'],
            color_discrete_map={
                T['class_disease']: '#FF4B4B',
                T['class_healthy']: '#1E88E5'
            }
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    with tab4:
        st.markdown(f"**{T['correlation_title']}**")
        st.write(T['correlation_desc'])
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
        corr_matrix = df[numeric_cols].corr()
        fig_heatmap, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", ax=ax, annot_kws={"size": 8})
        ax.set_title(T['correlation_heatmap_title'], fontsize=16)
        st.pyplot(fig_heatmap)
        plt.clf()

    with tab5:
        st.markdown(f"**{T['feature_importance_title']}**")
        st.write(T['feature_importance_desc'])
        if model:
            importances = model.named_steps['classifier'].feature_importances_
            feature_names = model.named_steps['preprocessor'].get_feature_names_out()
            feature_importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': importances
            }).sort_values(by='Importance', ascending=False)
            fig_importance = px.bar(
                feature_importance_df, x='Importance', y='Feature', orientation='h',
                title=T['feature_importance_chart_title'],
                labels={'Importance': T['feature_importance_xaxis'], 'Feature': T['feature_importance_yaxis']}
            )
            fig_importance.update_layout(yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig_importance, use_container_width=True)
        else:
            st.warning("Model tidak berhasil dimuat untuk menampilkan feature importance.")

    st.markdown("<br>", unsafe_allow_html=True)
    st.error(f"**{T['home_disclaimer_title']}**\n{T['home_disclaimer_md']}")

def tampilkan_halaman_prediksi(T, model, explainer, preprocessor):
    # ... (fungsi ini tidak berubah)
    st.info(T['form_intro'])
    st.markdown(f"**{T['risk_key_title']}**")
    col1, col2, col3 = st.columns(3)
    with col1:
        with st.container(border=True): st.markdown(T['risk_key_low_title']); st.write(T['risk_key_low_desc'])
    with col2:
        with st.container(border=True): st.markdown(T['risk_key_medium_title']); st.write(T['risk_key_medium_desc'])
    with col3:
        with st.container(border=True): st.markdown(T['risk_key_high_title']); st.write(T['risk_key_high_desc'])
    st.markdown("<br>", unsafe_allow_html=True)
    with st.container(border=True):
        with st.form("prediction_form"):
            st.header(T['patient_data_header'])
            col1, col2, col3 = st.columns(3)
            with col1:
                age = st.number_input(T['age_label'], min_value=15, max_value=80, value=58, step=1); sex = st.selectbox(T['sex_label'], ['M', 'F']); chest_pain_type = st.selectbox(T['chest_pain_type_label'], ['ATA', 'NAP', 'ASY', 'TA'])
            with col2:
                resting_bp = st.number_input(T['resting_bp_label'], min_value=80, max_value=200, value=130, step=1); cholesterol = st.number_input(T['cholesterol_label'], min_value=100, max_value=600, value=240, step=1); fasting_bs = st.selectbox(T['fasting_bs_label'], [0, 1], format_func=lambda x: "Ya" if x == 1 else "Tidak")
            with col3:
                max_hr = st.number_input(T['max_hr_label'], min_value=70, max_value=205, value=150, step=1); exercise_angina = st.selectbox(T['exercise_angina_label'], ['Y', 'N']); oldpeak = st.number_input(T['oldpeak_label'], min_value=-2.0, max_value=6.5, value=1.0, step=0.1, format="%.1f"); st_slope = st.selectbox(T['st_slope_label'], ['Up', 'Flat', 'Down']); resting_ecg = st.selectbox(T['resting_ecg_label'], ['Normal', 'ST', 'LVH'])
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
                st.markdown(f"**{T['model_confidence_title']}**"); st.markdown(f"**{T['prediction_confidence_label']}:** `{max(prob_sakit, 1-prob_sakit):.1%}`")
                st.markdown(f"**{T['no_disease_prob']}:** `{1-prob_sakit:.1%}`"); st.markdown(f"**{T['disease_prob']}:** `{prob_sakit:.1%}`")
                st.markdown("---"); st.info(T['prob_explanation_low'].format(score=prob_sakit_percent) if risk_class == 'low' else T['prob_explanation_medium'].format(score=prob_sakit_percent) if risk_class == 'medium' else T['prob_explanation_high'].format(score=prob_sakit_percent))
        with st.expander(T['lime_analysis_title']):
            st.markdown(T['lime_explanation']); fig = explanation.as_pyplot_figure(label=1); ax = plt.gca(); bars = ax.patches; exp_list = explanation.as_list(label=1); exp_list.reverse()
            for i, patch in enumerate(bars):
                if i < len(exp_list):
                    weight = exp_list[i][1]
                    if weight > 0: patch.set_color('red')
                    else: patch.set_color('blue')
            ax.set_title(f"{T['lime_plot_title_prefix']} {T['class_disease']}", fontsize=15); st.pyplot(fig, use_container_width=True); plt.clf()

def tampilkan_halaman_about(T):
    # ... (fungsi ini tidak berubah)
    st.markdown(f"### {T['about_what_is_it_title']}")
    with st.container(border=True): st.write(T['about_what_is_it_desc'])
    st.markdown(f"### {T['about_main_features_title']}")
    with st.container(border=True):
        st.markdown(f"- {T['about_feature_1']}"); st.markdown(f"- {T['about_feature_2']}"); st.markdown(f"- {T['about_feature_3']}"); st.markdown(f"- {T['about_feature_4']}")
    st.markdown(f"### {T['about_dataset_info_title']}")
    col1, col2 = st.columns(2)
    with col1:
        with st.container(border=True, height=280):
            st.markdown(f"#### {T['about_datasource_title']}"); st.markdown(f"- {T['about_datasource_1']}"); st.markdown(f"- {T['about_datasource_2']}"); st.markdown(f"- {T['about_datasource_3']}"); st.markdown(f"- {T['about_datasource_4']}"); st.markdown(f"- {T['about_datasource_5']}")
    with col2:
        with st.container(border=True, height=280):
            st.markdown(f"#### {T['about_health_indicators_title']}")
            indicators = T['about_indicators_list']
            sub_col1, sub_col2 = st.columns(2)
            with sub_col1:
                for item in indicators[:6]: st.markdown(f"- {item}")
            with sub_col2:
                for item in indicators[6:]: st.markdown(f"- {item}")
    st.markdown(f"### {T['about_dev_team_title']}")
    with st.container(border=True):
        col1, col2 = st.columns([0.2, 0.7])
        with col1:
            try: st.image("gambar/lutfi.jpg", width=250)
            except Exception: st.warning("Gambar lutfi.jpg tidak ditemukan.")
        with col2:
            st.markdown(f"**Lutfi Julpian**"); st.markdown(f"_{T['about_developer']}_")
            st.markdown("---"); st.markdown(f"**{T['about_advisor_1']}**"); st.markdown(f"**{T['about_advisor_2']}**")

# ==============================================================================
# 3. FUNGSI UTAMA (MAIN) UNTUK MENJALANKAN APLIKASI
# ==============================================================================

def main():
    atur_gaya()
    if 'lang' not in st.session_state: st.session_state.lang = 'id'
    
    T = TRANSLATIONS.get(st.session_state.lang, TRANSLATIONS['id'])
    model, explainer, preprocessor, df = setup_resources()
    selected_page = atur_navigasi(T)
    
    tampilkan_header_banner(T)

    # <<< UBAH PANGGILAN FUNGSI DI BAWAH INI >>>
    if selected_page == T['nav_home']:
        # Teruskan 'model' ke fungsi halaman home
        tampilkan_halaman_home(T, df, model)
    elif selected_page == T['nav_predict']:
        
        tampilkan_halaman_prediksi(T, model, explainer, preprocessor)
        # Teruskan 'model', 'explainer', dan 'preprocessor' ke fungsi
        # ... (tidak ada perubahan di sini)
    elif selected_page == T['nav_about']:
        tampilkan_halaman_about(T)
        # ... (tidak ada perubahan di sini)

if __name__ == "__main__":
    main()