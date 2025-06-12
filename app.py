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

# Konfigurasi halaman utama Streamlit (Perintah Streamlit pertama)
st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Kamus terjemahan LENGKAP dan FINAL
TRANSLATIONS = {
    'id': {
        'nav_home': "🏠 Home", 'nav_predict': "🔍 Prediksi", 'nav_about': "ℹ️ Tentang",
        'nav_title': "📌 Navigasi", 'lang_select_title': "Pilih Bahasa",
        'home_title': "🫀 Aplikasi Prediksi Penyakit Jantung",
        'home_intro': "Aplikasi ini menggunakan model Machine Learning yang dilatih dengan SMOTE untuk mengatasi ketidakseimbangan data dan GridSearchCV untuk menemukan hyperparameter terbaik. Tujuannya adalah untuk memberikan prediksi risiko penyakit jantung berdasarkan data klinis pasien.",
        'home_disclaimer_title': "⚠️ Disclaimer Penting",
        'home_disclaimer_md': "Aplikasi ini adalah prototipe untuk tujuan edukasi dan **tidak boleh digunakan untuk diagnosis medis nyata**. Hasil prediksi tidak menggantikan konsultasi dengan tenaga medis profesional.",
        'predict_title': "🔍 Prediksi & Analisis Risiko Penyak-it Jantung",
        'model_load_error': "❌ Gagal memuat model/data. Pastikan file 'model_pipeline_terbaik.pkl' dan 'heart.csv' tersedia.",
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
        'probability_breakdown': "Rincian Probabilitas", 'risk_distribution': "Distribusi Probabilitas Risiko",
        'no_disease': "Tidak Sakit Jantung", 'disease': "Sakit Jantung",
        'model_confidence_header': "Kepercayaan Diri Model", 'prediction_confidence': "Tingkat Kepercayaan Prediksi",
        'no_disease_prob': "Probabilitas Tidak Sakit Jantung", 'disease_prob': "Probabilitas Sakit Jantung",
        'lime_analysis_title': "Penjelasan Faktor Risiko (Analisis LIME)",
        'lime_explanation': "Grafik di bawah ini menunjukkan faktor-faktor yang paling berpengaruh pada prediksi untuk pasien ini. Faktor dengan bar hijau mendukung prediksi 'Sakit Jantung', sedangkan bar merah menentangnya.",
        'lime_plot_title_prefix': "Penjelasan lokal untuk kelas",
        'class_disease': "Penyakit", 'patient_risk_status': "Status Risiko Pasien",
        'about_title': "ℹ️ Tentang Aplikasi Ini",
        'about_body': "Aplikasi ini dibuat sebagai bagian dari studi kasus dalam penerapan Machine Learning untuk prediksi penyakit jantung.",
        'about_developer': "Pengembang",
        'about_technology': "Teknologi yang Digunakan"
    },
    'en': {
        'nav_home': "🏠 Home", 'nav_predict': "🔍 Prediction", 'nav_about': "ℹ️ About",
        'nav_title': "📌 Navigation", 'lang_select_title': "Select Language",
        'home_title': "🫀 Heart Disease Prediction App",
        'home_intro': "This application uses a Machine Learning model trained with SMOTE to handle data imbalance and GridSearchCV to find the best hyperparameters. Its goal is to provide heart disease risk predictions based on patient clinical data.",
        'home_disclaimer_title': "⚠️ Important Disclaimer",
        'home_disclaimer_md': "This application is a prototype for educational purposes and **must not be used for real medical diagnosis**. The prediction results do not replace consultation with a professional healthcare provider.",
        'predict_title': "🔍 Predict & Analyze Heart Disease Risk",
        'model_load_error': "❌ Failed to load model/data. Ensure 'model_pipeline_terbaik.pkl' and 'heart.csv' are available.",
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
        'probability_breakdown': "Probability Breakdown", 'risk_distribution': "Risk Probability Distribution",
        'no_disease': "No Heart Disease", 'disease': "Heart Disease",
        'model_confidence_header': "Model Confidence", 'prediction_confidence': "Prediction Confidence Level",
        'no_disease_prob': "Probability of No Heart Disease", 'disease_prob': "Probability of Heart Disease",
        'lime_analysis_title': "Risk Factor Explanation (LIME Analysis)",
        'lime_explanation': "The plot below shows the factors that most influenced this prediction. Features with green bars support the 'Heart Disease' prediction, while red bars oppose it.",
        'lime_plot_title_prefix': "Local explanation for class",
        'class_disease': "Disease", 'patient_risk_status': "Patient's Risk Status",
        'about_title': "ℹ️ About This Application",
        'about_body': "This application was created as part of a case study in implementing Machine Learning for heart disease prediction.",
        'about_developer': "Developer",
        'about_technology': "Technology Used"
    }
}

def atur_gaya():
    st.markdown("""<style>.stApp { background-color: #f0f2f6; } .stButton button { width: 100%; border-radius: 8px; background-color: #4CAF50; color: white; } .stSpinner > div > div { border-top-color: #4CAF50; }</style>""", unsafe_allow_html=True)

# ==============================================================================
# 2. FUNGSI-FUNGSI UTAMA (LOGIKA & HALAMAN)
# ==============================================================================

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
        return model, explainer, preprocessor
    except FileNotFoundError:
        return None, None, None

def create_gauge_chart(probability_score, T):
    fig = go.Figure(go.Indicator(
        mode = "gauge+number", value = probability_score,
        title = {'text': T['risk_level']},
        gauge = {'axis': {'range': [None, 100]}, 'bar': {'color': "#2E3B4E"},
                 'steps' : [{'range': [0, 25], 'color': "green"}, {'range': [25, 50], 'color': "yellow"}, {'range': [50, 100], 'color': "red"}]}))
    fig.update_layout(height=250, margin=dict(l=10, r=10, t=40, b=10))
    return fig

def create_pie_chart(probability_score, T):
    labels = [T['no_disease'], T['disease']]
    values = [100 - probability_score, probability_score]
    fig = px.pie(values=values, names=labels, color=labels,
                 color_discrete_map={T['no_disease']:'green', T['disease']:'red'}, hole=.3)
    fig.update_layout(title_text=T['risk_distribution'], showlegend=True, height=300, margin=dict(l=10, r=10, t=40, b=10))
    return fig

def atur_navigasi(T):
    with st.sidebar:
        st.image("https://img.icons8.com/?size=100&id=35583&format=png", width=100)
        st.markdown(f"### {T['lang_select_title']}")
        selected_lang_display = option_menu(menu_title=None, options=["Indonesia", "English"], icons=["globe2", "globe-americas"], menu_icon="translate", default_index=0 if st.session_state.lang == 'id' else 1, orientation="horizontal", key="lang_menu", styles={"container": {"padding": "0!important", "background-color": "#e0e0e0", "border-radius": "8px"}, "nav-link-selected": {"background-color": "#2c3e50", "color": "white"}})
        new_lang = 'id' if selected_lang_display == 'Indonesia' else 'en'
        if st.session_state.lang != new_lang:
            st.session_state.lang = new_lang
            st.rerun()
        st.markdown(f"## {T['nav_title']}")
        selected_page = option_menu(menu_title=None, options=[T['nav_home'], T['nav_predict'], T['nav_about']], icons=["house-heart-fill", "search-heart", "info-circle-fill"], menu_icon="cast", default_index=0, key="main_menu", styles={"container": {"padding": "5px", "background-color": "#fafafa"}, "nav-link-selected": {"background-color": "#4CAF50"}})
    return selected_page

def tampilkan_halaman_home(T):
    st.title(T['home_title'])
    st.markdown("---")
    st.markdown(T['home_intro'])
    st.subheader("Informasi Standar Input Fitur")
    st.code("""- Usia (Age): Rentang 28 - 77 tahun\n- Tekanan Darah Istirahat (RestingBP): Rentang 80 - 200 mm Hg\n- Kolesterol (Cholesterol): Rentang data dalam dataset: 0–603 mg/dl \n  ⚠️ Nilai normal klinis: < 200 mg/dl\n- Detak Jantung Maksimum (MaxHR): Rentang 60 - 202 denyut per menit\n- Oldpeak: Rentang -2.6 - 6.2\n- Fitur Kategorikal lainnya memiliki pilihan yang sudah disediakan di form.""", language='markdown')
    st.subheader("Sumber Dataset")
    st.markdown("""Dataset yang digunakan untuk melatih model ini berasal dari platform **Kaggle**. \nSecara spesifik, dataset ini berjudul "Heart Failure Prediction Dataset" yang dikompilasi oleh Fedesoriano dan tersedia di bawah lisensi **Open Data Commons Open Database License (ODbL)**.""")
    st.error(f"**{T['home_disclaimer_title']}**\n{T['home_disclaimer_md']}")

def tampilkan_halaman_prediksi(T, model, explainer, preprocessor):
    st.title(T['predict_title'])
    st.info(T['form_intro'])
    with st.form("prediction_form"):
        st.header(T['patient_data_header'])
        col1, col2, col3 = st.columns(3)
        with col1:
            age = st.number_input(T['age_label'], min_value=15, max_value=80, value=58, step=1)
            sex = st.selectbox(T['sex_label'], ['M', 'F'])
            chest_pain_type = st.selectbox(T['chest_pain_type_label'], ['ATA', 'NAP', 'ASY', 'TA'])
            resting_bp = st.number_input(T['resting_bp_label'], min_value=50, max_value=250, value=146, step=1)
        with col2:
            cholesterol = st.number_input(T['cholesterol_label'], min_value=100, max_value=300, value=200, step=1)
            fasting_bs = st.selectbox(T['fasting_bs_label'], [0, 1], format_func=lambda x: "Ya" if x == 1 else "Tidak")
            resting_ecg = st.selectbox(T['resting_ecg_label'], ['Normal', 'ST', 'LVH'])
            max_hr = st.number_input(T['max_hr_label'], min_value=50, max_value=220, value=105, step=1)
        with col3:
            exercise_angina = st.selectbox(T['exercise_angina_label'], ['Y', 'N'])
            oldpeak = st.number_input(T['oldpeak_label'], min_value=-3.0, max_value=7.0, value=2.0, step=0.1, format="%.1f")
            st_slope = st.selectbox(T['st_slope_label'], ['Up', 'Flat', 'Down'])
        submitted = st.form_submit_button(T['predict_button'], use_container_width=True)

    if submitted:
        with st.spinner(T['spinner_text']):
            input_data = pd.DataFrame({'Age': [age], 'Sex': [sex], 'ChestPainType': [chest_pain_type], 'RestingBP': [resting_bp], 'Cholesterol': [cholesterol], 'FastingBS': [fasting_bs], 'RestingECG': [resting_ecg], 'MaxHR': [max_hr], 'ExerciseAngina': [exercise_angina], 'Oldpeak': [oldpeak], 'ST_Slope': [st_slope]})
            probabilitas = model.predict_proba(input_data)
            prob_sakit = probabilitas[0][1]
            input_processed = preprocessor.transform(input_data)
            explanation = explainer.explain_instance(input_processed[0], model.named_steps['classifier'].predict_proba, num_features=11, labels=(1,))

        st.header(T['result_header'])
        prob_sakit_percent = prob_sakit * 100
        if prob_sakit_percent >= 50:
            risk_text = T['result_high_risk']; risk_color = "#FF4B4B"; recommendation_text = T['recommendation_high']
        elif prob_sakit_percent >= 25:
            risk_text = T['result_medium_risk']; risk_color = "#FFD700"; recommendation_text = T['recommendation_medium']
        else:
            risk_text = T['result_low_risk']; risk_color = "#28A745"; recommendation_text = T['recommendation_low']
        
        col1, col2 = st.columns([0.65, 0.35])
        with col1:
            st.markdown(f"""<div style="background-color: {risk_color}20; padding: 20px; border-radius: 10px; border-left: 7px solid {risk_color};"><h2 style='color: {risk_color}; margin: 0;'>{risk_text}</h2><h4 style='color: #333; margin: 0;'>{T['risk_score']}: {prob_sakit_percent:.1f}%</h4><p style='color: #555; margin-top: 15px;'><b>{T['recommendation']}:</b> {recommendation_text}</p></div>""", unsafe_allow_html=True)
        with col2:
            st.plotly_chart(create_gauge_chart(prob_sakit_percent, T), use_container_width=True)
        st.markdown("<br>", unsafe_allow_html=True)
        st.subheader(T['probability_breakdown'])
        col1, col2 = st.columns([0.4, 0.6])
        with col1:
            st.plotly_chart(create_pie_chart(prob_sakit_percent, T), use_container_width=True)
        with col2:
            st.markdown(f"<h5>{T['model_confidence_header']}</h5>", unsafe_allow_html=True)
            st.info(f"**{T['prediction_confidence']}:** {max(prob_sakit, 1-prob_sakit):.1%}")
            st.markdown(f"- **{T['no_disease_prob']}:** {1-prob_sakit:.1%}")
            st.markdown(f"- **{T['disease_prob']}:** {prob_sakit:.1%}")
        
        with st.expander(T['lime_analysis_title']):
            st.markdown(T['lime_explanation'])
            fig = explanation.as_pyplot_figure(label=1)
            ax = plt.gca()
            title_text = f"{T['lime_plot_title_prefix']} {T['class_disease']}"
            ax.set_title(title_text, fontsize=15)
            st.pyplot(fig, use_container_width=True)
            plt.clf()

def tampilkan_halaman_about(T):
    st.title(T['about_title'])
    st.markdown("---")
    st.markdown(f"**{T['about_developer']}** Lutfi Julpian")
    st.markdown(f"**{T['about_technology']}**: Python, Streamlit, Scikit-learn, Imblearn, Pandas, LIME, Plotly")
    st.markdown(T['about_body'])

# ==============================================================================
# 3. FUNGSI UTAMA (MAIN) UNTUK MENJALANKAN APLIKASI
# ==============================================================================

def main():
    atur_gaya()
    if 'lang' not in st.session_state:
        st.session_state.lang = 'id'
    T = TRANSLATIONS[st.session_state.lang]
    model, explainer, preprocessor = setup_resources()
    selected_page = atur_navigasi(T)
    if selected_page == T['nav_home']:
        tampilkan_halaman_home(T)
    elif selected_page == T['nav_predict']:
        if model is None:
            st.error(T['model_load_error'])
        else:
            tampilkan_halaman_prediksi(T, model, explainer, preprocessor)
    elif selected_page == T['nav_about']:
        tampilkan_halaman_about(T)

if __name__ == "__main__":
    main()