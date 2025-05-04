import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# Judul aplikasi
st.title("Aplikasi Web Interaktif dengan Streamlit")
st.subheader("Selamat Datang!")

# Input nama pengguna
nama = st.text_input("Masukkan nama Anda:", "Tulis nama di sini")
if nama != "Tulis nama di sini":
    st.write(f"Halo, {nama}! Selamat menikmati aplikasi ini.")

# Pilihan visualisasi
st.sidebar.header("Pilih Visualisasi")
visualisasi = st.sidebar.selectbox(
    "Pilih jenis grafik:",
    ["Line Chart", "Bar Chart", "Scatter Plot"]
)

# Buat data dummy
data = pd.DataFrame({
    'X': np.arange(10),
    'Y': np.random.randn(10) * 10
})

# Tampilkan grafik berdasarkan pilihan
if visualisasi == "Line Chart":
    fig = px.line(data, x='X', y='Y', title="Grafik Garis")
    st.plotly_chart(fig)
elif visualisasi == "Bar Chart":
    fig = px.bar(data, x='X', y='Y', title="Grafik Batang")
    st.plotly_chart(fig)
elif visualisasi == "Scatter Plot":
    fig = px.scatter(data, x='X', y='Y', title="Grafik Sebar")
    st.plotly_chart(fig)

# Tambahkan footer
st.markdown("Dibuat dengan ❤️ menggunakan Streamlit")