import streamlit as st
import pandas as pd
import pickle

# 📦 Modeli yükle
with open("airbnb_model.pkl", "rb") as f:
    model = pickle.load(f)

st.title("Airbnb Fiyat Tahmini")

# 📋 Kullanıcıdan veri al
room_type = st.sidebar.selectbox("Oda Tipi", ["Private room", "Entire home/apt", "Shared room"])
neighbourhood_group = st.sidebar.selectbox("Bölge", ["Manhattan", "Brooklyn", "Queens", "Staten Island", "Bronx"])
minimum_nights = st.sidebar.number_input("Minimum Gece Sayısı", 1, 100, 2)
availability_365 = st.sidebar.slider("Yıl Boyu Müsaitlik", 0, 365, 150)
number_of_reviews = st.sidebar.slider("Yorum Sayısı", 0, 500, 10)

# 🎯 Doğru formatta input_dict oluştur
input_dict = {
    "minimum_nights": minimum_nights,
    "availability_365": availability_365,
    "number_of_reviews": number_of_reviews,
    "room_type_Private room": 1 if room_type == "Private room" else 0,
    "room_type_Shared room": 1 if room_type == "Shared room" else 0,
    "neighbourhood_group_Brooklyn": 1 if neighbourhood_group == "Brooklyn" else 0,
    "neighbourhood_group_Manhattan": 1 if neighbourhood_group == "Manhattan" else 0,
    "neighbourhood_group_Queens": 1 if neighbourhood_group == "Queens" else 0,
    "neighbourhood_group_Staten Island": 1 if neighbourhood_group == "Staten Island" else 0
}

# 🧮 Tahmin için DataFrame oluştur
input_df_encoded = pd.DataFrame([input_dict])

# Tahmini hesapla
if st.button("Tahmini Fiyatı Göster"):
    prediction = model.predict(input_df_encoded)[0]
    st.success(f"Tahmini gecelik fiyat: ${prediction:.2f}")
