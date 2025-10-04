# app.py
import streamlit as st
import requests
import pandas as pd
import numpy as np
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from datetime import datetime
import joblib
import os
# كلمة مرور بسيطة لحماية التطبيق
password = "1994"  # كلمة المرور الخاصة بك
input_pass = st.text_input("🔒 أدخل كلمة المرور:", type="password")
if input_pass != password:
    st.error("كلمة المرور غير صحيحة ❌")
    st.stop()
# ---------------------------
# إعداد صفحة التطبيق
# ---------------------------
st.set_page_config(page_title="Crash Predictor 🚀", page_icon="🤖")

st.title("🚀 تطبيق يتنبأ بانفجارات الطائرة")
st.write("أدخل بيانات الطائرة والتفاصيل للحصول على التوقعات:")

# ---------------------------
# إدخال البيانات من المستخدم
# ---------------------------
col1, col2 = st.columns(2)

with col1:
    speed = st.number_input("سرعة الطائرة (كم/س)", min_value=100, max_value=2000, value=500)
    altitude = st.number_input("الارتفاع (قدم)", min_value=1000, max_value=50000, value=30000)
    temperature = st.number_input("درجة الحرارة (°C)", min_value=-50, max_value=60, value=25)

with col2:
    fuel = st.number_input("كمية الوقود (لتر)", min_value=100, max_value=200000, value=50000)
    engine_hours = st.number_input("عدد ساعات تشغيل المحرك", min_value=1, max_value=100000, value=2000)
    age = st.number_input("عمر الطائرة (سنوات)", min_value=1, max_value=80, value=10)

# ---------------------------
# تحميل أو إنشاء نموذج تدريبي
# ---------------------------
MODEL_FILE = "model.pkl"

if os.path.exists(MODEL_FILE):
    model = joblib.load(MODEL_FILE)
else:
    # بيانات افتراضية للتدريب (بدون قاعدة بيانات حقيقية)
    X = pd.DataFrame({
        "speed": np.random.randint(200, 1000, 500),
        "altitude": np.random.randint(5000, 40000, 500),
        "temperature": np.random.randint(-40, 50, 500),
        "fuel": np.random.randint(5000, 150000, 500),
        "engine_hours": np.random.randint(500, 50000, 500),
        "age": np.random.randint(1, 40, 500),
    })

    y = np.random.randint(0, 2, 500)  # 0 = آمن، 1 = خطر

    model = RandomForestClassifier()
    model.fit(X, y)

    joblib.dump(model, MODEL_FILE)

# ---------------------------
# معالجة البيانات والتنبؤ
# ---------------------------
if st.button("🔮 تنبؤ"):
    input_data = pd.DataFrame([{
        "speed": speed,
        "altitude": altitude,
        "temperature": temperature,
        "fuel": fuel,
        "engine_hours": engine_hours,
        "age": age
    }])

    prediction = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0][prediction] * 100

    if prediction == 1:
        st.error(f"⚠️ هناك خطر محتمل! (نسبة الخطورة: {prob:.2f}%)")
    else:
    
        st.success(f"✅ الوضع آمن (نسبة الأمان: {prob:.2f}%)")
