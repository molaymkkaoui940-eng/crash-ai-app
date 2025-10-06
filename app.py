import streamlit as st
import requests
from bs4 import BeautifulSoup
import time
import random
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler

# عناوين User-Agent عشوائية لتقليل الكشف
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/15.1 Safari/605.1.15"
]

def get_headers():
    return {'User-Agent': random.choice(USER_AGENTS)}

def fetch_html(account_id):
    url = f"https://melbet.com/player/{account_id}/crash"  # تأكد من صحة الرابط
    try:
        response = requests.get(url, headers=get_headers(), timeout=10)
        if response.status_code == 200:
            return response.text
        else:
            st.warning(f"لم يتم جلب البيانات. رمز الخطأ: {response.status_code}")
            return None
    except Exception as e:
        st.error(f"خطأ في الاتصال بالموقع: {e}")
        return None

def parse_multiplier(html):
    soup = BeautifulSoup(html, 'html.parser')
    elem = soup.find("div", class_="flight-speed")  # قم بتعديل العنصر حسب الصفحة الحقيقية
    if elem:
        try:
            return float(elem.text.strip())
        except:
            return None
    return None

def create_lstm_model(input_shape):
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(25),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def prepare_data(data, n_steps=10):
    X, y = [], []
    for i in range(n_steps, len(data)):
        X.append(data[i-n_steps:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

def scale_and_prepare(series):
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler.fit_transform(series)
    return scaled_data, scaler

def main():
    st.set_page_config(page_title="تطبيق ذكي تتبع انفجارات الطائرة - كراش", page_icon="✈️", layout='centered')

    st.markdown("""
    <style>
    .main {background-color: #1e1e2e; color: white;}
    .block-container {padding: 2rem 5rem;}
    h1 {color: #00bfff;}
    .stButton>button {background-color: #00bfff; color: white;}
    .stTextInput>div>div>input {background-color: #2e2e3e; color: white;}
    </style>
    """, unsafe_allow_html=True)

    st.title("✈️ تطبيق تتبع انفجارات الطائرة - لعبة كراش")

    password = st.text_input("أدخل كلمة المرور", type='password')
    if password != "1994":
        if password:
            st.error("كلمة المرور غير صحيحة!")
        return
    st.success("تم التحقق من كلمة المرور!")

    account_id = st.text_input("أدخل رقم الحساب (ID)")

    if not account_id:
        st.warning("يرجى إدخال رقم الحساب")
        return

    data_list = []
    n_steps = 10
    model = None
    scaler = None

    placeholder_current = st.empty()
    placeholder_pred = st.empty()
    stop_button = st.button("أوقف التتبع")
    stop_flag = False

    while not stop_flag:
        if stop_button:
            st.warning("تم إيقاف التطبيق بناءً على طلبك.")
            stop_flag = True
            break

        html = fetch_html(account_id)
        if html:
            current_mul = parse_multiplier(html)
            if current_mul is not None:
                data_list.append(current_mul)
                placeholder_current.markdown(f"### مضاعف الجولة الحالية: {current_mul:.2f}x")

                if len(data_list) > n_steps:
                    series = np.array(data_list).reshape(-1,1)
                    scaled_data, scaler = scale_and_prepare(series)

                    X, y = prepare_data(scaled_data, n_steps)
                    X = X.reshape((X.shape[0], X.shape[1], 1))

                    if model is None:
                        model = create_lstm_model((X.shape[1], 1))
                    
                    model.fit(X, y, epochs=5, batch_size=1, verbose=0)

                    x_input = scaled_data[-n_steps:]
                    x_input = x_input.reshape((1, n_steps, 1))
                    pred_scaled = model.predict(x_input, verbose=0)
                    pred = scaler.inverse_transform(pred_scaled)[0][0]

                    placeholder_pred.markdown(f"### التوقع للجولة القادمة: {pred:.2f}x")
                else:
                    placeholder_pred.markdown("جاري جمع بيانات كافية للتدريب...")
            else:
                placeholder_current.error("فشل في قراءة بيانات الجولة الحالية.")
                break
        else:
            st.error("فشل في جلب الصفحة")
            break

        # تأخير عشوائي لتقليل فرص الكشف والحظر
        time.sleep(random.randint(8, 15))

if __name__ == "__main__":
    main()
