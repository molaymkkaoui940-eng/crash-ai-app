import streamlit as st
import requests
from bs4 import BeautifulSoup
import time
import random
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
import os
import json

# قائمة رؤوس طلب متنوعة لمحاكاة متصفحات متعددة وأساليب تصفح حقيقية
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/15.1 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36",
    "Mozilla/5.0 (iPhone; CPU iPhone OS 15_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/15.0 Mobile/15E148 Safari/604.1",
    "Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Firefox/115.0"
]

# أمثلة على بروكسيات يمكن استبدالها أو توسيعها حسب الحاجة
PROXIES = [
    "http://51.158.68.68:8811",
    "http://34.91.135.46:3128",
    "http://45.77.202.108:8080"
]

def get_random_proxy():
    proxy = random.choice(PROXIES)
    return {'http': proxy, 'https': proxy}

def get_headers():
    headers = {
        'User-Agent': random.choice(USER_AGENTS),
        'Accept-Language': 'en-US,en;q=0.9,ar;q=0.8',
        'Referer': 'https://melbet.com/',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1'
    }
    return headers

def fetch_html(account_id, use_proxy=True):
    url = f"https://melbet.com/player/{account_id}/crash"
    try:
        proxy = get_random_proxy() if use_proxy else None
        response = requests.get(url, headers=get_headers(), proxies=proxy, timeout=15)
        if response.status_code == 200:
            return response.text
        else:
            return None
    except Exception as e:
        return None

def parse_multiplier(html):
    soup = BeautifulSoup(html, 'html.parser')
    elem = soup.find("div", class_="flight-speed")
    if elem:
        try:
            value = elem.text.strip()
            # تأكد من أن النص رقمي ويمكن تحويله
            return float(value)
        except:
            return None
    return None

def create_lstm_model(input_shape):
    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=input_shape),
        Dropout(0.3),
        LSTM(64, return_sequences=True),
        Dropout(0.3),
        LSTM(32),
        Dropout(0.3),
        Dense(50, activation='relu'),
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

def scale_and_prepare(series, scaler=None):
    if scaler is None:
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(series)
    else:
        scaled_data = scaler.transform(series)
    return scaled_data, scaler

def save_data(data_list):
    with open("data.json", "w") as f:
        json.dump(data_list, f)

def load_data():
    if os.path.exists("data.json"):
        with open("data.json", "r") as f:
            return json.load(f)
    return []

def main():
    st.set_page_config(page_title="تتبع انفجارات الطائرة - كراش محسّن", page_icon="✈️")
    st.title("تطبيق تتبع انفجارات الطائرة في لعبة كراش - نسخة متطورة")

    password = st.text_input("أدخل كلمة المرور", type='password')
    if password != "1994":
        if password:
            st.error("كلمة المرور غير صحيحة!")
        return
    st.success("تم التحقق من كلمة المرور!")

    account_id = st.text_input("أدخل رقم الحساب (ID)")
    if not account_id:
        st.warning("يرجى إدخال رقم الحساب.")
        return

    start_button = st.button("ابدأ التتبع")
    stop_button = st.button("أوقف التتبع")

    if 'running' not in st.session_state:
        st.session_state.running = False
    if 'data_list' not in st.session_state:
        st.session_state.data_list = load_data()
    if 'model' not in st.session_state:
        st.session_state.model = None
    if 'scaler' not in st.session_state:
        st.session_state.scaler = None

    if start_button:
        st.session_state.running = True
    if stop_button:
        st.session_state.running = False

    placeholder_current = st.empty()
    placeholder_pred = st.empty()

    n_steps = 10
    early_stopping = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)

    while st.session_state.running:
        html = fetch_html(account_id, use_proxy=True)
        if html:
            current_mul = parse_multiplier(html)
            if current_mul is not None:
                # تجاهل القيم غير المنطقية أو الشاذة لتقوية جودة البيانات
                if current_mul > 0 and current_mul < 100:
                    st.session_state.data_list.append(current_mul)
                    # حفظ البيانات بشكل دوري
                    save_data(st.session_state.data_list)
                    placeholder_current.markdown(f"**مضاعف الجولة الحالية:** {current_mul:.2f}x")

                else:
                    placeholder_current.warning("تم تجاهل قيمة خارجة عن النطاق المتوقع.")
            else:
                placeholder_current.error("فشل قراءة بيانات الجولة الحالية.")
                time.sleep(random.randint(20, 40))
                continue
        else:
            st.error("فشل في جلب الصفحة.")
            time.sleep(random.randint(20, 40))
            continue

        if len(st.session_state.data_list) > n_steps:
            series = np.array(st.session_state.data_list).reshape(-1, 1)

            if st.session_state.scaler is None:
                scaled_data, scaler = scale_and_prepare(series)
                st.session_state.scaler = scaler
            else:
                scaled_data, _ = scale_and_prepare(series, st.session_state.scaler)

            X, y = prepare_data(scaled_data, n_steps)
            X = X.reshape((X.shape[0], X.shape[1], 1))

            if st.session_state.model is None:
                st.session_state.model = create_lstm_model((X.shape[1], 1))

            st.session_state.model.fit(X, y, epochs=20, batch_size=8, verbose=0, callbacks=[early_stopping])

            x_input = scaled_data[-n_steps:].reshape(1, n_steps, 1)
            pred_scaled = st.session_state.model.predict(x_input, verbose=0)
            pred = st.session_state.scaler.inverse_transform(pred_scaled)[0][0]

            placeholder_pred.markdown(f"**التوقع لآخر انفجار محتمل:** {pred:.2f}x")
        else:
            placeholder_pred.markdown("... جاري جمع بيانات كافية للتدريب")

        # تأخير عشوائي بين 15 إلى 60 ثانية لمحاكاة استخدام بشري
        time.sleep(random.randint(15, 60))

    if not st.session_state.running:
        st.info("تم إيقاف التتبع.")


if __name__ == "__main__":
    main()
