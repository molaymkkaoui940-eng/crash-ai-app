import streamlit as st
import asyncio
from playwright.async_api import async_playwright
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, Multiply, Permute, Activation, RepeatVector, Lambda
from sklearn.preprocessing import MinMaxScaler
import json
import time
import random

PASSWORD = "1994"

# آلية انتباه Attention Layer
def attention_3d_block(inputs):
    # inputs.shape = (batch_size, time_steps, input_dim)
    input_dim = int(inputs.shape[2])
    a = Permute((2, 1))(inputs)
    a = Dense(time_steps, activation='softmax')(a)
    a_probs = Permute((2, 1))(a)
    output_attention_mul = Multiply()([inputs, a_probs])
    return output_attention_mul

time_steps = 10  # خطوات الزمنية لبيانات المدخلات

def create_attention_lstm_model(input_shape):
    inputs = Input(shape=input_shape)
    lstm_out = LSTM(128, return_sequences=True)(inputs)
    lstm_out = Dropout(0.3)(lstm_out)

    attention_mul = attention_3d_block(lstm_out)

    lstm_out2 = LSTM(64)(attention_mul)
    lstm_out2 = Dropout(0.3)(lstm_out2)

    dense1 = Dense(50, activation='relu')(lstm_out2)
    outputs = Dense(1)(dense1)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# تجهيز البيانات
def prepare_data(data, n_steps=time_steps):
    X, y = [], []
    for i in range(n_steps, len(data)):
        X.append(data[i-n_steps:i])
        y.append(data[i])
    return np.array(X), np.array(y)

# التقاط بيانات WebSocket الحية من 1XBet عبر Playwright
async def capture_crash_data(account_id, data_container, stop_flag):
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        url = f"https://1xbet.com/player/{account_id}/crash"
        await page.goto(url)

        def ws_handler(ws):
            async def on_message(msg):
                try:
                    text = msg['text']
                    data = json.loads(text)
                    if "crashPoint" in data:
                        val = float(data["crashPoint"])
                        if 0 < val < 100:
                            data_container.append(val)
                except:
                    pass
            ws.on("framereceived", on_message)
        
        page.on("websocket", ws_handler)

        while not stop_flag['stop']:
            await asyncio.sleep(1)
        await browser.close()

def run_async_capture(account_id, data_container, stop_flag):
    asyncio.run(capture_crash_data(account_id, data_container, stop_flag))

# Streamlit interface
def main():
    st.title("تتبع وتوقع انفجارات الطائرة - 1XBET Crash مع تحسينات")

    password = st.text_input("أدخل كلمة المرور:", type="password")
    if password != PASSWORD:
        if password:
            st.error("كلمة المرور خاطئة!")
        return

    account_id = st.text_input("أدخل رقم الحساب (ID):")
    if not account_id:
        st.warning("يرجى إدخال رقم الحساب.")
        return

    if 'running' not in st.session_state:
        st.session_state.running = False
    if 'data' not in st.session_state:
        st.session_state.data = []
    if 'scaler' not in st.session_state:
        st.session_state.scaler = None
    if 'model' not in st.session_state:
        st.session_state.model = None
    if 'stop_flag' not in st.session_state:
        st.session_state.stop_flag = {'stop': False}

    col1, col2 = st.columns(2)
    with col1:
        if st.button("ابدأ التتبع"):
            st.session_state.running = True
            st.session_state.stop_flag['stop'] = False
            st.session_state.data.clear()
            st.experimental_rerun()

    with col2:
        if st.button("أوقف التتبع"):
            st.session_state.running = False
            st.session_state.stop_flag['stop'] = True

    status_text = st.empty()
    pred_text = st.empty()

    if st.session_state.running:
        status_text.text(f"جارِ جمع البيانات... الجولات المسجلة: {len(st.session_state.data)}")
        
        # بدء الالتقاط بشكل غير متزامن في الخلفية
        if 'capture_task' not in st.session_state or st.session_state.capture_task.done():
            st.session_state.capture_task = asyncio.ensure_future(
                capture_crash_data(account_id, st.session_state.data, st.session_state.stop_flag)
            )

        # بعد جمع بيانات كافية ... تدريب النموذج والتنبؤ
        if len(st.session_state.data) > time_steps + 5:
            series = np.array(st.session_state.data).reshape(-1, 1)
            if st.session_state.scaler is None:
                scaler = MinMaxScaler(feature_range=(0, 1))
                scaled = scaler.fit_transform(series)
                st.session_state.scaler = scaler
            else:
                scaled = st.session_state.scaler.transform(series)

            X, y = prepare_data(scaled.flatten(), time_steps)
            X = X.reshape((X.shape[0], X.shape[1], 1))

            if st.session_state.model is None:
                st.session_state.model = create_attention_lstm_model((time_steps, 1))

            st.session_state.model.fit(X, y, epochs=20, batch_size=16, verbose=0)

            x_input = scaled[-time_steps:].reshape((1, time_steps, 1))
            pred_scaled = st.session_state.model.predict(x_input, verbose=0)
            pred = st.session_state.scaler.inverse_transform(pred_scaled.reshape(-1, 1))[0][0]

            pred_text.markdown(f"**التوقع للمضاعف القادم:** {pred:.2f}x")
        else:
            pred_text.text("... جارٍ جمع بيانات كافية للتنبؤ (10 جولات على الأقل)")

    else:
        status_text.text("تم إيقاف التتبع.")
        pred_text.text("")

if __name__ == "__main__":
    main()
