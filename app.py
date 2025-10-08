# app.py  (محسّن: Playwright fallback + direct WebSocket + feature engineering + LSTM+Attention + robust training)
import streamlit as st
import threading
import asyncio
import time
import json
import os
import pickle
from collections import deque
from datetime import datetime

# data science
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# tensorflow keras
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, Multiply, Permute, Concatenate
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam

# Playwright
from playwright.async_api import async_playwright

# optional direct ws library (async)
import websockets

# -----------------------
# Config
# -----------------------
PASSWORD = "1994"
TIME_STEPS = 10
MODEL_PATH = "crash_model.h5"
SCALER_PATH = "scaler.pkl"
META_SCALER_PATH = "meta_scaler.pkl"
DATA_CSV = "collected_crash_data.csv"
MIN_SAMPLES_TO_TRAIN = TIME_STEPS + 100  # recommended
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # reduce TF logs

# -----------------------
# Utilities: CSV store / load
# -----------------------
def append_to_csv(value):
    ts = datetime.utcnow().isoformat()
    df = pd.DataFrame([{"timestamp": ts, "value": float(value)}])
    if not os.path.exists(DATA_CSV):
        df.to_csv(DATA_CSV, index=False)
    else:
        df.to_csv(DATA_CSV, mode='a', header=False, index=False)

def load_history():
    if os.path.exists(DATA_CSV):
        return pd.read_csv(DATA_CSV)
    else:
        return pd.DataFrame(columns=["timestamp", "value"])

# -----------------------
# Feature engineering (meta features for each window)
# -----------------------
def compute_meta_features(window):
    # window: 1D numpy array of length TIME_STEPS
    arr = np.array(window).astype(float)
    mean = arr.mean()
    std = arr.std(ddof=0)
    median = np.median(arr)
    mx = arr.max()
    mn = arr.min()
    last = arr[-1]
    slope = last - arr[0]
    prop_gt_2 = float((arr > 2.0).sum()) / len(arr)  # proportion of multipliers > 2
    # return vector as floats
    return np.array([mean, std, median, mx, mn, last, slope, prop_gt_2], dtype=float)

# -----------------------
# Prepare dataset (X_seq, X_meta, y)
# -----------------------
def prepare_dataset(series_raw, n_steps=TIME_STEPS):
    # series_raw: 1D numpy raw values
    series = np.array(series_raw).astype(float)
    if len(series) <= n_steps:
        return None, None, None
    X_seq = []
    X_meta = []
    y = []
    for i in range(n_steps, len(series)):
        window = series[i-n_steps:i]
        X_seq.append(window.reshape(n_steps, 1))
        X_meta.append(compute_meta_features(window))
        y.append(series[i])
    return np.array(X_seq), np.array(X_meta), np.array(y)

# -----------------------
# Attention helper
# -----------------------
def attention_3d_block(inputs, time_steps):
    # inputs shape (batch, time_steps, features)
    a = Permute((2,1))(inputs)
    a = Dense(time_steps, activation='softmax')(a)
    a_probs = Permute((2,1))(a)
    output_attention_mul = Multiply()([inputs, a_probs])
    return output_attention_mul

# -----------------------
# Model builder (seq + meta)
# -----------------------
def build_model(n_steps=TIME_STEPS, meta_dim=8):
    seq_input = Input(shape=(n_steps, 1), name="seq_input")
    meta_input = Input(shape=(meta_dim,), name="meta_input")

    x = LSTM(128, return_sequences=True)(seq_input)
    x = Dropout(0.2)(x)
    att = attention_3d_block(x, n_steps)
    x = LSTM(64)(att)
    x = Dropout(0.2)(x)

    combined = Concatenate()([x, meta_input])
    dense = Dense(64, activation='relu')(combined)
    dense = Dropout(0.2)(dense)
    dense = Dense(32, activation='relu')(dense)
    out = Dense(1, name="output")(dense)

    model = Model(inputs=[seq_input, meta_input], outputs=out)
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
    return model

# -----------------------
# Train & save model & scalers
# -----------------------
def train_and_save(series_raw, epochs=100, n_steps=TIME_STEPS):
    X_seq, X_meta, y = prepare_dataset(series_raw, n_steps=n_steps)
    if X_seq is None:
        raise ValueError("Not enough data to prepare dataset.")
    # scale series (fit on all values)
    series_vals = np.array(series_raw).reshape(-1,1)
    series_scaler = MinMaxScaler(feature_range=(0,1))
    scaled_all = series_scaler.fit_transform(series_vals).flatten()

    # we need to recompute X_seq_scaled and X_meta_scaled from scaled_all
    X_seq_scaled = []
    X_meta = []
    y_scaled = []
    for i in range(n_steps, len(scaled_all)):
        window = scaled_all[i-n_steps:i]
        X_seq_scaled.append(window.reshape(n_steps,1))
        # compute meta features from scaled window (consistent)
        meta = compute_meta_features(window)
        X_meta.append(meta)
        y_scaled.append(scaled_all[i])
    X_seq_scaled = np.array(X_seq_scaled)
    X_meta = np.array(X_meta)
    y_scaled = np.array(y_scaled)

    # scale meta features
    meta_scaler = StandardScaler()
    X_meta_scaled = meta_scaler.fit_transform(X_meta)

    # build model
    model = build_model(n_steps=n_steps, meta_dim=X_meta_scaled.shape[1])

    # callbacks
    chk_path = "best_model.tmp.h5"
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, verbose=1, min_lr=1e-6),
        ModelCheckpoint(chk_path, monitor="val_loss", save_best_only=True, verbose=1)
    ]

    hist = model.fit(
        {"seq_input": X_seq_scaled, "meta_input": X_meta_scaled},
        y_scaled,
        validation_split=0.12,
        epochs=epochs,
        batch_size=32,
        callbacks=callbacks,
        verbose=1
    )

    # save final model (best weights already restored)
    model.save(MODEL_PATH)
    with open(SCALER_PATH, "wb") as f:
        pickle.dump(series_scaler, f)
    with open(META_SCALER_PATH, "wb") as f:
        pickle.dump(meta_scaler, f)

    # compute last validation rmse approx (take min val_loss from history)
    val_losses = hist.history.get("val_loss", [])
    val_rmse = np.sqrt(min(val_losses)) if len(val_losses)>0 else None

    return model, series_scaler, meta_scaler, val_rmse

def load_model_and_scalers():
    model = None
    series_scaler = None
    meta_scaler = None
    if os.path.exists(MODEL_PATH):
        model = load_model(MODEL_PATH, compile=True)
    if os.path.exists(SCALER_PATH):
        with open(SCALER_PATH, "rb") as f:
            series_scaler = pickle.load(f)
    if os.path.exists(META_SCALER_PATH):
        with open(META_SCALER_PATH, "rb") as f:
            meta_scaler = pickle.load(f)
    return model, series_scaler, meta_scaler

# -----------------------
# Prediction wrapper
# -----------------------
def predict_next(model, series_scaler, meta_scaler, recent_raw, n_steps=TIME_STEPS):
    # recent_raw: list of raw floats, length >= n_steps
    arr = np.array(recent_raw[-n_steps:]).astype(float).reshape(-1,1)
    scaled = series_scaler.transform(arr).flatten()
    meta = compute_meta_features(scaled)
    seq_input = scaled.reshape(1, n_steps, 1)
    meta_input = meta_scaler.transform(meta.reshape(1, -1))
    pred_scaled = model.predict({"seq_input": seq_input, "meta_input": meta_input}, verbose=0)
    pred_raw = series_scaler.inverse_transform(pred_scaled.reshape(-1,1))[0][0]
    return float(pred_raw)

# -----------------------
# Playwright capture (async)
# -----------------------
async def capture_with_playwright(account_id, stop_flag, data_container, url_override=None, headless=True, user_agent=None, retries=3):
    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=headless, args=["--no-sandbox", "--disable-setuid-sandbox"])
            ctx_args = {}
            if user_agent:
                ctx_args["user_agent"] = user_agent
            context = await browser.new_context(**ctx_args)
            page = await context.new_page()
            target_url = url_override if url_override else f"https://1xbet.com/player/{account_id}/crash"
            # try multiple times
            for attempt in range(1, retries+1):
                try:
                    await page.goto(target_url, timeout=30000, wait_until="networkidle")
                    break
                except Exception as e:
                    if attempt == retries:
                        raise
                    await asyncio.sleep(2)
            # handle websockets
            def on_ws(ws):
                async def on_frame(frame):
                    try:
                        text = frame
                        if isinstance(text, dict) and "text" in text:
                            text = text["text"]
                        # parse json
                        data = json.loads(text)
                        if isinstance(data, dict):
                            for key in ("crashPoint", "crash", "multiplier", "value"):
                                if key in data:
                                    try:
                                        val = float(data[key])
                                        if 0 < val < 10000:
                                            data_container.append(val)
                                            append_to_csv(val)
                                    except:
                                        pass
                    except:
                        pass
                ws.on("framereceived", lambda frame: asyncio.create_task(on_frame(frame)))
            page.on("websocket", on_ws)
            # keep alive until stop
            while not stop_flag["stop"]:
                await asyncio.sleep(1)
            await browser.close()
    except Exception as e:
        # append error object so UI can show it
        data_container.append({"__error__": str(e)})

# -----------------------
# Direct websockets capture (async)
# -----------------------
async def capture_with_direct_ws(ws_url, stop_flag, data_container):
    try:
        async with websockets.connect(ws_url) as websocket:
            while not stop_flag["stop"]:
                try:
                    msg = await asyncio.wait_for(websocket.recv(), timeout=10)
                except asyncio.TimeoutError:
                    continue
                try:
                    data = json.loads(msg)
                    if isinstance(data, dict):
                        for key in ("crashPoint", "crash", "multiplier", "value"):
                            if key in data:
                                try:
                                    val = float(data[key])
                                    if 0 < val < 10000:
                                        data_container.append(val)
                                        append_to_csv(val)
                                except:
                                    pass
                except:
                    # sometimes messages are not json - ignore
                    pass
    except Exception as e:
        data_container.append({"__error__": str(e)})

# wrapper to run capture in thread
def run_capture_runner(use_direct_ws, target, stop_flag, data_container, headless=True, user_agent=None):
    # runs in separate thread
    try:
        if use_direct_ws:
            asyncio.run(capture_with_direct_ws(target, stop_flag, data_container))
        else:
            asyncio.run(capture_with_playwright(target, stop_flag, data_container, headless=headless, user_agent=user_agent))
    except Exception as e:
        data_container.append({"__error__": str(e)})

# -----------------------
# Streamlit UI
# -----------------------
st.set_page_config(page_title="Crash Tracker — Improved", layout="centered")
st.title("📡 Crash Tracker — نسخة محسّنة (Playwright + WebSocket + ML)")

password = st.text_input("أدخل كلمة المرور:", type="password")
if password != PASSWORD:
    if password:
        st.error("كلمة المرور خاطئة!")
    st.stop()

col_top = st.columns(2)
with col_top[0]:
    account_id = st.text_input("رقم الحساب (ID) — أو اتركه فارغًا إذا لديك رابط/WS مباشر:")
with col_top[1]:
    ws_url = st.text_input("اختياري: رابط WebSocket مباشر (مثلاً: wss://... )")

url_override = st.text_input("اختياري: رابط الصفحة المباشر (مثلاً: https://...)")

# Playwright options
st.write("خيارات التقاط (اختياري):")
col_opts = st.columns(3)
with col_opts[0]:
    use_playwright = st.checkbox("استخدم Playwright (افتراضياً)", value=True)
with col_opts[1]:
    headless = st.checkbox("Headless (تشغيل بدون نافذة)", value=True)
with col_opts[2]:
    debug_mode = st.checkbox("وضع التصحيح (يعرض أخطاء مفصلة)", value=False)
user_agent = st.text_input("اختياري: User-Agent مخصص (لو فشل التحميل قد يساعد)","")

# controls
col_btns = st.columns(3)
with col_btns[0]:
    start_btn = st.button("ابدأ الجمع")
with col_btns[1]:
    stop_btn = st.button("أوقف الجمع")
with col_btns[2]:
    train_btn = st.button("درّب النموذج الآن")

# session state
if "data_deque" not in st.session_state:
    st.session_state.data_deque = deque(maxlen=20000)
if "stop_flag" not in st.session_state:
    st.session_state.stop_flag = {"stop": True}
if "capture_thread" not in st.session_state:
    st.session_state.capture_thread = None
if "model" not in st.session_state:
    model, series_scaler, meta_scaler = load_model_and_scalers()
    st.session_state.model = model
    st.session_state.series_scaler = series_scaler
    st.session_state.meta_scaler = meta_scaler
if "last_val_rmse" not in st.session_state:
    st.session_state.last_val_rmse = None

status_area = st.empty()
chart_area = st.empty()
log_area = st.empty()

# pre-load CSV history (to memory)
hist_df = load_history()
if not hist_df.empty and len(st.session_state.data_deque) < 100:
    for v in hist_df['value'].astype(float).tolist()[-5000:]:
        st.session_state.data_deque.append(v)

# start capture
if start_btn:
    if ws_url.strip() == "" and account_id.strip() == "" and url_override.strip() == "":
        st.warning("أدخل رقم الحساب أو رابط الصفحة أو رابط WebSocket المباشر.")
    else:
        if st.session_state.capture_thread and st.session_state.capture_thread.is_alive():
            st.warning("التقاط يعمل بالفعل.")
        else:
            st.session_state.stop_flag = {"stop": False}
            st.session_state.data_deque.clear()
            # reload CSV into deque
            if not hist_df.empty:
                for v in hist_df['value'].astype(float).tolist()[-5000:]:
                    st.session_state.data_deque.append(v)
            # choose capture method
            if ws_url.strip() != "":
                # direct websocket
                t = threading.Thread(target=run_capture_runner, args=(True, ws_url.strip(), st.session_state.stop_flag, st.session_state.data_deque), daemon=True)
            else:
                target = url_override.strip() if url_override.strip() != "" else account_id.strip()
                t = threading.Thread(target=run_capture_runner, args=(False, target, st.session_state.stop_flag, st.session_state.data_deque, headless, user_agent if user_agent else None), daemon=True)
            st.session_state.capture_thread = t
            t.start()
            status_area.success("🟢 بدأ جمع البيانات. انتظر قليلاً لظهور عينات في الرسم.")

# stop capture
if stop_btn:
    if st.session_state.capture_thread and st.session_state.capture_thread.is_alive():
        st.session_state.stop_flag["stop"] = True
        status_area.info("🔴 طلب إيقاف. انتظر ثوانٍ حتى يتوقف المتصفح/الاتصال.")
    else:
        status_area.info("لا يوجد عملية جمع تعمل الآن.")

# show recent
st.write("---")
recent_vals = list(st.session_state.data_deque)[-1000:]
if len(recent_vals) == 0:
    st.info("لا توجد بيانات حتى الآن — اضغط 'ابدأ الجمع' أو حمّل CSV.")
else:
    df_recent = pd.DataFrame({"value": recent_vals})
    chart_area.line_chart(df_recent["value"])
    st.write(f"عدد العينات المجمعة (في الذاكرة): {len(recent_vals)}")
    st.metric("آخر قيمة", f"{recent_vals[-1]:.2f}")
    st.metric("أكبر قيمة", f"{max(recent_vals):.2f}")

# manual save
if st.button("احفظ السجل الحالي إلى CSV الآن"):
    if len(recent_vals) > 0:
        temp_df = pd.DataFrame({"timestamp":[datetime.utcnow().isoformat()]*len(recent_vals), "value": recent_vals})
        if not os.path.exists(DATA_CSV):
            temp_df.to_csv(DATA_CSV, index=False)
        else:
            temp_df.to_csv(DATA_CSV, mode='a', header=False, index=False)
        st.success("تم الحفظ.")
    else:
        st.warning("لا يوجد بيانات للحفظ.")

# prepare combined series for training/prediction
combined_list = []
if not hist_df.empty:
    combined_list = hist_df['value'].astype(float).tolist()
# append deque values after CSV tail to avoid duplicates
for v in list(st.session_state.data_deque):
    if len(combined_list) == 0 or abs(combined_list[-1] - v) > 1e-12:
        combined_list.append(float(v))
series_arr = np.array(combined_list) if combined_list else np.array([])

# training
if series_arr.size >= MIN_SAMPLES_TO_TRAIN:
    st.write(f"يوجد {series_arr.size} عينة متاحة للتدريب.")
    if st.session_state.model is None or train_btn:
        with st.spinner("⏳ جارٍ تدريب النموذج..."):
            try:
                model, s_scaler, m_scaler, val_rmse = train_and_save(series_arr, epochs=80, n_steps=TIME_STEPS)
                st.session_state.model = model
                st.session_state.series_scaler = s_scaler
                st.session_state.meta_scaler = m_scaler
                st.session_state.last_val_rmse = val_rmse
                st.success("✅ تم تدريب النموذج وحفظه.")
                if val_rmse is not None:
                    st.info(f"تقدير RMSE على التحقق: {val_rmse:.4f} (استخدمه كمؤشر دقة تقريبي).")
            except Exception as e:
                st.error("حدث خطأ أثناء التدريب: " + str(e))
else:
    st.info(f"تحتاج {MIN_SAMPLES_TO_TRAIN} عينة على الأقل للتدريب الموثوق. حالياً: {series_arr.size}")

# prediction (if model exists)
st.write("---")
if st.session_state.model is not None and series_arr.size >= TIME_STEPS:
    try:
        pred = predict_next(st.session_state.model, st.session_state.series_scaler, st.session_state.meta_scaler, series_arr, n_steps=TIME_STEPS)
        st.markdown(f"### 🔮 التنبؤ التالي: **{pred:.2f}x**")
        if st.session_state.last_val_rmse:
            st.caption(f"تقديري للدقة التقريبية (RMSE): {st.session_state.last_val_rmse:.4f} — كلما نقصت القيمة تحسنت الدقة.")
    except Exception as e:
        st.error("خطأ أثناء التنبؤ: " + str(e))
else:
    st.write("لم يتوفر موديل مدرّب بعد أو لا توجد بيانات كافية للتنبؤ.")

# debug area
if st.checkbox("إظهار التفاصيل/سجل الأخطاء (Debug)"):
    st.write("Capture thread alive:", st.session_state.capture_thread.is_alive() if st.session_state.capture_thread else False)
    st.write("Data deque length:", len(st.session_state.data_deque))
    st.write("Model loaded:", st.session_state.model is not None)
    st.write("Series scaler present:", st.session_state.series_scaler is not None)
    st.write("Meta scaler present:", st.session_state.meta_scaler is not None)
    # show any errors in deque
    errors = [x for x in list(st.session_state.data_deque) if isinstance(x, dict) and "__error__" in x]
    st.write("Errors captured:", errors[:5])

st.write("---")
st.caption("ملاحظات: 1) إن لم ينجح Playwright في تحميل الصفحة، جرب إدخال رابط WebSocket مباشرة (افتح أدوات المطور في المتصفح → Network → WS → انسخ عنوان wss://). 2) دقة النموذج تعتمد تمامًا على كمية وجودة البيانات.")
