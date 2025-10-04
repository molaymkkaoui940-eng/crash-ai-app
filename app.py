# app.py
import streamlit as st
import requests
import pandas as pd
import numpy as np
import time
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from datetime import datetime
import joblib
import os

st.set_page_config(page_title="Crash Tracker AI", layout="centered")

st.title("🚀 تطبيق تتبع انفجارات الطائرة (Crash Tracker) - تجريبي")
st.write("أدخل رقم الـ id للعبة للحصول على بيانات وتوقعات. (استبدل رابط API في الكود بالرابط الحقيقي)")

# ---------- إعدادات ---------- #
API_BASE = "https://api.example.com/game_data"  # <-- غيره إلى رابط الـ API الحقيقي
HISTORY_CSV = "history.csv"
MODEL_FILE = "model.joblib"

# تحميل/إنشاء سجل تاريخي
if os.path.exists(HISTORY_CSV):
    history_df = pd.read_csv(HISTORY_CSV)
else:
    history_df = pd.DataFrame(columns=["timestamp", "game_id", "time", "explosion_val"])

# إدخال المستخدم
game_id = st.text_input("أدخل رقم الـ ID للعبة", value="")
sensitivity = st.slider("حساسية التنبيه (أصغر = إنذار مبكر)", 0.1, 10.0, 1.0, step=0.1)

col1, col2 = st.columns(2)
with col1:
    if st.button("جلب بيانات جديدة الآن"):
        if not game_id:
            st.warning("رجاءً ادخل رقم الـ ID أولاً.")
        else:
            # ----- استدعاء API للحصول على بيانات الجولة الحالية -----
            try:
                resp = requests.get(f"{API_BASE}/{game_id}")
                if resp.status_code == 200:
                    data = resp.json()
                    # هنا نفترض أن الـ API يعطي حقلَي: time (timestamp) و explosion (قيمة/مدة او معامل)
                    # مثال بنية JSON متوقعة: {"time": 1680000000, "explosion": 12.34}
                    now = datetime.utcnow().isoformat()
                    row = {
                        "timestamp": now,
                        "game_id": game_id,
                        "time": data.get("time", time.time()),
                        "explosion_val": data.get("explosion", np.nan)
                    }
                    history_df = history_df.append(row, ignore_index=True)
                    history_df.to_csv(HISTORY_CSV, index=False)
                    st.success("تم جلب البيانات وحفظها في السجل.")
                    st.json(data)
                else:
                    st.error(f"فشل جلب البيانات من الخادم: {resp.status_code}")
            except Exception as e:
                st.error(f"خطأ اتصال: {e}")

with col2:
    if st.button("تدريب / تحديث النموذج"):
        if len(history_df) < 5:
            st.warning("لا توجد بيانات كافية للتدريب (يحتاج 5 عينات على الأقل).")
        else:
            # تجهيز البيانات البسيطة
            df = history_df.dropna(subset=["time", "explosion_val"]).copy()
            X = np.array(df["time"]).reshape(-1, 1)  # مثال مبسط: نستخدم الوقت ك feature
            y = np.array(df["explosion_val"])
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            model = RandomForestRegressor(n_estimators=50, random_state=42)
            model.fit(X_train, y_train)
            # حفظ النموذج
            joblib.dump(model, MODEL_FILE)
            st.success("تم تدريب النموذج وحفظه.")
            st.write(f"نماذج محفوظة في: {MODEL_FILE}")

st.markdown("---")
st.subheader("سجل البيانات (History)")
st.dataframe(history_df.tail(20))

# ---------- التنبؤ ---------- #
st.markdown("---")
st.subheader("توقع الانفجار (تجريبي)")

if os.path.exists(MODEL_FILE):
    model = joblib.load(MODEL_FILE)
    current_time = time.time()
    pred = model.predict(np.array([[current_time]]))[0]
    st.write(f"🔮 التوقع الحالي لوقت/قيمة الانفجار: **{pred:.3f}**")
    # إذا كان التنبؤ أقل من الحساسية -> عرض تنبيه
    if pred < sensitivity:
        st.warning(f"⚠️ تنبيه: التنبؤ ({pred:.3f}) أقل من حساسية التنبيه ({sensitivity}). فكر في السحب الآن.")
else:
    st.info("لم يتم تدريب نموذج بعد. اضغط على 'تدريب / تحديث النموذج' بعد جمع بيانات كافية.")

# ---------- تنزيل سجل أو مسحه ---------- #
st.markdown("---")
cold1, cold2 = st.columns([1, 1])
with cold1:
    if st.button("تنزيل سجل CSV"):
        st.download_button("Download CSV", history_df.to_csv(index=False), file_name="history.csv")
with cold2:
    if st.button("مسح السجل"):
        if st.confirm("هل تريد مسح سجل التاريخ؟"):
            history_df = history_df.iloc[0:0]
            history_df.to_csv(HISTORY_CSV, index=False)
            st.success("تم مسح السجل.")
