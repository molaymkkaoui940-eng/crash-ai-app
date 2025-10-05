import streamlit as st
from datetime import datetime
import pandas as pd

# كلمة مرور بسيطة لحماية التطبيق
PASSWORD = "1994"

def main():
    st.title("تطبيق ذكي لتتبع انفجارات الطائرة في لعبة كراش")
    
    # تحقق من كلمة المرور
    input_pass = st.text_input("أدخل كلمة المرور:", type="password")
    if input_pass != PASSWORD:
        st.error("كلمة المرور غير صحيحة!")
        st.stop()

    st.success("تم التحقق من كلمة المرور. أهلاً بك!")
    
    # إعداد صفحة التطبيق
    st.markdown("---")
    st.header("إدخال بيانات الجولة")
    
    speed = st.number_input("سرعة الطائرة (كم/س):", min_value=0, step=10)
    altitude = st.number_input("ارتفاع الطائرة (قدم):", min_value=0, step=1000)
    temperature = st.number_input("درجة الحرارة (°C):", min_value=-50, max_value=100)
    fuel = st.number_input("كمية الوقود (لتر):", min_value=0)
    
    sensitivity = st.slider("حساسية التنبيه:", 1, 10, 5)
    
    if st.button("توقع انفجار الطائرة"):
        # هنا يمكن وضع خوارزمية الذكاء الاصطناعي الخاصة بك
        # النموذج هنا تبسيطي يعتمد على علاقة افتراضية بين السرعة والارتفاع وكمية الوقود
        # استبدل هذا الجزء بكود النموذج الحقيقي الذي تستخدمه
        
        risk_score = (speed * 0.3 + altitude * 0.2 + fuel * 0.5) / 10000
        
        if risk_score >= (sensitivity * 0.1):
            st.warning(f"تنبيه: انفجار متوقع قريباً! (نقاط الخطر: {risk_score:.2f})")
            st.markdown(f"⚠️ **سحب الرهان الآن!** ⚠️")
        else:
            st.info(f"الطائرة مستقرة حالياً. (نقاط الخطر: {risk_score:.2f})")
        
        # تسجيل النتائج في سجل التاريخ
        record = {
            "الوقت": datetime.now(),
            "سرعة": speed,
            "ارتفاع": altitude,
            "درجة الحرارة": temperature,
            "كمية الوقود": fuel,
            "نقاط الخطر": risk_score,
            "حساسية": sensitivity,
            "النتيجة": "انفجار متوقع" if risk_score >= (sensitivity * 0.1) else "مستقرة"
        }
        
        if "history" not in st.session_state:
            st.session_state.history = []
        st.session_state.history.append(record)
        
    # عرض سجل النتائج
    if "history" in st.session_state and len(st.session_state.history) > 0:
        st.header("سجل التوقعات")
        df = pd.DataFrame(st.session_state.history)
        st.dataframe(df)

if __name__ == "__main__":
    main()
