import streamlit as st
from datetime import datetime, timedelta
import time

flight_data = []
explosion_predictions = []

def predict_explosion(speed, sensitivity):
    threshold = 100 - (sensitivity * 10)
    if speed > threshold:
        return 5  # ثواني حتى الانفجار (كمثال)
    else:
        return None

def main():
    st.title('تطبيق ذكي لتتبع انفجارات الطائرة في لعبة كراش')

    password = st.text_input('أدخل كلمة المرور:', type='password')
    if password != '1994':
        if password:
            st.error('كلمة المرور غير صحيحة، حاول مرة أخرى.')
        return
    else:
        st.success('تم التحقق من كلمة المرور. أهلاً بك!')

    sensitivity = st.slider('إعداد حساسية التنبيهات:', 1, 10, 5)
    speed = st.number_input('سرعة الطائرة (كم/س):', min_value=0.0, step=0.1)

    if st.button('تشغيل التتبع'):
        explosion_time = predict_explosion(speed, sensitivity)
        if explosion_time:
            start_time = datetime.now()
            end_time = start_time + timedelta(seconds=explosion_time)
            st.warning(f'تنبيه: الانفجار متوقع خلال {explosion_time} ثانية!')

            # عرض عد تنازلي للوقت المتبقي داخل التطبيق
            placeholder = st.empty()
            while datetime.now() < end_time:
                remaining = (end_time - datetime.now()).total_seconds()
                placeholder.markdown(f"### الوقت المتبقي للسحب: {int(remaining)} ثانية")
                time.sleep(1)
            placeholder.markdown("### 💥 الطائرة انفجرت! هل قمت بالسحب؟")

            explosion_predictions.append({'time': datetime.now(), 'predicted_seconds': explosion_time})
        else:
            st.info('لا يوجد انفجار متوقع حالياً.')

        flight_data.append({'time': datetime.now(), 'speed': speed})

    if len(flight_data) > 0:
        st.subheader('سجل بيانات الجولة')
        for entry in flight_data[-10:]:
            st.write(f"الوقت: {entry['time'].strftime('%H:%M:%S')} - السرعة: {entry['speed']} كم/س")

    if len(explosion_predictions) > 0:
        st.subheader('سجل التنبؤات')
        for pred in explosion_predictions[-5:]:
            st.write(f"الوقت: {pred['time'].strftime('%H:%M:%S')} - وقت الانفجار المتوقع: {pred['predicted_seconds']} ثواني")

if __name__ == '__main__':
    main()
