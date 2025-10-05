import streamlit as st
import requests
from bs4 import BeautifulSoup
import time
import random
from streamlit_extras.switch_page_button import switch_page

# قائمة User-Agent لتغيير رأس الطلب
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64)...",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)...",
]

def get_headers():
    return {'User-Agent': random.choice(USER_AGENTS)}

def fetch_flight_data(account_id):
    # مثال رابط وهمي لموقع MELBET
    url = f"https://melbet.com/player/{account_id}/crash"
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

def parse_flight_data(html):
    soup = BeautifulSoup(html, 'html.parser')
    # مثال مجرد لاستخراج بيانات
    speed_element = soup.find("div", class_="flight-speed")
    if speed_element:
        speed = speed_element.text.strip()
        return speed
    return "لا توجد بيانات حاليًا"

def main():
    st.set_page_config(page_title="تتبع انفجارات الطائرة في كراش", page_icon="✈️", layout="centered")
    
    st.markdown(
        """
        <style>
        .main {background-color: #1e1e2e; color: #f0f0f5;}
        .block-container {padding: 2rem 5rem 2rem 5rem;}
        h1 {color: #00bfff;}
        .stButton>button {background-color: #00bfff; color: white;}
        .stTextInput>div>div>input {background-color: #2e2e3e; color: white;}
        </style>
        """, unsafe_allow_html=True
    )
    
    st.title("✈️ تطبيق تتبع انفجارات الطائرة - لعبة كراش")

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

    start_tracking = st.button("⬆️ بدء التتبع")
    stop_tracking = st.button("🛑 إيقاف التطبيق")

    if start_tracking:
        st.info("جارٍ تتبع حركة الطائرة - اضغط على إيقاف التطبيق لإغلاق التتبع.")
        with st.empty() as placeholder:
            while True:
                if stop_tracking:
                    st.warning("تم إيقاف التتبع بناءً على طلبك.")
                    break
                
                html = fetch_flight_data(account_id)
                if html:
                    flight_info = parse_flight_data(html)
                    placeholder.markdown(f"### حركة الطائرة: {flight_info}")
                else:
                    placeholder.error("فشل في جلب البيانات.")

                time.sleep(random.randint(8,15))  # دلّل الحظر بالتوقف العشوائي

if __name__ == "__main__":
    main()
