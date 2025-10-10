import requests
import random
import re
import time

class CrashTracker:
    USER_AGENTS = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Linux; Android 10; Mobile; rv:108.0) Gecko/108.0 Firefox/108.0",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_3) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.4 Safari/605.1.15",
        "Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.159 Safari/537.36",
        "Mozilla/5.0 (Linux; U; Android 11; en-US; Pixel 4a Build/RP1A.200720.009) AppleWebKit/537.36 (KHTML, like Gecko) Version/4.0 Chrome/102.0.5005.99 Mobile Safari/537.36"
    ]

    def __init__(self, user_id, promo_code, base_url, password):
        self.user_id = user_id
        self.promo_code = promo_code
        self.base_url = base_url
        self.password = password
        self.session = requests.Session()
        self.running = False

    def get_random_headers(self):
        ua = random.choice(self.USER_AGENTS)
        headers = {
            'User-Agent': ua,
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'Referer': self.base_url,
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8'
        }
        return headers

    def fetch_page(self):
        url = f"{self.base_url}/user/{self.user_id}/promo/{self.promo_code}"
        tries = 3
        for attempt in range(tries):
            headers = self.get_random_headers()
            try:
                response = self.session.get(url, headers=headers, timeout=10)
                if response.status_code == 200:
                    return response.text
                else:
                    print(f"Warning: status {response.status_code} @ attempt {attempt+1}")
            except Exception as e:
                print(f"Warning: Exception at attempt {attempt+1}: {e}")
            wait_time = random.uniform(1.5, 3.0)
            print(f"Retrying after {wait_time:.2f}s...")
            time.sleep(wait_time)
        return None

    def extract_crash_number(self, page_text):
        if not page_text:
            return None
        match = re.search(r'(d+.d+)x', page_text)
        if match:
            return float(match.group(1))
        return None

    def get_crash_number(self):
        page = self.fetch_page()
        crash_num = self.extract_crash_number(page)
        if crash_num is None:
            crash_num = round(random.uniform(1.0, 4.0), 2)
            print(f"No valid crash number found, using fallback: {crash_num}x")
        else:
            print(f"Crash number retrieved: {crash_num}x")
        return crash_num

    def start(self):
        if self.password != '1994':
            print("Access denied: wrong password.")
            return
        if self.running:
            print("Already running.")
            return
        self.running = True
        print("Tracker started...")
        self.run()

    def stop(self):
        if not self.running:
            print("Not running.")
            return
        self.running = False
        print("Tracker stopped.")

    def run(self):
        while self.running:
            crash_number = self.get_crash_number()
            pull_out_time = crash_number - 0.2
            print(f"Pull out BEFORE: {pull_out_time}x")
            sleep_time = random.uniform(4.8, 6.0)
            time.sleep(sleep_time)


def main():
    user_id = "1403702935"
    promo_code = "1x_2051622"
    base_url = "https://refpa58144.com"
    password = "1994"

    tracker = CrashTracker(user_id, promo_code, base_url, password)

    while True:
        cmd = input("Enter command (start/stop/exit): ").lower()
        if cmd == "start":
            tracker.start()
        elif cmd == "stop":
            tracker.stop()
        elif cmd == "exit":
            if tracker.running:
                tracker.stop()
            break
        else:
            print("Unknown command.")

if __name__ == "__main__":
    main()
