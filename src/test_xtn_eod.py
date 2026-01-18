from dotenv import load_dotenv
import os
import requests

load_dotenv()

API_KEY = os.getenv("FMP_API_KEY")

url = "https://financialmodelingprep.com/stable/historical-price-eod/light"
params = {
    "symbol": "XTN",
    "apikey": API_KEY
}

r = requests.get(url, params=params, timeout=10)
r.raise_for_status()

data = r.json()
print("Rows:", len(data))
print("Sample:", data[:3])
