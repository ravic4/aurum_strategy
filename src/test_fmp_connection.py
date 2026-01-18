from dotenv import load_dotenv
import os
import requests

# Load .env from repo root
load_dotenv()

API_KEY = os.getenv("FMP_API_KEY")
assert API_KEY, "FMP_API_KEY not found in environment"

BASE_URL = "https://financialmodelingprep.com/stable"
ENDPOINT = f"{BASE_URL}/quote-short"

params = {
    "symbol": "AAPL",
    "apikey": API_KEY
}

response = requests.get(ENDPOINT, params=params, timeout=10)

print("Status Code:", response.status_code)
print("Raw Response:", response.text)

response.raise_for_status()

data = response.json()
print("\nParsed JSON:")
print(data)
