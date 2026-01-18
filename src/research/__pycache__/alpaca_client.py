import os
from dotenv import load_dotenv
from alpaca_trade_api import REST


def get_alpaca_client() -> REST:
    load_dotenv()

    api_key = os.getenv("ALPACA_API_KEY")
    secret_key = os.getenv("ALPACA_SECRET_KEY")
    base_url = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")

    if not api_key or not secret_key:
        raise RuntimeError("Missing ALPACA_API_KEY / ALPACA_SECRET_KEY in .env")

    # Optional safety: block live trading by default
    if "paper-api" not in base_url:
        raise RuntimeError("LIVE TRADING BLOCKED: ALPACA_BASE_URL is not paper-api")

    return REST(api_key, secret_key, base_url, api_version="v2")
