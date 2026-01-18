import os
from dotenv import load_dotenv
from src.alpaca_client import get_alpaca_client


def main():
    load_dotenv()
    symbol = os.getenv("SYMBOL", "AAPL")

    api = get_alpaca_client()
    acct = api.get_account()

    print("Connected ✅")
    print("Account status:", acct.status)
    print("Buying power:", acct.buying_power)

    bars = api.get_bars(symbol, timeframe="1Day", limit=5)
    print(f"\nLast 5 daily closes for {symbol}:")
    for b in bars:
        print(b.t, float(b.c))


if __name__ == "__main__":
    main()
