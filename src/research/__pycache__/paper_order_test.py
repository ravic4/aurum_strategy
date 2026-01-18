import os
from dotenv import load_dotenv
from src.alpaca_client import get_alpaca_client


def main():
    load_dotenv()
    symbol = os.getenv("SYMBOL", "AAPL")
    qty = int(os.getenv("TEST_QTY", "1"))

    api = get_alpaca_client()

    acct = api.get_account()
    print("Connected ✅", "status:", acct.status, "buying_power:", acct.buying_power)

    # Safety: confirm paper endpoint
    base_url = os.getenv("ALPACA_BASE_URL", "")
    if "paper-api" not in base_url:
        raise RuntimeError("🚨 Refusing to trade: ALPACA_BASE_URL is not paper-api")

    # Optional: show last trade price snapshot (best-effort)
    try:
        last = api.get_latest_trade(symbol)
        print(f"Latest trade for {symbol}: price={last.price}")
    except Exception as e:
        print("Could not fetch latest trade (ok):", e)

    print(f"\nSubmitting PAPER market BUY: {qty} share(s) of {symbol} ...")
    order = api.submit_order(
        symbol=symbol,
        qty=1,
        side="buy",
        type="limit",
        time_in_force="day",
        limit_price=round(float(api.get_latest_trade(symbol).price) * 1.02, 2),  # pay a bit above last
        extended_hours=True,
    )
    print(order.id, order.status)
    print("Submitted order:", order.id, "status:", order.status)

    # Fetch the updated order state
    refreshed = api.get_order(order.id)
    print("Order now:", refreshed.id, "status:", refreshed.status)

    print("\nOpen orders:")
    for o in api.list_orders(status="open", limit=20):
        print(" ", o.id, o.symbol, o.side, o.qty, o.status)

    print("\nPositions:")
    for p in api.list_positions():
        print(" ", p.symbol, "qty=", p.qty, "avg=", p.avg_entry_price, "unrealized_pl=", p.unrealized_pl)


if __name__ == "__main__":
    main()
