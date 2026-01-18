import pandas as pd
from pathlib import Path

def convert_eod_to_monthly(input_file, output_file):
    """
    Convert end-of-day price data to monthly data using the last trading day of each month.

    Args:
        input_file: Path to the input CSV file with EOD data
        output_file: Path to the output CSV file for monthly data
    """
    # Read the CSV file
    df = pd.read_csv(input_file)

    # Convert date column to datetime
    df['date'] = pd.to_datetime(df['date'])

    # Sort by symbol and date to ensure proper ordering
    df = df.sort_values(['symbol', 'date'])

    # Create a year-month column for grouping
    df['year_month'] = df['date'].dt.to_period('M')

    # Group by symbol and year_month, taking the last row (last trading day) of each month
    monthly_df = df.groupby(['symbol', 'year_month']).last().reset_index()

    # Drop the year_month helper column
    monthly_df = monthly_df.drop('year_month', axis=1)

    # Sort by symbol and date
    monthly_df = monthly_df.sort_values(['symbol', 'date']).reset_index(drop=True)

    # Save to CSV
    monthly_df.to_csv(output_file, index=False)

    print(f"Conversion complete!")
    print(f"Original data: {len(df)} rows")
    print(f"Monthly data: {len(monthly_df)} rows")
    print(f"Number of symbols: {df['symbol'].nunique()}")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"Output saved to: {output_file}")

if __name__ == "__main__":
    # Define paths
    base_path = Path(__file__).parent / "data"
    input_file = base_path / "prices_eod_long.csv"
    output_file = base_path / "prices_monthly_long.csv"

    # Convert to monthly
    convert_eod_to_monthly(input_file, output_file)
