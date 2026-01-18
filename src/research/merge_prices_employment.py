import pandas as pd
from pathlib import Path

def merge_prices_with_employment(prices_file, employment_file, output_file):
    """
    Merge monthly stock prices with airline employment data.

    Cross-references source_name from prices with Carrier Name from employment data,
    and combines stock prices with part-time, full-time, and total worker counts.

    Args:
        prices_file: Path to the monthly prices CSV
        employment_file: Path to the airline employment Excel file
        output_file: Path to the output Excel file
    """
    # Read the data files
    prices_df = pd.read_csv(prices_file)
    employment_df = pd.read_excel(employment_file)

    # Convert dates to datetime and extract year-month for matching
    prices_df['date'] = pd.to_datetime(prices_df['date'])
    employment_df['Month Date'] = pd.to_datetime(employment_df['Month Date'])

    # Create year-month columns for merging
    prices_df['year_month'] = prices_df['date'].dt.to_period('M')
    employment_df['year_month'] = employment_df['Month Date'].dt.to_period('M')

    # Select relevant columns from employment data
    employment_subset = employment_df[['Carrier Name', 'Full Time', 'Part Time', 'Total', 'year_month']].copy()
    employment_subset = employment_subset.rename(columns={
        'Carrier Name': 'source_name',
        'Full Time': 'full_time_employees',
        'Part Time': 'part_time_employees',
        'Total': 'total_employees'
    })

    # Merge prices with employment data
    merged_df = prices_df.merge(
        employment_subset,
        on=['source_name', 'year_month'],
        how='left'
    )

    # Drop the helper year_month column
    merged_df = merged_df.drop('year_month', axis=1)

    # Reorder columns for better readability
    column_order = [
        'date',
        'symbol',
        'source_name',
        'open',
        'high',
        'low',
        'close',
        'volume',
        'vwap',
        'change',
        'changePercent',
        'exchange',
        'ticker_status',
        'full_time_employees',
        'part_time_employees',
        'total_employees'
    ]
    merged_df = merged_df[column_order]

    # Sort by symbol and date
    merged_df = merged_df.sort_values(['symbol', 'date']).reset_index(drop=True)

    # Save to Excel
    merged_df.to_excel(output_file, index=False, sheet_name='Prices_Employment')

    # Print summary statistics
    print("Merge complete!")
    print(f"\nTotal rows: {len(merged_df)}")
    print(f"Unique symbols: {merged_df['symbol'].nunique()}")
    print(f"Date range: {merged_df['date'].min().date()} to {merged_df['date'].max().date()}")

    # Check for any missing employment data
    missing_employment = merged_df['total_employees'].isna().sum()
    if missing_employment > 0:
        print(f"\nWarning: {missing_employment} rows have missing employment data")
        missing_by_company = merged_df[merged_df['total_employees'].isna()].groupby('source_name').size()
        print("Missing data by company:")
        print(missing_by_company)
    else:
        print("\nAll rows have employment data matched!")

    print(f"\nOutput saved to: {output_file}")

    # Show sample of merged data
    print("\nSample of merged data:")
    print(merged_df.head(10).to_string())

    return merged_df


if __name__ == "__main__":
    # Define paths
    base_path = Path(__file__).parent
    prices_file = base_path / "data" / "prices_monthly_long.csv"
    employment_file = base_path / "airline_employment.xlsx"
    output_file = base_path / "data" / "prices_employment_merged.xlsx"

    # Run the merge
    merged_df = merge_prices_with_employment(prices_file, employment_file, output_file)
