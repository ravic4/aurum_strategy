import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def analyze_price_vs_employment(data_file, output_dir):
    """
    Analyze correlation between stock prices and employment levels.
    Uses percent changes (month-over-month) for proper time-series analysis.
    Generates correlation statistics and visualizations.
    """
    # Read the merged data
    df = pd.read_excel(data_file)
    df['date'] = pd.to_datetime(df['date'])

    # Drop rows with missing employment data
    df = df.dropna(subset=['total_employees'])

    # Sort by symbol and date
    df = df.sort_values(['symbol', 'date'])

    # Calculate month-over-month percent changes for each company
    df['price_return'] = df.groupby('symbol')['close'].pct_change() * 100
    df['emp_change_pct'] = df.groupby('symbol')['total_employees'].pct_change() * 100
    df['ft_emp_change_pct'] = df.groupby('symbol')['full_time_employees'].pct_change() * 100
    df['pt_emp_change_pct'] = df.groupby('symbol')['part_time_employees'].pct_change() * 100

    # Drop the first row of each company (NaN from pct_change)
    df_clean = df.dropna(subset=['price_return', 'emp_change_pct'])

    print("=" * 60)
    print("STOCK PRICE VS EMPLOYMENT ANALYSIS")
    print("=" * 60)
    print(f"\nData: {len(df_clean)} rows ({len(df) - len(df_clean)} rows dropped due to missing data)")
    print(f"Companies: {df_clean['symbol'].nunique()}")
    print(f"Date range: {df_clean['date'].min().date()} to {df_clean['date'].max().date()}")

    # =========================================================================
    # CORRELATION ANALYSIS
    # =========================================================================
    print("\n" + "=" * 60)
    print("CORRELATION ANALYSIS")
    print("=" * 60)

    # Percent change columns
    return_col = 'price_return'
    emp_change_cols = ['emp_change_pct', 'ft_emp_change_pct', 'pt_emp_change_pct']

    print("\n--- Overall Correlation (All Companies) ---")
    overall_corr = df_clean[[return_col] + emp_change_cols].corr()
    print("\nStock Returns vs Employment Changes (Month-over-Month %):")
    for emp_col in emp_change_cols:
        corr = overall_corr.loc[return_col, emp_col]
        print(f"  Price Return vs {emp_col}: {corr:.4f}")

    # Correlation by company
    print("\n--- Correlation by Company (Price Return vs Employment Change %) ---")
    company_correlations = []
    for symbol in df_clean['symbol'].unique():
        company_data = df_clean[df_clean['symbol'] == symbol]
        source_name = company_data['source_name'].iloc[0]
        corr = company_data['price_return'].corr(company_data['emp_change_pct'])
        company_correlations.append({
            'symbol': symbol,
            'source_name': source_name,
            'correlation': corr,
            'n_observations': len(company_data)
        })
        print(f"  {symbol} ({source_name}): {corr:.4f}")

    corr_df = pd.DataFrame(company_correlations)
    corr_df = corr_df.sort_values('correlation', ascending=False)

    print("\n--- Summary Statistics ---")
    print(f"  Mean correlation: {corr_df['correlation'].mean():.4f}")
    print(f"  Median correlation: {corr_df['correlation'].median():.4f}")
    print(f"  Std deviation: {corr_df['correlation'].std():.4f}")
    print(f"  Strongest positive: {corr_df.iloc[0]['symbol']} ({corr_df.iloc[0]['correlation']:.4f})")
    print(f"  Strongest negative: {corr_df.iloc[-1]['symbol']} ({corr_df.iloc[-1]['correlation']:.4f})")

    # =========================================================================
    # VISUALIZATIONS
    # =========================================================================
    print("\n" + "=" * 60)
    print("GENERATING VISUALIZATIONS")
    print("=" * 60)

    # Set style
    plt.style.use('seaborn-v0_8-whitegrid')

    # 1. Correlation heatmap by company
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    corr_matrix = corr_df.set_index('symbol')[['correlation']]
    corr_matrix = corr_matrix.sort_values('correlation', ascending=True)
    colors = ['#d73027' if x < 0 else '#1a9850' for x in corr_matrix['correlation']]
    ax1.barh(corr_matrix.index, corr_matrix['correlation'], color=colors)
    ax1.axvline(x=0, color='black', linewidth=0.5)
    ax1.set_xlabel('Correlation Coefficient')
    ax1.set_title('Stock Returns vs Employment Change (%) Correlation by Company')
    ax1.set_xlim(-1, 1)
    for i, row in enumerate(corr_matrix.itertuples()):
        ax1.text(row.correlation + 0.02 if row.correlation >= 0 else row.correlation - 0.02,
                 i, f"{row.correlation:.3f}", va='center',
                 ha='left' if row.correlation >= 0 else 'right')
    plt.tight_layout()
    fig1.savefig(output_dir / 'correlation_by_company.png', dpi=150)
    print(f"  Saved: correlation_by_company.png")

    # 2. Scatter plots for each company
    n_companies = df_clean['symbol'].nunique()
    n_cols = 3
    n_rows = (n_companies + n_cols - 1) // n_cols
    fig2, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
    axes = axes.flatten()

    for i, symbol in enumerate(sorted(df_clean['symbol'].unique())):
        company_data = df_clean[df_clean['symbol'] == symbol]
        source_name = company_data['source_name'].iloc[0]
        corr = company_data['price_return'].corr(company_data['emp_change_pct'])

        ax = axes[i]
        ax.scatter(company_data['emp_change_pct'], company_data['price_return'],
                   alpha=0.6, edgecolors='black', linewidth=0.5)

        # Add trend line
        z = np.polyfit(company_data['emp_change_pct'], company_data['price_return'], 1)
        p = np.poly1d(z)
        x_line = np.linspace(company_data['emp_change_pct'].min(),
                            company_data['emp_change_pct'].max(), 100)
        ax.plot(x_line, p(x_line), "r--", alpha=0.8, linewidth=2)

        ax.set_xlabel('Employment Change (%)')
        ax.set_ylabel('Stock Return (%)')
        ax.set_title(f'{symbol} ({source_name})\nr = {corr:.3f}')

    # Hide empty subplots
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    fig2.savefig(output_dir / 'scatter_plots_by_company.png', dpi=150)
    print(f"  Saved: scatter_plots_by_company.png")

    # 3. Time series - dual axis plot for each company
    fig3, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
    axes = axes.flatten()

    for i, symbol in enumerate(sorted(df_clean['symbol'].unique())):
        company_data = df_clean[df_clean['symbol'] == symbol].sort_values('date')
        source_name = company_data['source_name'].iloc[0]

        ax1 = axes[i]
        ax2 = ax1.twinx()

        line1 = ax1.plot(company_data['date'], company_data['close'],
                         'b-', linewidth=2, label='Stock Price')
        line2 = ax2.plot(company_data['date'], company_data['total_employees'],
                         'g-', linewidth=2, label='Total Employees')

        ax1.set_xlabel('Date')
        ax1.set_ylabel('Stock Price ($)', color='blue')
        ax2.set_ylabel('Total Employees', color='green')
        ax1.tick_params(axis='y', labelcolor='blue')
        ax2.tick_params(axis='y', labelcolor='green')
        ax1.set_title(f'{symbol} ({source_name})')

        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc='upper left', fontsize=8)

        ax1.tick_params(axis='x', rotation=45)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    fig3.savefig(output_dir / 'time_series_by_company.png', dpi=150)
    print(f"  Saved: time_series_by_company.png")

    # 4. Overall correlation heatmap
    fig4, ax4 = plt.subplots(figsize=(8, 6))
    corr_all = df_clean[['price_return', 'emp_change_pct', 'ft_emp_change_pct', 'pt_emp_change_pct']].corr()
    sns.heatmap(corr_all, annot=True, fmt='.3f', cmap='RdYlGn', center=0,
                square=True, linewidths=0.5, ax=ax4)
    ax4.set_title('Correlation Matrix: Returns vs Employment Changes (%)')
    plt.tight_layout()
    fig4.savefig(output_dir / 'correlation_heatmap.png', dpi=150)
    print(f"  Saved: correlation_heatmap.png")

    plt.close('all')

    # Save correlation results to CSV
    corr_df.to_csv(output_dir / 'correlation_results.csv', index=False)
    print(f"  Saved: correlation_results.csv")

    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)

    return corr_df


if __name__ == "__main__":
    base_path = Path(__file__).parent.parent  # Go up to research folder
    data_file = base_path / "data" / "prices_employment_merged.xlsx"
    output_dir = base_path / "data" / "analysis_output"

    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run analysis
    results = analyze_price_vs_employment(data_file, output_dir)
