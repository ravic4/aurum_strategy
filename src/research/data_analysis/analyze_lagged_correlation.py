import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def analyze_lagged_correlation(data_file, output_dir, lag_months):
    """
    Analyze correlation between stock returns and lagged employment changes.

    Args:
        data_file: Path to merged prices/employment data
        output_dir: Directory for output files
        lag_months: Number of months to lag employment data (e.g., 1 means employment from t-1)
    """
    # Read the merged data
    df = pd.read_excel(data_file)
    df['date'] = pd.to_datetime(df['date'])

    # Drop rows with missing employment data
    df = df.dropna(subset=['total_employees'])

    # Sort by symbol and date
    df = df.sort_values(['symbol', 'date'])

    # Calculate month-over-month percent changes
    df['price_return'] = df.groupby('symbol')['close'].pct_change() * 100
    df['emp_change_pct'] = df.groupby('symbol')['total_employees'].pct_change() * 100
    df['ft_emp_change_pct'] = df.groupby('symbol')['full_time_employees'].pct_change() * 100
    df['pt_emp_change_pct'] = df.groupby('symbol')['part_time_employees'].pct_change() * 100

    # Create lagged employment changes (shift employment back by lag_months)
    # This means we're comparing current stock return with employment change from lag_months ago
    df['emp_change_pct_lagged'] = df.groupby('symbol')['emp_change_pct'].shift(lag_months)
    df['ft_emp_change_pct_lagged'] = df.groupby('symbol')['ft_emp_change_pct'].shift(lag_months)
    df['pt_emp_change_pct_lagged'] = df.groupby('symbol')['pt_emp_change_pct'].shift(lag_months)

    # Drop rows with NaN
    df_clean = df.dropna(subset=['price_return', 'emp_change_pct_lagged'])

    print("=" * 60)
    print(f"LAGGED CORRELATION ANALYSIS (Lag = {lag_months} month{'s' if lag_months > 1 else ''})")
    print("=" * 60)
    print(f"\nData: {len(df_clean)} rows")
    print(f"Companies: {df_clean['symbol'].nunique()}")
    print(f"Date range: {df_clean['date'].min().date()} to {df_clean['date'].max().date()}")

    # =========================================================================
    # CORRELATION ANALYSIS
    # =========================================================================
    print("\n" + "-" * 60)
    print("CORRELATION ANALYSIS")
    print("-" * 60)

    # Overall correlation
    return_col = 'price_return'
    emp_change_cols = ['emp_change_pct_lagged', 'ft_emp_change_pct_lagged', 'pt_emp_change_pct_lagged']

    print(f"\n--- Overall Correlation (Employment at t-{lag_months}) ---")
    overall_corr = df_clean[[return_col] + emp_change_cols].corr()
    print("\nStock Returns (t) vs Employment Changes (t-" + str(lag_months) + "):")
    for emp_col in emp_change_cols:
        corr = overall_corr.loc[return_col, emp_col]
        print(f"  Price Return vs {emp_col.replace('_lagged', '')}: {corr:.4f}")

    # Correlation by company
    print(f"\n--- Correlation by Company (Price Return vs Employment Change at t-{lag_months}) ---")
    company_correlations = []
    for symbol in df_clean['symbol'].unique():
        company_data = df_clean[df_clean['symbol'] == symbol]
        source_name = company_data['source_name'].iloc[0]
        corr = company_data['price_return'].corr(company_data['emp_change_pct_lagged'])
        company_correlations.append({
            'symbol': symbol,
            'source_name': source_name,
            'correlation': corr,
            'n_observations': len(company_data),
            'lag_months': lag_months
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
    print("\n" + "-" * 60)
    print("GENERATING VISUALIZATIONS")
    print("-" * 60)

    plt.style.use('seaborn-v0_8-whitegrid')
    lag_dir = output_dir / f'lag_{lag_months}'
    lag_dir.mkdir(exist_ok=True)

    # 1. Correlation bar chart by company
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    corr_matrix = corr_df.set_index('symbol')[['correlation']]
    corr_matrix = corr_matrix.sort_values('correlation', ascending=True)
    colors = ['#d73027' if x < 0 else '#1a9850' for x in corr_matrix['correlation']]
    ax1.barh(corr_matrix.index, corr_matrix['correlation'], color=colors)
    ax1.axvline(x=0, color='black', linewidth=0.5)
    ax1.set_xlabel('Correlation Coefficient')
    ax1.set_title(f'Stock Returns (t) vs Employment Change (t-{lag_months}) by Company')
    ax1.set_xlim(-1, 1)
    for i, row in enumerate(corr_matrix.itertuples()):
        ax1.text(row.correlation + 0.02 if row.correlation >= 0 else row.correlation - 0.02,
                 i, f"{row.correlation:.3f}", va='center',
                 ha='left' if row.correlation >= 0 else 'right')
    plt.tight_layout()
    fig1.savefig(lag_dir / 'correlation_by_company.png', dpi=150)
    print(f"  Saved: lag_{lag_months}/correlation_by_company.png")

    # 2. Scatter plots for each company
    n_companies = df_clean['symbol'].nunique()
    n_cols = 3
    n_rows = (n_companies + n_cols - 1) // n_cols
    fig2, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
    axes = axes.flatten()

    for i, symbol in enumerate(sorted(df_clean['symbol'].unique())):
        company_data = df_clean[df_clean['symbol'] == symbol]
        source_name = company_data['source_name'].iloc[0]
        corr = company_data['price_return'].corr(company_data['emp_change_pct_lagged'])

        ax = axes[i]
        ax.scatter(company_data['emp_change_pct_lagged'], company_data['price_return'],
                   alpha=0.6, edgecolors='black', linewidth=0.5)

        # Add trend line
        z = np.polyfit(company_data['emp_change_pct_lagged'], company_data['price_return'], 1)
        p = np.poly1d(z)
        x_line = np.linspace(company_data['emp_change_pct_lagged'].min(),
                            company_data['emp_change_pct_lagged'].max(), 100)
        ax.plot(x_line, p(x_line), "r--", alpha=0.8, linewidth=2)

        ax.set_xlabel(f'Employment Change % (t-{lag_months})')
        ax.set_ylabel('Stock Return % (t)')
        ax.set_title(f'{symbol} ({source_name})\nr = {corr:.3f}')

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    fig2.savefig(lag_dir / 'scatter_plots_by_company.png', dpi=150)
    print(f"  Saved: lag_{lag_months}/scatter_plots_by_company.png")

    plt.close('all')

    # Save correlation results to CSV
    corr_df.to_csv(lag_dir / 'correlation_results.csv', index=False)
    print(f"  Saved: lag_{lag_months}/correlation_results.csv")

    return corr_df


def run_all_lags(data_file, output_dir, max_lag=7):
    """
    Run lagged correlation analysis for lags 1 through max_lag months.
    Also creates a summary comparison across all lags.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    all_results = []

    for lag in range(1, max_lag + 1):
        print("\n" + "=" * 60)
        corr_df = analyze_lagged_correlation(data_file, output_dir, lag)
        all_results.append(corr_df)
        print()

    # Combine all results
    combined_df = pd.concat(all_results, ignore_index=True)

    # Create summary pivot table
    summary_pivot = combined_df.pivot(index='symbol', columns='lag_months', values='correlation')
    summary_pivot.columns = [f'lag_{col}' for col in summary_pivot.columns]

    # Add company names
    symbol_to_name = combined_df.drop_duplicates('symbol').set_index('symbol')['source_name']
    summary_pivot['source_name'] = summary_pivot.index.map(symbol_to_name)
    summary_pivot = summary_pivot[['source_name'] + [c for c in summary_pivot.columns if c != 'source_name']]

    # Save summary
    summary_pivot.to_csv(output_dir / 'lag_summary_by_company.csv')
    print(f"\nSaved: lag_summary_by_company.csv")

    # Calculate overall mean correlation for each lag
    overall_summary = combined_df.groupby('lag_months')['correlation'].agg(['mean', 'median', 'std']).reset_index()
    overall_summary.columns = ['lag_months', 'mean_correlation', 'median_correlation', 'std_correlation']
    overall_summary.to_csv(output_dir / 'lag_summary_overall.csv', index=False)
    print(f"Saved: lag_summary_overall.csv")

    # Create summary visualization
    plt.style.use('seaborn-v0_8-whitegrid')

    # 1. Line plot of correlations by lag for each company
    fig1, ax1 = plt.subplots(figsize=(12, 6))
    for symbol in combined_df['symbol'].unique():
        symbol_data = combined_df[combined_df['symbol'] == symbol].sort_values('lag_months')
        ax1.plot(symbol_data['lag_months'], symbol_data['correlation'],
                 marker='o', label=symbol, linewidth=2, markersize=6)
    ax1.axhline(y=0, color='black', linestyle='--', linewidth=0.5)
    ax1.set_xlabel('Lag (Months)')
    ax1.set_ylabel('Correlation Coefficient')
    ax1.set_title('Stock Returns vs Lagged Employment Changes by Company')
    ax1.set_xticks(range(1, max_lag + 1))
    ax1.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
    ax1.set_ylim(-1, 1)
    plt.tight_layout()
    fig1.savefig(output_dir / 'lag_comparison_by_company.png', dpi=150)
    print(f"Saved: lag_comparison_by_company.png")

    # 2. Overall mean correlation by lag
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    ax2.bar(overall_summary['lag_months'], overall_summary['mean_correlation'],
            color=['#1a9850' if x >= 0 else '#d73027' for x in overall_summary['mean_correlation']],
            edgecolor='black', linewidth=0.5)
    ax2.errorbar(overall_summary['lag_months'], overall_summary['mean_correlation'],
                 yerr=overall_summary['std_correlation'], fmt='none', color='black', capsize=5)
    ax2.axhline(y=0, color='black', linestyle='--', linewidth=0.5)
    ax2.set_xlabel('Lag (Months)')
    ax2.set_ylabel('Mean Correlation Coefficient')
    ax2.set_title('Mean Correlation: Stock Returns vs Lagged Employment Changes')
    ax2.set_xticks(range(1, max_lag + 1))
    ax2.set_ylim(-0.5, 0.5)
    plt.tight_layout()
    fig2.savefig(output_dir / 'lag_comparison_overall.png', dpi=150)
    print(f"Saved: lag_comparison_overall.png")

    # 3. Heatmap of all correlations
    fig3, ax3 = plt.subplots(figsize=(10, 8))
    heatmap_data = summary_pivot.drop('source_name', axis=1)
    sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='RdYlGn', center=0,
                linewidths=0.5, ax=ax3, vmin=-1, vmax=1)
    ax3.set_title('Correlation Heatmap: Stock Returns vs Lagged Employment Changes')
    ax3.set_xlabel('Lag (Months)')
    ax3.set_ylabel('Company')
    plt.tight_layout()
    fig3.savefig(output_dir / 'lag_heatmap.png', dpi=150)
    print(f"Saved: lag_heatmap.png")

    plt.close('all')

    # Print final summary
    print("\n" + "=" * 60)
    print("OVERALL SUMMARY")
    print("=" * 60)
    print("\nMean Correlation by Lag:")
    for _, row in overall_summary.iterrows():
        print(f"  Lag {int(row['lag_months'])}: {row['mean_correlation']:.4f} (±{row['std_correlation']:.4f})")

    best_lag = overall_summary.loc[overall_summary['mean_correlation'].abs().idxmax()]
    print(f"\nStrongest overall signal at lag {int(best_lag['lag_months'])}: {best_lag['mean_correlation']:.4f}")

    return combined_df, summary_pivot, overall_summary


if __name__ == "__main__":
    base_path = Path(__file__).parent
    data_file = base_path / "data" / "prices_employment_merged.xlsx"
    output_dir = base_path / "data" / "lagged_analysis"

    # Run analysis for lags 1-7
    combined_df, summary_pivot, overall_summary = run_all_lags(data_file, output_dir, max_lag=7)
