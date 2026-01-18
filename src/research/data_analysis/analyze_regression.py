import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import cross_val_score
import statsmodels.api as sm


def run_regression_analysis(data_file, output_dir, max_lag=7):
    """
    Run linear regression analysis for stock returns vs employment changes
    at various lags (t-0 through t-max_lag).
    """
    # Read and prepare data
    df = pd.read_excel(data_file)
    df['date'] = pd.to_datetime(df['date'])
    df = df.dropna(subset=['total_employees'])
    df = df.sort_values(['symbol', 'date'])

    # Calculate percent changes
    df['price_return'] = df.groupby('symbol')['close'].pct_change() * 100
    df['emp_change_pct'] = df.groupby('symbol')['total_employees'].pct_change() * 100
    df['ft_emp_change_pct'] = df.groupby('symbol')['full_time_employees'].pct_change() * 100
    df['pt_emp_change_pct'] = df.groupby('symbol')['part_time_employees'].pct_change() * 100

    # Create lagged features for all lags
    for lag in range(1, max_lag + 1):
        df[f'emp_lag_{lag}'] = df.groupby('symbol')['emp_change_pct'].shift(lag)
        df[f'ft_emp_lag_{lag}'] = df.groupby('symbol')['ft_emp_change_pct'].shift(lag)
        df[f'pt_emp_lag_{lag}'] = df.groupby('symbol')['pt_emp_change_pct'].shift(lag)

    # Drop NaN rows
    lag_cols = [f'emp_lag_{i}' for i in range(1, max_lag + 1)]
    df_clean = df.dropna(subset=['price_return', 'emp_change_pct'] + lag_cols)

    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    print("=" * 70)
    print("LINEAR REGRESSION ANALYSIS: STOCK RETURNS vs EMPLOYMENT CHANGES")
    print("=" * 70)
    print(f"\nData: {len(df_clean)} rows")
    print(f"Companies: {df_clean['symbol'].nunique()}")
    print(f"Date range: {df_clean['date'].min().date()} to {df_clean['date'].max().date()}")

    results = []

    # =========================================================================
    # 1. SIMPLE LINEAR REGRESSION FOR EACH LAG (ALL COMPANIES POOLED)
    # =========================================================================
    print("\n" + "=" * 70)
    print("1. SIMPLE LINEAR REGRESSION BY LAG (Pooled Data)")
    print("=" * 70)

    simple_results = []

    for lag in range(0, max_lag + 1):
        if lag == 0:
            X_col = 'emp_change_pct'
        else:
            X_col = f'emp_lag_{lag}'

        X = df_clean[[X_col]].values
        y = df_clean['price_return'].values

        # Fit sklearn model
        model = LinearRegression()
        model.fit(X, y)
        y_pred = model.predict(X)

        # Calculate metrics
        r2 = r2_score(y, y_pred)
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        mae = mean_absolute_error(y, y_pred)

        # Cross-validation
        cv_scores = cross_val_score(model, X, y, cv=5, scoring='r2')

        # Statsmodels for p-values
        X_sm = sm.add_constant(X)
        sm_model = sm.OLS(y, X_sm).fit()
        coef = sm_model.params[1]
        p_value = sm_model.pvalues[1]
        intercept = sm_model.params[0]

        simple_results.append({
            'lag': lag,
            'coefficient': coef,
            'intercept': intercept,
            'r2': r2,
            'adj_r2': sm_model.rsquared_adj,
            'rmse': rmse,
            'mae': mae,
            'p_value': p_value,
            'cv_r2_mean': cv_scores.mean(),
            'cv_r2_std': cv_scores.std(),
            'significant': p_value < 0.05
        })

        sig_marker = "*" if p_value < 0.05 else ""
        print(f"\nLag {lag}:")
        print(f"  Coefficient: {coef:.4f} (p={p_value:.4f}){sig_marker}")
        print(f"  R²: {r2:.4f} | Adj R²: {sm_model.rsquared_adj:.4f}")
        print(f"  RMSE: {rmse:.4f} | MAE: {mae:.4f}")
        print(f"  CV R² (5-fold): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    simple_df = pd.DataFrame(simple_results)
    simple_df.to_csv(output_dir / 'regression_simple_by_lag.csv', index=False)
    print(f"\nSaved: regression_simple_by_lag.csv")

    # =========================================================================
    # 2. SIMPLE LINEAR REGRESSION BY COMPANY AND LAG
    # =========================================================================
    print("\n" + "=" * 70)
    print("2. SIMPLE LINEAR REGRESSION BY COMPANY AND LAG")
    print("=" * 70)

    company_results = []

    for symbol in df_clean['symbol'].unique():
        company_data = df_clean[df_clean['symbol'] == symbol]
        source_name = company_data['source_name'].iloc[0]

        print(f"\n--- {symbol} ({source_name}) ---")

        for lag in range(0, max_lag + 1):
            if lag == 0:
                X_col = 'emp_change_pct'
            else:
                X_col = f'emp_lag_{lag}'

            X = company_data[[X_col]].values
            y = company_data['price_return'].values

            # Fit model
            model = LinearRegression()
            model.fit(X, y)
            y_pred = model.predict(X)

            r2 = r2_score(y, y_pred)

            # Statsmodels for p-values
            X_sm = sm.add_constant(X)
            sm_model = sm.OLS(y, X_sm).fit()
            coef = sm_model.params[1]
            p_value = sm_model.pvalues[1]

            company_results.append({
                'symbol': symbol,
                'source_name': source_name,
                'lag': lag,
                'coefficient': coef,
                'r2': r2,
                'adj_r2': sm_model.rsquared_adj,
                'p_value': p_value,
                'n_obs': len(company_data),
                'significant': p_value < 0.05
            })

        # Print summary for this company
        company_summary = [r for r in company_results if r['symbol'] == symbol]
        best_lag = max(company_summary, key=lambda x: abs(x['r2']))
        sig_lags = [r['lag'] for r in company_summary if r['significant']]
        print(f"  Best R² at lag {best_lag['lag']}: {best_lag['r2']:.4f}")
        print(f"  Significant lags (p<0.05): {sig_lags if sig_lags else 'None'}")

    company_df = pd.DataFrame(company_results)
    company_df.to_csv(output_dir / 'regression_by_company_and_lag.csv', index=False)
    print(f"\nSaved: regression_by_company_and_lag.csv")

    # =========================================================================
    # 3. MULTIPLE REGRESSION WITH ALL LAGS
    # =========================================================================
    print("\n" + "=" * 70)
    print("3. MULTIPLE REGRESSION (All Lags as Features)")
    print("=" * 70)

    # Features: emp_change at t-0 through t-7
    feature_cols = ['emp_change_pct'] + [f'emp_lag_{i}' for i in range(1, max_lag + 1)]
    X_multi = df_clean[feature_cols].values
    y_multi = df_clean['price_return'].values

    # Sklearn model
    multi_model = LinearRegression()
    multi_model.fit(X_multi, y_multi)
    y_pred_multi = multi_model.predict(X_multi)

    r2_multi = r2_score(y_multi, y_pred_multi)
    cv_scores_multi = cross_val_score(multi_model, X_multi, y_multi, cv=5, scoring='r2')

    # Statsmodels for detailed output
    X_multi_sm = sm.add_constant(X_multi)
    sm_multi = sm.OLS(y_multi, X_multi_sm).fit()

    print(f"\nMultiple Regression Results (all lags t-0 to t-{max_lag}):")
    print(f"  R²: {r2_multi:.4f} | Adj R²: {sm_multi.rsquared_adj:.4f}")
    print(f"  CV R² (5-fold): {cv_scores_multi.mean():.4f} ± {cv_scores_multi.std():.4f}")
    print(f"  F-statistic: {sm_multi.fvalue:.4f} (p={sm_multi.f_pvalue:.4e})")
    print("\n  Coefficients:")
    print(f"    {'Feature':<20} {'Coef':>10} {'Std Err':>10} {'t':>10} {'P>|t|':>10} {'Sig':>5}")
    print(f"    {'-'*65}")
    print(f"    {'Intercept':<20} {sm_multi.params[0]:>10.4f} {sm_multi.bse[0]:>10.4f} {sm_multi.tvalues[0]:>10.4f} {sm_multi.pvalues[0]:>10.4f}")
    for i, col in enumerate(feature_cols):
        sig = "*" if sm_multi.pvalues[i+1] < 0.05 else ""
        print(f"    {col:<20} {sm_multi.params[i+1]:>10.4f} {sm_multi.bse[i+1]:>10.4f} {sm_multi.tvalues[i+1]:>10.4f} {sm_multi.pvalues[i+1]:>10.4f} {sig:>5}")

    # Save multiple regression summary
    multi_summary = pd.DataFrame({
        'feature': ['intercept'] + feature_cols,
        'coefficient': sm_multi.params,
        'std_error': sm_multi.bse,
        't_statistic': sm_multi.tvalues,
        'p_value': sm_multi.pvalues,
        'significant': sm_multi.pvalues < 0.05
    })
    multi_summary.to_csv(output_dir / 'regression_multiple_all_lags.csv', index=False)
    print(f"\nSaved: regression_multiple_all_lags.csv")

    # =========================================================================
    # 4. VISUALIZATIONS
    # =========================================================================
    print("\n" + "=" * 70)
    print("4. GENERATING VISUALIZATIONS")
    print("=" * 70)

    plt.style.use('seaborn-v0_8-whitegrid')

    # 4a. R² by lag (simple regression)
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    colors = ['#1a9850' if s else '#d73027' for s in simple_df['significant']]
    bars = ax1.bar(simple_df['lag'], simple_df['r2'], color=colors, edgecolor='black', alpha=0.7)
    ax1.set_xlabel('Lag (Months)')
    ax1.set_ylabel('R² Score')
    ax1.set_title('Simple Regression R² by Lag (Green = p<0.05)')
    ax1.set_xticks(range(0, max_lag + 1))
    for i, row in simple_df.iterrows():
        ax1.text(row['lag'], row['r2'] + 0.001, f"{row['r2']:.4f}", ha='center', fontsize=9)
    plt.tight_layout()
    fig1.savefig(output_dir / 'r2_by_lag.png', dpi=150)
    print("  Saved: r2_by_lag.png")

    # 4b. Coefficient by lag with confidence intervals
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    ax2.errorbar(simple_df['lag'], simple_df['coefficient'],
                 yerr=1.96 * simple_df['rmse'] / np.sqrt(len(df_clean)),
                 fmt='o-', capsize=5, capthick=2, markersize=8, linewidth=2)
    ax2.axhline(y=0, color='black', linestyle='--', linewidth=0.5)
    ax2.set_xlabel('Lag (Months)')
    ax2.set_ylabel('Regression Coefficient')
    ax2.set_title('Regression Coefficient by Lag (with 95% CI)')
    ax2.set_xticks(range(0, max_lag + 1))
    plt.tight_layout()
    fig2.savefig(output_dir / 'coefficient_by_lag.png', dpi=150)
    print("  Saved: coefficient_by_lag.png")

    # 4c. Heatmap of R² by company and lag
    fig3, ax3 = plt.subplots(figsize=(12, 8))
    pivot_r2 = company_df.pivot(index='symbol', columns='lag', values='r2')
    sns.heatmap(pivot_r2, annot=True, fmt='.3f', cmap='YlOrRd', ax=ax3,
                linewidths=0.5, cbar_kws={'label': 'R²'})
    ax3.set_title('R² by Company and Lag')
    ax3.set_xlabel('Lag (Months)')
    ax3.set_ylabel('Company')
    plt.tight_layout()
    fig3.savefig(output_dir / 'r2_heatmap_company_lag.png', dpi=150)
    print("  Saved: r2_heatmap_company_lag.png")

    # 4d. P-value heatmap by company and lag
    fig4, ax4 = plt.subplots(figsize=(12, 8))
    pivot_pval = company_df.pivot(index='symbol', columns='lag', values='p_value')
    sns.heatmap(pivot_pval, annot=True, fmt='.3f', cmap='RdYlGn_r', ax=ax4,
                linewidths=0.5, vmin=0, vmax=0.1, cbar_kws={'label': 'p-value'})
    ax4.set_title('P-value by Company and Lag (Green = Significant)')
    ax4.set_xlabel('Lag (Months)')
    ax4.set_ylabel('Company')
    plt.tight_layout()
    fig4.savefig(output_dir / 'pvalue_heatmap_company_lag.png', dpi=150)
    print("  Saved: pvalue_heatmap_company_lag.png")

    # 4e. Multiple regression coefficients
    fig5, ax5 = plt.subplots(figsize=(10, 6))
    coef_data = multi_summary[multi_summary['feature'] != 'intercept']
    colors = ['#1a9850' if s else '#d73027' for s in coef_data['significant']]
    ax5.barh(coef_data['feature'], coef_data['coefficient'], color=colors, edgecolor='black')
    ax5.axvline(x=0, color='black', linewidth=0.5)
    ax5.set_xlabel('Coefficient')
    ax5.set_title('Multiple Regression Coefficients (Green = p<0.05)')
    plt.tight_layout()
    fig5.savefig(output_dir / 'multiple_regression_coefficients.png', dpi=150)
    print("  Saved: multiple_regression_coefficients.png")

    # 4f. Actual vs Predicted scatter (multiple regression)
    fig6, ax6 = plt.subplots(figsize=(8, 8))
    ax6.scatter(y_multi, y_pred_multi, alpha=0.3, edgecolors='black', linewidth=0.3)
    ax6.plot([y_multi.min(), y_multi.max()], [y_multi.min(), y_multi.max()], 'r--', linewidth=2)
    ax6.set_xlabel('Actual Stock Return (%)')
    ax6.set_ylabel('Predicted Stock Return (%)')
    ax6.set_title(f'Multiple Regression: Actual vs Predicted (R²={r2_multi:.4f})')
    plt.tight_layout()
    fig6.savefig(output_dir / 'actual_vs_predicted.png', dpi=150)
    print("  Saved: actual_vs_predicted.png")

    plt.close('all')

    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print("\nSimple Regression - Best lag by R²:")
    best_simple = simple_df.loc[simple_df['r2'].idxmax()]
    print(f"  Lag {int(best_simple['lag'])}: R²={best_simple['r2']:.4f}, p={best_simple['p_value']:.4f}")

    print("\nSignificant relationships (p<0.05) in simple regressions:")
    sig_simple = simple_df[simple_df['significant']]
    if len(sig_simple) > 0:
        for _, row in sig_simple.iterrows():
            print(f"  Lag {int(row['lag'])}: coef={row['coefficient']:.4f}, p={row['p_value']:.4f}")
    else:
        print("  None")

    print(f"\nMultiple Regression (all lags):")
    print(f"  R²: {r2_multi:.4f} | Adj R²: {sm_multi.rsquared_adj:.4f}")
    sig_features = multi_summary[(multi_summary['significant']) & (multi_summary['feature'] != 'intercept')]
    if len(sig_features) > 0:
        print("  Significant features:")
        for _, row in sig_features.iterrows():
            print(f"    {row['feature']}: coef={row['coefficient']:.4f}, p={row['p_value']:.4f}")
    else:
        print("  No significant features")

    return simple_df, company_df, multi_summary


if __name__ == "__main__":
    base_path = Path(__file__).parent
    data_file = base_path / "data" / "prices_employment_merged.xlsx"
    output_dir = base_path / "data" / "regression_analysis"

    simple_df, company_df, multi_summary = run_regression_analysis(data_file, output_dir, max_lag=7)
