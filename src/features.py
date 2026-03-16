"""
features.py
-----------
Feature engineering functions for the PowerCo churn prediction project.
Extracted from 02_feature_engineering.ipynb so the same logic can be reused
in a scoring pipeline without copying notebook code around.

Each function takes a DataFrame, adds new columns, and returns a copy.
Nothing is modified in-place.
"""

import pandas as pd
import numpy as np


def add_date_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pull temporal signals out of contract date columns.

    Uses date_end as the observation reference point for all calculations.
    This avoids leaking future information into features — a subtle but
    important distinction when building a model that will score live accounts.

    All date columns must be parsed as datetime before calling this function.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain: date_activ, date_end, date_modif_prod, date_renewal
    """
    df = df.copy()

    # How long has this customer been active?
    # Longer tenure generally means lower churn risk — more inertia built up.
    df['tenure_months'] = (
        (df['date_end'] - df['date_activ']).dt.days / 30.44
    ).round(1)

    # Time since the last product modification.
    # A recent change can mean the customer asked for something different,
    # or that the company tried to retain them. Either way, worth capturing.
    df['months_since_last_modif'] = (
        (df['date_end'] - df['date_modif_prod']).dt.days / 30.44
    ).round(1)

    # Gap between renewal date and contract end.
    # Renewing 12 months early is a different risk profile from renewing with two weeks left.
    df['renewal_to_end_months'] = (
        (df['date_end'] - df['date_renewal']).dt.days / 30.44
    ).round(1)

    # Which month does the contract end? Churn decisions cluster around certain months.
    df['contract_end_month'] = df['date_end'].dt.month

    # Was the product modified in the last six months of the contract?
    # Binary flag — easier for the model to use than the raw month count.
    df['modified_recently'] = (df['months_since_last_modif'] <= 6).astype(int)

    # Which quarter did the customer sign up in?
    df['activation_quarter'] = df['date_activ'].dt.quarter

    return df


def add_consumption_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build behavioural signals from energy consumption data.

    The most useful one here is cons_ratio_last_to_avg — a customer whose
    last month of usage dropped sharply below their annual average may have
    already moved some load to a competitor. Not definitive, but worth catching.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain: cons_12m, cons_gas_12m, cons_last_month, forecast_cons_12m
    """
    df = df.copy()

    df['avg_monthly_cons'] = (df['cons_12m'] / 12).round(2)

    # Last month relative to annual average.
    # Capped at 10 so a single spike month does not dominate the signal.
    df['cons_ratio_last_to_avg'] = (
        df['cons_last_month'] / df['avg_monthly_cons'].replace(0, np.nan)
    ).clip(upper=10).fillna(0).round(4)

    # Zero consumption last month — possible explanation: the customer already switched.
    df['zero_consumption_last_month'] = (df['cons_last_month'] == 0).astype(int)

    # Gas as a share of total consumption.
    # Bundle customers face higher switching friction and tend to churn less.
    total_cons = df['cons_12m'] + df['cons_gas_12m']
    df['gas_share_of_consumption'] = (
        df['cons_gas_12m'] / total_cons.replace(0, np.nan)
    ).fillna(0).round(4)

    # Log-transformed annual consumption.
    # cons_12m spans several orders of magnitude — a handful of industrial customers
    # would otherwise dominate any distance-based calculation in the model.
    df['log_cons_12m'] = np.log1p(df['cons_12m']).round(4)

    # Forecast vs actual consumption ratio.
    # A large gap suggests the customer's usage changed unpredictably during the year.
    df['forecast_vs_actual_cons'] = (
        df['forecast_cons_12m'] / df['cons_12m'].replace(0, np.nan)
    ).clip(upper=20).fillna(0).round(4)

    return df


def add_price_variation_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compress 18 raw price-change columns into a handful of usable signals.

    Feeding all 18 columns to the model individually risks multicollinearity
    and makes feature importance harder to interpret. These composites capture
    the same information more cleanly.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain the var_year_price_* and var_6m_price_* columns.
    """
    df = df.copy()

    year_cols = [
        'var_year_price_off_peak_var', 'var_year_price_peak_var',
        'var_year_price_mid_peak_var', 'var_year_price_off_peak_fix',
        'var_year_price_peak_fix', 'var_year_price_mid_peak_fix'
    ]
    six_m_cols = [
        'var_6m_price_off_peak_var', 'var_6m_price_peak_var',
        'var_6m_price_mid_peak_var', 'var_6m_price_off_peak_fix',
        'var_6m_price_peak_fix', 'var_6m_price_mid_peak_fix'
    ]

    # Total absolute price movement — a single volatility score per customer per window.
    df['total_abs_price_change_year'] = df[year_cols].abs().sum(axis=1).round(6)
    df['total_abs_price_change_6m']   = df[six_m_cols].abs().sum(axis=1).round(6)

    # Direction of off-peak price change: +1 up, -1 down, 0 flat.
    df['offpeak_price_direction_year'] = np.sign(df['var_year_price_off_peak_var'])
    df['offpeak_price_direction_6m']   = np.sign(df['var_6m_price_off_peak_var'])

    # Is price change accelerating? A ratio above 0.5 means more than half
    # the annual change happened in the last six months.
    df['price_acceleration_ratio'] = (
        df['var_6m_price_off_peak_var'].abs() /
        df['var_year_price_off_peak_var'].abs().replace(0, np.nan)
    ).clip(upper=5).fillna(0).round(4)

    # Binary: did the off-peak price increase at all over the year?
    df['price_increased_year'] = (df['var_year_price_off_peak_var'] > 0).astype(int)

    return df


def add_financial_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build profitability and discount signals from financial columns.

    These turned out to be the strongest predictors in the model.
    net_margin_per_kwh (importance 0.118) and margin_gross_pow_ele (0.106)
    both ranked above every price variable. Low-margin customers are also
    the customers most likely to leave.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain margin and forecast price columns.
    """
    df = df.copy()

    # Revenue per unit consumed. Very low or negative means the contract is losing money.
    df['net_margin_per_kwh'] = (
        df['net_margin'] / df['cons_12m'].replace(0, np.nan)
    ).fillna(0).round(6)

    # Gap between gross and net margin on electricity.
    df['margin_diff_gross_net'] = (
        df['margin_gross_pow_ele'] - df['margin_net_pow_ele']
    ).round(4)

    # Discount as a fraction of forecast off-peak price.
    # A large discount may mean this customer was already identified as at-risk —
    # the discount itself then becomes a churn signal.
    df['discount_ratio'] = (
        df['forecast_discount_energy'] /
        df['forecast_price_energy_off_peak'].replace(0, np.nan)
    ).fillna(0).clip(upper=1).round(4)

    df['has_discount'] = (df['forecast_discount_energy'] > 0).astype(int)

    df['pow_per_gross_margin'] = (
        df['pow_max'] / df['margin_gross_pow_ele'].replace(0, np.nan)
    ).fillna(0).clip(upper=500).round(4)

    # Meter rent as a share of total forecast cost.
    df['meter_rent_ratio'] = (
        df['forecast_meter_rent_12m'] /
        (df['forecast_cons_12m'] + df['forecast_meter_rent_12m']).replace(0, np.nan)
    ).fillna(0).round(4)

    return df


def compute_offpeak_dec_jan_diff(price_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate the change in off-peak price between January and December
    for each customer.

    The idea: a large swing between the start and end of the year signals
    pricing instability. The hypothesis held up in the model, though margin
    features ultimately ranked higher.

    Parameters
    ----------
    price_df : pd.DataFrame
        Must contain: id, price_date (datetime), price_off_peak_var, price_off_peak_fix

    Returns
    -------
    pd.DataFrame with columns: id, offpeak_diff_dec_january_energy, offpeak_diff_dec_january_power
    """
    monthly = (
        price_df
        .groupby(['id', 'price_date'])
        .agg({'price_off_peak_var': 'mean', 'price_off_peak_fix': 'mean'})
        .reset_index()
    )

    jan = monthly.groupby('id').first().reset_index()
    dec = monthly.groupby('id').last().reset_index()

    diff = pd.merge(
        dec.rename(columns={'price_off_peak_var': 'dec_var', 'price_off_peak_fix': 'dec_fix'}),
        jan.drop(columns='price_date'),
        on='id'
    )

    diff['offpeak_diff_dec_january_energy'] = diff['dec_var'] - diff['price_off_peak_var']
    diff['offpeak_diff_dec_january_power']  = diff['dec_fix'] - diff['price_off_peak_fix']

    return diff[['id', 'offpeak_diff_dec_january_energy', 'offpeak_diff_dec_january_power']]


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert string categorical columns to integer codes.

    pd.factorize() assigns integers in order of first appearance in the data.
    For tree-based models this is fine — Random Forest does not treat the codes
    as ordered. For linear models, one-hot encoding would be needed instead.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain: channel_sales, origin_up, has_gas
    """
    df = df.copy()

    df['channel_sales_encoded'], _ = pd.factorize(df['channel_sales'])
    df['origin_up_encoded'],     _ = pd.factorize(df['origin_up'])
    df['has_gas_binary']           = (df['has_gas'] == 't').astype(int)

    return df
