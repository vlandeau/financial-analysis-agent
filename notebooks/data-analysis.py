# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
import pandas as pd
from pathlib import Path

# %%
current_path = Path.cwd()
data_path = current_path / ".." / "data"

# %% [markdown]
# # Data loading

# %%
pnl_df = pd.read_exgcel(data_path / "P&L.xlsx", skiprows=[1])

# %%
pnl_df


# %%
def extract_kpis_df(first_row_index: int = 0, last_row_index: int = len(pnl_df)) -> pd.DataFrame:
    kpis_df = pnl_df[first_row_index:last_row_index].set_index("Unnamed: 0").T
    kpis_df.columns.name = None
    kpis_df.index = pd.to_datetime(kpis_df.index, format='%b %y')
    return kpis_df.fillna(0)


# %%
transactions_df = extract_kpis_df(last_row_index=10)
transactions_df

# %%
revenue_df = extract_kpis_df(10, 20)
revenue_df

# %%
services_cost_df = extract_kpis_df(20, 30)
services_cost_df

# %%
credit_card_fees_df = extract_kpis_df(30, 40)
credit_card_fees_df

# %%
gross_profit_df = extract_kpis_df(40, 50)
gross_profit_df

# %%
opex_df = extract_kpis_df(first_row_index=50, last_row_index=56)
opex_df

# %%
ebitda_df = extract_kpis_df(first_row_index=56, last_row_index=57)
ebitda_df

# %% [markdown]
# # Data coherence check

# %%
transactions = "Transactions"
segments = list(set(transactions_df.columns) - {transactions})
segments

# %%
epsilon = 0.000001
assert (transactions_df[transactions] - transactions_df[segments].sum(axis=1) < epsilon).all()

# %%
total_revenue = "Total Revenue"
assert (revenue_df[total_revenue] - revenue_df[segments].sum(axis=1) < epsilon).all()

# %%
cost_of_services = "Cost of Services"
assert (services_cost_df[cost_of_services] - services_cost_df[segments].sum(axis=1) < epsilon).all()

# %%
credit_card_fees = "Credit Card Fees"
assert (credit_card_fees_df[credit_card_fees].fillna(0) - credit_card_fees_df[segments].fillna(0).sum(axis=1) < epsilon).all()

# %%
gross_profit = "Gross Profit"
assert (gross_profit_df[gross_profit] - gross_profit_df[segments].sum(axis=1) < epsilon).all()

# %%
total_opex = "Total OPEX"
opex_dimensions = list(set(opex_df.columns) - {total_opex})
assert (opex_df[total_opex] - opex_df[opex_dimensions].sum(axis=1) < epsilon).all()

# %%
ebitda = "EBITDA"
estimated_ebitda_df = gross_profit_df[gross_profit] + opex_df[total_opex]
assert (ebitda_df[ebitda] - estimated_ebitda_df < epsilon).all()


# %% [markdown]
# # Aggregation per quarter

# %%
def aggregate_per_quarter(df: pd.DataFrame) -> pd.DataFrame:
    df_sampled_per_quarter = df.resample('QE', label='right').sum()
    return df_sampled_per_quarter


# %%
transactions_per_quarter_df = aggregate_per_quarter(transactions_df)
revenue_per_quarter_df = aggregate_per_quarter(revenue_df)
cost_of_services_per_quarter_df = aggregate_per_quarter(services_cost_df)
credit_card_fees_per_quarter_df = aggregate_per_quarter(credit_card_fees_df)
gross_profit_per_quarter_df = aggregate_per_quarter(gross_profit_df)
opex_per_quarter_df = aggregate_per_quarter(opex_df)
ebitda_per_quarter_df = aggregate_per_quarter(ebitda_df)

# %%
output_data_path = data_path / "output"

transactions_per_quarter_df.to_parquet(output_data_path / "transactions.parquet")
revenue_per_quarter_df.to_parquet(output_data_path / "revenue.parquet")
cost_of_services_per_quarter_df.to_parquet(output_data_path / "cost_of_services.parquet")
credit_card_fees_per_quarter_df.to_parquet(output_data_path / "credit_card_fees.parquet")
gross_profit_per_quarter_df.to_parquet(output_data_path / "gross_profit.parquet")
opex_per_quarter_df.to_parquet(output_data_path / "opex.parquet")
ebitda_per_quarter_df.to_parquet(output_data_path / "ebitda.parquet")

# %%
