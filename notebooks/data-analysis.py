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
import plotly.express as px

# %%
# %matplotlib inline
load_dotenv()
current_path = Path.cwd()

# %% [markdown]
# # Data loading

# %%
pnl_df = pd.read_excel(current_path / ".." / "data" / "P&L.xlsx", index_col=0, skiprows=[1])

# %%
pnl_df

# %%
pnl_df.index

# %% [markdown]
# # Data exploration

# %%
kpis = ["Staff Costs", "Total OPEX", "EBITDA"]
px.line(pnl_df.loc[kpis].T)

# %%
px.line((pnl_df.loc[kpis].T - pnl_df.loc[kpis].T.shift(12)).dropna())

# %%
import numpy as np

ebitda_calc = (
    pnl_df.loc['Total Revenue']
  - pnl_df.loc['Cost of Services']
  - pnl_df.loc['Credit Card Fees']
  - pnl_df.loc['Total OPEX']
)
diff = pnl_df.loc['EBITDA'] - ebitda_calc
ok = np.allclose(diff.values, 0, equal_nan=True)
ok

# %%
pnl_df.loc['EBITDA'] - ebitda_calc

# %%
import pandas as pd

headers = {'Transactions','Total Revenue','Cost of Services','Credit Card Fees','Gross Profit'}
opex = {'Business Operations','Customer Service','Marketing Expenses','Staff Costs','Other OPEX','Total OPEX'}

rows = []
current = None
for lab in pnl_df.index:
    if lab in headers:
        current = lab
        rows.append((lab, 'TOTAL'))     # la ligne d’agrégat si elle existe
    elif lab in opex:
        rows.append(('OPEX', lab))
    elif lab == 'EBITDA':
        rows.append(('P&L', 'EBITDA'))
    else:
        rows.append((current, lab))     # lignes segmentées

df2 = pnl_df.copy()
df2.index = pd.MultiIndex.from_tuples(rows, names=['Section','Item'])

segments = ['Enterprise','Mid Market','A1','Marketplaces','Small+','Small','Local','Editions','Other']
rev_tot  = df2.loc[('Total Revenue', segments)].sum()          # somme des lignes segments
cos_tot  = df2.loc[('Cost of Services', segments)].sum()
fees_tot = df2.loc[('Credit Card Fees', segments)].sum()
gross_from_seg = rev_tot - cos_tot - fees_tot

opex_tot = df2.loc[('OPEX','Total OPEX')]
ebitda_from_seg = gross_from_seg - opex_tot


# %%
