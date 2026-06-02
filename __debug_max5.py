"""Debug MAX5 difference specifically."""
import numpy as np
import pandas as pd

np.random.seed(42)
dates = pd.date_range('2020-01-01', periods=2, freq='D')
instruments = ['A', 'B', 'C']
idx = pd.MultiIndex.from_product([dates, instruments], names=['datetime', 'instrument'])

# Simulate MAX5 values that produce NaN after (x-1)**0.5
df = pd.DataFrame({
    'MAX5': [-0.2, 1.72, 4.77, 1.52, -1.14, 2.0],
}, index=idx)

print("Raw:")
print(df)

# Original approach
def _feature_norm_orig(x):
    x = x - x.median()
    x /= x.abs().median() * 1.4826
    x.where(x <= 3, 3 + (x - 3).div(x.max() - 3) * 0.5, inplace=True)
    x.where(x >= -3, -3 - (x + 3).div(x.min() + 3) * 0.5, inplace=True)
    x.fillna(0, inplace=True)
    return x

df_orig = df.copy()
df_orig[['MAX5']] = df_orig[['MAX5']].apply(lambda x: (x-1)**0.5).groupby(level='datetime', group_keys=False).apply(_feature_norm_orig)
print("\nOrig:")
print(df_orig)

# New approach
df_new = df.copy()
df_new['MAX5'] = (df_new['MAX5'] - 1) ** 0.5
print("\nAfter pre-transform:")
print(df_new)

def _norm_new(g):
    for c in ['MAX5']:
        col = g[c] - g[c].median()
        col = col / (col.abs().median() * 1.4826)
        col = col.where(col <= 3, 3 + (col - 3).div(col.max() - 3) * 0.5)
        col = col.where(col >= -3, -3 - (col + 3).div(col.min() + 3) * 0.5)
        col = col.fillna(0)
        g[c] = col
    return g

df_new[['MAX5']] = df_new[['MAX5']].groupby(level='datetime', group_keys=False).apply(_norm_new)
print("\nNew:")
print(df_new)

print("\nDiff:")
print(df_orig - df_new)

# Let me also check what happens when pre-transform is applied inside groupby
print("\n--- Checking with more columns ---")
df2 = pd.DataFrame({
    'MAX5': [-0.2, 1.72, 4.77, 1.52, -1.14, 2.0],
    'KMID': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
}, index=idx)

df2_orig = df2.copy()
df2_orig[['MAX5']] = df2_orig[['MAX5']].apply(lambda x: (x-1)**0.5).groupby(level='datetime', group_keys=False).apply(_feature_norm_orig)
df2_orig[['KMID']] = df2_orig[['KMID']].groupby(level='datetime', group_keys=False).apply(_feature_norm_orig)
print("\nOrig (separate groupby for MAX5 and KMID):")
print(df2_orig)

# What if we process both in the same groupby?
df2_new = df2.copy()
df2_new['MAX5'] = (df2_new['MAX5'] - 1) ** 0.5

def _norm_both(g):
    for c in ['MAX5', 'KMID']:
        col = g[c] - g[c].median()
        col = col / (col.abs().median() * 1.4826)
        col = col.where(col <= 3, 3 + (col - 3).div(col.max() - 3) * 0.5)
        col = col.where(col >= -3, -3 - (col + 3).div(col.min() + 3) * 0.5)
        col = col.fillna(0)
        g[c] = col
    return g

df2_new[['MAX5', 'KMID']] = df2_new[['MAX5', 'KMID']].groupby(level='datetime', group_keys=False).apply(_norm_both)
print("\nNew (same groupby for MAX5 and KMID):")
print(df2_new)

print("\nDiff:")
print(df2_orig - df2_new)
