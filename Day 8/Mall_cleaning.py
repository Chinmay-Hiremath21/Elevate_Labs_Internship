import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from scipy.stats import zscore

df = pd.read_csv("Mall_Customers.csv")
df = df.dropna(axis=1, how='all')
df = df.drop_duplicates()

for col in df.columns:
    if col.lower() == 'customerid' or df[col].nunique() == 1:
        df = df.drop(columns=[col])

categorical_cols = df.select_dtypes(include=["object", "category"]).columns
numeric_cols = df.select_dtypes(include=[np.number]).columns

for col in categorical_cols:
    df[col] = LabelEncoder().fit_transform(df[col])

for col in numeric_cols:
    if df[col].isnull().sum() > 0:
        df[col] = df[col].fillna(df[col].median())

z_scores = np.abs(zscore(df[numeric_cols]))
df = df[(z_scores < 3).all(axis=1)]

scaler = StandardScaler()
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

df.to_csv("cleaned_customers.csv", index=False)
