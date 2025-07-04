import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler

df = pd.read_csv("breast-cancer.csv")

# Drop completely empty columns
df.dropna(axis=1, how='all', inplace=True)

# Drop duplicate rows
df.drop_duplicates(inplace=True)

# Drop non-informative columns like 'id' or constant columns
df = df.loc[:, df.nunique() > 1]
df = df.loc[:, ~df.columns.str.lower().isin(['id'])]

# Identify categorical and numeric columns
categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

# Encode categorical columns
le = LabelEncoder()
for col in categorical_cols:
    df[col] = le.fit_transform(df[col].astype(str))

# Fill missing numeric values with median (if any)
df[numeric_cols] = df[numeric_cols].apply(lambda x: x.fillna(x.median()) if x.isnull().any() else x)

# Standardize numeric columns only
scaler = StandardScaler()
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

# Save cleaned dataset
df.to_csv("cleaned_breast_cancer.csv", index=False)
