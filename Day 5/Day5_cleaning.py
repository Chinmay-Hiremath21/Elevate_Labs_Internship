import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer

df = pd.read_csv("heart.csv")

# Drop duplicates if any
if df.duplicated().any():
    df = df.drop_duplicates()

# Drop rows with all null values
if df.isnull().all(axis=1).any():
    df = df.dropna(how='all')

# Strip column names and standardize
df.columns = df.columns.str.strip()
df.columns = [col.lower().replace(' ', '_') for col in df.columns]

# Separate column types
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

# Handle missing numeric values
if numeric_cols:
    if df[numeric_cols].isnull().any().any():
        imputer_num = SimpleImputer(strategy='mean')
        df[numeric_cols] = imputer_num.fit_transform(df[numeric_cols])

# Handle missing categorical values
if categorical_cols:
    if df[categorical_cols].isnull().any().any():
        imputer_cat = SimpleImputer(strategy='most_frequent')
        df[categorical_cols] = imputer_cat.fit_transform(df[categorical_cols])

# Encode categorical variables
encoded_cols = []
for col in categorical_cols:
    if df[col].nunique() <= 10:
        df[col] = LabelEncoder().fit_transform(df[col])
    else:
        df = pd.get_dummies(df, columns=[col], drop_first=True)
        encoded_cols.append(col)

# Feature scaling
if numeric_cols:
    scaler = StandardScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

# Outlier removal using IQR
if numeric_cols:
    Q1 = df[numeric_cols].quantile(0.25)
    Q3 = df[numeric_cols].quantile(0.75)
    IQR = Q3 - Q1
    mask = ~((df[numeric_cols] < (Q1 - 1.5 * IQR)) | (df[numeric_cols] > (Q3 + 1.5 * IQR))).any(axis=1)
    if mask.sum() < len(df):
        df = df[mask]

# Save cleaned data
df.to_csv("cleaned_heart.csv", index=False)
