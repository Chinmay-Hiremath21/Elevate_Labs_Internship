import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler

df = pd.read_csv('House_Rent_Dataset.csv')

df['Floor'] = df['Floor'].str.extract(r'(\d+)').astype(float)
df['Posted On'] = pd.to_datetime(df['Posted On'])
df['Area Locality'] = df['Area Locality'].str.strip()
df.fillna({'Floor': df['Floor'].median()}, inplace=True)

categorical_cols = ['Area Type', 'Area Locality', 'City', 'Furnishing Status', 'Tenant Preferred', 'Point of Contact']
le = LabelEncoder()
for col in categorical_cols:
    df[col] = le.fit_transform(df[col])

num_cols = ['Rent', 'Size', 'Floor', 'Bathroom']
scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])

plt.figure(figsize=(12, 8))
sns.boxplot(data=df[num_cols])
plt.xticks(rotation=45)
plt.show()

def remove_outliers_iqr(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    return data[(data[column] >= Q1 - 1.5 * IQR) & (data[column] <= Q3 + 1.5 * IQR)]

for col in num_cols:
    df = remove_outliers_iqr(df, col)

df.to_csv('House_Rent_Dataset_Cleaned.csv', index=False)
