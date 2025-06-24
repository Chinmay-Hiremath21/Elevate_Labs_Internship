import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.figure_factory as ff

df = pd.read_csv("House_Rent_Dataset_Cleaned.csv")

summary_stats = df.describe()

numeric_cols = ['Rent', 'Size', 'Floor', 'Bathroom']
df[numeric_cols].hist(bins=20, figsize=(10, 6), color='skyblue', edgecolor='black')
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
for i, col in enumerate(numeric_cols):
    plt.subplot(2, 2, i+1)
    sns.boxplot(y=df[col], color='orange')
    plt.title(f'Boxplot - {col}')
plt.tight_layout()
plt.show()

sns.pairplot(df[numeric_cols])
plt.show()

plt.figure(figsize=(8, 6))
sns.heatmap(df[numeric_cols].corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()

fig = px.histogram(df, x='Furnishing Status', color='Rent', title='Rent Distribution by Furnishing Status')
fig.show()

fig = px.histogram(df, x='Tenant Preferred', color='Rent', title='Rent Distribution by Tenant Type')
fig.show()

fig = px.scatter(df, x='Size', y='Rent', color='Bathroom', title="Rent vs Size (Colored by Bathroom)")
fig.show()

furnishing_map = {0: "Furnished", 1: "Semi-Furnished", 2: "Unfurnished"}
df['Furnishing_Label'] = df['Furnishing Status'].map(furnishing_map)

groups = []
labels = []

for label in df['Furnishing_Label'].unique():
    group = df[df['Furnishing_Label'] == label]['Rent']
    groups.append(group)
    labels.append(label)

fig = ff.create_distplot(groups, group_labels=labels, show_hist=False)
fig.update_layout(title_text='Rent Distribution by Furnishing Status')
fig.show()
