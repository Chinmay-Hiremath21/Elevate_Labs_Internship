import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("cleaned_customers.csv")

# Elbow Method to find optimal K
sse = []
K_range = range(1, 11)
for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(df)
    sse.append(kmeans.inertia_)

plt.figure(figsize=(8, 4))
plt.plot(K_range, sse, 'bo-')
plt.xlabel('Number of clusters (K)')
plt.ylabel('SSE (Inertia)')
plt.title('Elbow Method For Optimal K')
plt.grid(True)
plt.tight_layout()
plt.show()

# Fit KMeans with optimal K (assume K=5 for example)
k_optimal = 5
kmeans = KMeans(n_clusters=k_optimal, random_state=42, n_init=10)
df['Cluster'] = kmeans.fit_predict(df)

# Evaluate with Silhouette Score
score = silhouette_score(df.drop('Cluster', axis=1), df['Cluster'])
print("Silhouette Score:", round(score, 3))

# Visualize clusters using PCA
pca = PCA(n_components=2)
reduced = pca.fit_transform(df.drop('Cluster', axis=1))
df_pca = pd.DataFrame(reduced, columns=["PC1", "PC2"])
df_pca["Cluster"] = df["Cluster"]

plt.figure(figsize=(8, 6))
sns.scatterplot(data=df_pca, x="PC1", y="PC2", hue="Cluster", palette="tab10", s=60)
plt.title("K-Means Clusters Visualized with PCA")
plt.tight_layout()
plt.show()
