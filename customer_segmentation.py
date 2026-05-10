import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Load dataset
data = pd.read_csv("Mall_Customers.csv")

# Show first 5 rows
print(data.head())

# Select required columns
X = data[['Annual Income (k$)', 'Spending Score (1-100)']]

# Elbow Method
wcss = []

for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

# Plot Elbow Graph
plt.plot(range(1, 11), wcss, marker='o')
plt.title('Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()

# Train Model
kmeans = KMeans(n_clusters=5, init='k-means++', random_state=42)

# Predict clusters
y_kmeans = kmeans.fit_predict(X)

# Visualize clusters
plt.figure(figsize=(8,6))

plt.scatter(X.iloc[y_kmeans == 0, 0],
            X.iloc[y_kmeans == 0, 1],
            s=100, c='red', label='Cluster 1')

plt.scatter(X.iloc[y_kmeans == 1, 0],
            X.iloc[y_kmeans == 1, 1],
            s=100, c='blue', label='Cluster 2')

plt.scatter(X.iloc[y_kmeans == 2, 0],
            X.iloc[y_kmeans == 2, 1],
            s=100, c='green', label='Cluster 3')

plt.scatter(X.iloc[y_kmeans == 3, 0],
            X.iloc[y_kmeans == 3, 1],
            s=100, c='cyan', label='Cluster 4')

plt.scatter(X.iloc[y_kmeans == 4, 0],
            X.iloc[y_kmeans == 4, 1],
            s=100, c='magenta', label='Cluster 5')

# Plot Centroids
plt.scatter(kmeans.cluster_centers_[:, 0],
            kmeans.cluster_centers_[:, 1],
            s=300,
            c='yellow',
            label='Centroids')

plt.title('Customer Segmentation')
plt.xlabel('Annual Income')
plt.ylabel('Spending Score')
plt.legend()
plt.show()