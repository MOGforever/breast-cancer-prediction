import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.datasets import load_breast_cancer
from sklearn.neighbors import NearestNeighbors

def prepare_data_for_clustering(dataset_loader):
    """Load and prepare dataset for clustering."""
    try:
        data = dataset_loader()
        X = data.data
        y = data.target
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
        print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
        print(f"Total explained variance: {sum(pca.explained_variance_ratio_):.4f}")
        return X, y, X_scaled, X_pca, data.feature_names, data.target_names
    except Exception as e:
        print(f"Error preparing data: {e}")
        return None

def plot_clustering_results(X_pca, labels, title, filename, ylabel='Second Principal Component'):
    """Plot clustering results and save to file."""
    plt.figure(figsize=(10, 8))
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis', alpha=0.8)
    plt.title(title)
    plt.xlabel('First Principal Component')
    plt.ylabel(ylabel)
    plt.colorbar(label='Cluster')
    plt.savefig(filename)
    plt.close()

# Main clustering task
print("===== TASK II: CLUSTERING WITH BREAST CANCER DATASET =====")
result = prepare_data_for_clustering(load_breast_cancer)
if not result:
    exit()

X, y, X_scaled, X_pca, feature_names, target_names = result

# Visualize true labels
plot_clustering_results(X_pca, y, 'PCA of Breast Cancer Dataset with True Labels', 'true_labels.png')

# 1. K-Means Clustering
print("\nPerforming K-Means clustering...")
inertia = []
silhouette = []
k_range = range(2, 11)

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)
    silhouette.append(silhouette_score(X_scaled, kmeans.labels_))

# Plot Elbow and Silhouette methods
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(k_range, inertia, 'bo-')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal K')
plt.grid(True)
plt.subplot(1, 2, 2)
plt.plot(k_range, silhouette, 'ro-')
plt.xlabel('Number of clusters')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Method for Optimal K')
plt.grid(True)
plt.savefig('kmeans_metrics.png')
plt.close()

# Apply K-Means with k=2
kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
y_kmeans = kmeans.fit_predict(X_scaled)
plot_clustering_results(X_pca, y_kmeans, 'K-Means Clustering (K=2)', 'kmeans_clustering.png')

# Evaluate K-Means
kmeans_ari = adjusted_rand_score(y, y_kmeans)
kmeans_silhouette = silhouette_score(X_scaled, y_kmeans)
print(f"K-Means - Adjusted Rand Index: {kmeans_ari:.4f}")
print(f"K-Means - Silhouette Score: {kmeans_silhouette:.4f}")

# 2. DBSCAN Clustering
print("\nPerforming DBSCAN clustering...")
nn = NearestNeighbors(n_neighbors=2)
nn.fit(X_scaled)
distances, indices = nn.kneighbors(X_scaled)
distances = np.sort(distances[:, 1])

# Automate eps selection (knee point approximation)
eps = np.percentile(distances, 75)  # Use 75th percentile as a heuristic
print(f"Selected eps: {eps:.4f}")

dbscan = DBSCAN(eps=eps, min_samples=5)
y_dbscan = dbscan.fit_predict(X_scaled)

n_clusters = len(set(y_dbscan)) - (1 if -1 in y_dbscan else 0)
n_noise = list(y_dbscan).count(-1)
print(f"DBSCAN - Number of clusters: {n_clusters}")
print(f"DBSCAN - Number of noise points: {n_noise} ({n_noise/len(y_dbscan):.2%})")

plot_clustering_results(X_pca, y_dbscan, 'DBSCAN Clustering', 'dbscan_clustering.png', ylabel='Cluster (-1 is noise)')

# Evaluate DBSCAN
if n_clusters > 1 and n_noise < len(y_dbscan):
    mask = y_dbscan != -1
    dbscan_ari = adjusted_rand_score(y[mask], y_dbscan[mask])
    try:
        dbscan_silhouette = silhouette_score(X_scaled[mask], y_dbscan[mask])
        print(f"DBSCAN - Adjusted Rand Index (excluding noise): {dbscan_ari:.4f}")
        print(f"DBSCAN - Silhouette Score (excluding noise): {dbscan_silhouette:.4f}")
    except:
        print("Could not compute silhouette score for DBSCAN")
else:
    print("DBSCAN did not find multiple clusters, skipping evaluation")

# 3. Hierarchical Clustering
print("\nPerforming Hierarchical clustering...")
linkages = ['ward', 'complete', 'average']
results = {}

for i, linkage in enumerate(linkages):
    agg = AgglomerativeClustering(n_clusters=2, linkage=linkage)
    y_agg = agg.fit_predict(X_scaled)
    
    agg_ari = adjusted_rand_score(y, y_agg)
    agg_silhouette = silhouette_score(X_scaled, y_agg)
    results[linkage] = (agg_ari, agg_silhouette)
    
    print(f"Hierarchical ({linkage}) - Adjusted Rand Index: {agg_ari:.4f}")
    print(f"Hierarchical ({linkage}) - Silhouette Score: {agg_silhouette:.4f}")
    
    plot_clustering_results(X_pca, y_agg, f'Hierarchical Clustering ({linkage})', f'hierarchical_{linkage}.png')

# Comparison plot
plt.figure(figsize=(8, 6))
methods = list(results.keys())
ari_scores = [results[m][0] for m in methods]
silhouette_scores = [results[m][1] for m in methods]
x = np.arange(len(methods))
width = 0.35
plt.bar(x - width/2, ari_scores, width, label='ARI')
plt.bar(x + width/2, silhouette_scores, width, label='Silhouette')
plt.xticks(x, methods)
plt.ylabel('Score')
plt.title('Clustering Performance Comparison')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('clustering_comparison.png')
plt.close()

# Summary
print("\n===== CLUSTERING SUMMARY =====")
print("K-Means clustering:")
print(f"- Adjusted Rand Index: {kmeans_ari:.4f}")
print(f"- Silhouette Score: {kmeans_silhouette:.4f}")

print("\nHierarchical clustering performance:")
for linkage, (ari, silhouette) in results.items():
    print(f"- {linkage.capitalize()}: ARI={ari:.4f}, Silhouette={silhouette:.4f}")

best_method = max(results.items(), key=lambda x: x[1][0])
print(f"\nBest hierarchical method based on ARI: {best_method[0]} with score {best_method[1][0]:.4f}")

if kmeans_ari > best_method[1][0]:
    print("K-Means performed better than all hierarchical methods")
else:
    print(f"{best_method[0].capitalize()} hierarchical clustering performed better than K-Means")