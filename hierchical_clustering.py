import pandas as pd
import json

import numpy as np
from sklearn.metrics import pairwise_distances
from sklearn.metrics import silhouette_samples
import random

def find_representative_embeddings(X_2d, labels, method='medoid', random_state=42):
    np.random.seed(random_state)
    random.seed(random_state)
    
    n_clusters = len(set(labels))
    representative_indices = {}
    
    for cluster in range(n_clusters):
        cluster_indices = np.where(labels == cluster)[0]
        
        if len(cluster_indices) == 0:
            continue
            
        cluster_points = X_2d[cluster_indices]
        
        if method == 'medoid':
            distances = pairwise_distances(cluster_points)
            total_distances = distances.sum(axis=1)
            medoid_idx = np.argmin(total_distances)
            representative_indices[cluster] = int(cluster_indices[medoid_idx])
            
        elif method == 'density':
            distances = pairwise_distances(cluster_points)
            avg_distance = np.mean(distances) / 2
            
            neighbor_counts = (distances < avg_distance).sum(axis=1) - 1
            density_idx = np.argmax(neighbor_counts)
            representative_indices[cluster] = int(cluster_indices[density_idx])
            
        elif method == 'silhouette':
            if len(set(labels)) <= 1 or len(cluster_indices) <= 1:
                representative_indices[cluster] = int(cluster_indices[0])
            else:
                silhouette_vals = silhouette_samples(X_2d, labels)
                cluster_silhouette = silhouette_vals[cluster_indices]
                silhouette_idx = np.argmax(cluster_silhouette)
                representative_indices[cluster] = int(cluster_indices[silhouette_idx])
                
        elif method == 'random':
            random_idx = random.choice(range(len(cluster_indices)))
            representative_indices[cluster] = int(cluster_indices[random_idx])
            
        elif method == 'frequency':
            distances = pairwise_distances(cluster_points)
            similarity_threshold = np.percentile(distances, 25)
            similar_pairs = (distances < similarity_threshold).sum(axis=1) - 1
            frequency_idx = np.argmax(similar_pairs)
            representative_indices[cluster] = int(cluster_indices[frequency_idx])
            
    return representative_indices

def visualize_clusters_with_representatives(X_2d, labels, representative_indices, method, pca, 
                                          cluster_labels=None, title=None, save_path=None):

    import matplotlib.pyplot as plt
    import seaborn as sns
    
    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_context("notebook", font_scale=1.2)
    
    n_clusters = len(set(labels))
    
    if cluster_labels is None:
        cluster_labels = {i: f'Cluster {i}' for i in range(n_clusters)}
    
    if title is None:
        title = f"Hierarchical Clustering with Representative Embeddings (XSTest)"
    
    colors = sns.color_palette("husl", n_colors=n_clusters)
    
    plt.figure(figsize=(14, 10))
    
    for i in range(n_clusters):
        plt.scatter(
            X_2d[labels == i, 0], 
            X_2d[labels == i, 1],
            c=[colors[i]],
            label=cluster_labels[i],
            s=70,
            alpha=0.75,
            edgecolors='w',
            linewidth=0.5
        )
    
    for i, rep_idx in representative_indices.items():
        plt.scatter(
            X_2d[rep_idx, 0],
            X_2d[rep_idx, 1],
            s=450,
            c=[colors[i]],
            marker='*',
            edgecolors='black',
            linewidth=2.0
        )
    
    plt.title(title, fontsize=20, pad=20)
    plt.xlabel("Principal Component 1", fontsize=14)
    plt.ylabel("Principal Component 2", fontsize=14)
    
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=3, fontsize=18)
    
    cluster_sizes = pd.Series(labels).value_counts().sort_index()
    print("\nCluster sizes (number of samples):")
    for cluster, size in cluster_sizes.items():
        cluster_name = cluster_labels[cluster] if cluster in cluster_labels else f'Cluster {cluster}'
        print(f"{cluster_name}: {size} samples")
    
    print(f"\nRepresentative embedding indices for each cluster ({method}):")
    for cluster, idx in representative_indices.items():
        cluster_name = cluster_labels[cluster] if (cluster_labels is not None and cluster in cluster_labels) else f'Cluster {cluster}'
        print(f"{cluster_name}: index {idx}")
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2)
    if save_path is not None:
        plt.savefig(save_path)
        print(f"Figure saved to {save_path}")
    plt.show()

def hierarchical_clustering_with_representatives(X, n_clusters_range=None, distance_threshold=None, 
                                               method='ward', representative_method='medoid',
                                               random_state=42, cluster_labels=None, visualize=True, save_path=None):
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from scipy.cluster.hierarchy import linkage, dendrogram
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.metrics import silhouette_score
    from sklearn.decomposition import PCA
    
    if n_clusters_range is None:
        n_clusters_range = range(2, min(11, len(X) // 10 + 1))
    
    print("Generating hierarchical clustering dendrogram...")
    Z = linkage(X, method=method)
    
    if visualize:
        plt.figure(figsize=(12, 8))
        dendrogram(Z, truncate_mode='lastp', p=30, leaf_font_size=10)
        plt.title(f"Hierarchical Clustering Dendrogram ({method} linkage)", fontsize=14)
        plt.xlabel("Sample index or (cluster size)")
        plt.ylabel("Distance")
        plt.tight_layout()
        if save_path is not None:
            dendro_path = save_path.replace('.png', '_dendrogram.png')
            plt.savefig(dendro_path)
            print(f"Dendrogram figure saved to {dendro_path}")
        plt.show()
    
    if distance_threshold is None:
        silhouette_scores = []
        models = []
        labels_list = []
        
        print("Searching for the optimal number of clusters...")
        for n_clusters in n_clusters_range:
            model = AgglomerativeClustering(n_clusters=n_clusters, linkage=method)
            cluster_labels_temp = model.fit_predict(X)
            
            score = silhouette_score(X, cluster_labels_temp)
            silhouette_scores.append(score)
            models.append(model)
            labels_list.append(cluster_labels_temp)
            
            print(f"Silhouette score for {n_clusters} clusters: {score:.3f}")
        
        if visualize:
            plt.figure(figsize=(10, 6))
            plt.plot(list(n_clusters_range), silhouette_scores, marker='o')
            plt.grid(True)
            plt.xlabel('Number of clusters')
            plt.ylabel('Silhouette Score')
            plt.title('Silhouette Score by Cluster Count (Higher is better)')
            
            max_idx = np.argmax(silhouette_scores)
            max_n_clusters = list(n_clusters_range)[max_idx]
            max_score = silhouette_scores[max_idx]
            
            plt.axvline(x=max_n_clusters, linestyle='--', color='r', alpha=0.7)
            plt.axhline(y=max_score, linestyle='--', color='r', alpha=0.7)
            plt.annotate(f'Optimal k = {max_n_clusters}\nScore = {max_score:.3f}',
                        xy=(max_n_clusters, max_score),
                        xytext=(max_n_clusters + 0.5, max_score - 0.02),
                        arrowprops=dict(facecolor='black', shrink=0.05, width=1.5))
            plt.tight_layout()
            if save_path is not None:
                sil_path = save_path.replace('.png', '_silhouette.png')
                plt.savefig(sil_path)
                print(f"Silhouette figure saved to {sil_path}")
            plt.show()
        else:
            max_idx = np.argmax(silhouette_scores)
            max_n_clusters = list(n_clusters_range)[max_idx]
            max_score = silhouette_scores[max_idx]
        
        optimal_model = models[max_idx]
        labels = labels_list[max_idx]
        print(f"\nOptimal number of clusters: {max_n_clusters} (Silhouette score: {max_score:.3f})")
    else:
        optimal_model = AgglomerativeClustering(
            distance_threshold=distance_threshold,
            n_clusters=None,
            linkage=method
        )
        labels = optimal_model.fit_predict(X)
        n_clusters = len(set(labels))
        print(f"\n{n_clusters} clusters created with distance threshold {distance_threshold}")
    
    print("Preparing for dimensionality reduction and visualization with PCA...")
    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X)
    
    print(f"Finding representative embedding for each cluster using {representative_method} method...")
    representative_indices = find_representative_embeddings(
        X_2d, labels, method=representative_method, random_state=random_state
    )
    
    if visualize:
        visualize_clusters_with_representatives(X_2d, labels, representative_indices, 
                                               representative_method, pca, cluster_labels, save_path=save_path)
    
    return optimal_model, labels, pca, representative_indices


def main(input_path, compressed_criteria_path, visualize=True, figure_output_path=None):
    with open(input_path, "r") as f:
        criteria_embeddings = []
        criteria = []
        for line in f:
            data = json.loads(line)
            if data['criteria'] != "":
                criteria_embeddings.append(data["embedding"])
                criteria.append(data['criteria'])

    optimal_model, labels, pca, representative_indices = hierarchical_clustering_with_representatives(
        np.array(criteria_embeddings), 
        distance_threshold=10.5,
        representative_method='medoid',
        visualize=visualize,
        save_path=figure_output_path
    )

    compressed_criteria = []
    for cluster, idx in representative_indices.items():
        print(f"Cluster {cluster}: {criteria[idx]}")
        compressed_criteria.append(criteria[idx])
    
    with open(compressed_criteria_path, "w") as f:
        for criterion in compressed_criteria:
            f.write(criterion + "\n")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Hierarchical clustering with representative selection and optional visualization.")
    parser.add_argument('--input_path', type=str, required=True, help='Path to input JSONL file with embeddings and criteria.')
    parser.add_argument('--compressed_criteria_path', type=str, required=True, help='Path to input txt file with compressed criteria.')
    parser.add_argument('--visualize', action='store_true', help='Enable visualization and figure saving.')
    parser.add_argument('--figure_output_path', type=str, default=None, help='Path to save the visualization figure (if visualize is enabled).')
    args = parser.parse_args()
    main(args.input_path, args.compressed_criteria_path, visualize=args.visualize, figure_output_path=args.figure_output_path)
