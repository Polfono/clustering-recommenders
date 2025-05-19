import os
import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.cluster import DBSCAN
import skfuzzy as fuzz
from sklearn.preprocessing import StandardScaler
from timeit import default_timer as timer
from tqdm import tqdm
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from sklearn.mixture import GaussianMixture
from numba import njit, prange, get_thread_id, config
import time

time_cluster = 0
time_similarities = 0
time_deviations = 0
time_prediction = 0

def cluster_entropy(labels, n_clusters):
    total = len(labels)
    counts = np.array([(labels == i).sum() for i in range(n_clusters)])
    probs = counts / total
    entropy = -np.sum([p * np.log2(p) for p in probs if p > 0])
    normalized_entropy = entropy / np.log2(n_clusters) if n_clusters > 1 else 0
    return normalized_entropy

@njit(fastmath=True)
def pearson_dist(x, y):
    # Convert to float64 arrays
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    n = x.size
    # mask out NaNs and zeros in‐place
    # (Numba supports simple loops faster than fancy indexing)
    tmpx = np.empty(n, np.float64)
    tmpy = np.empty(n, np.float64)
    m = 0
    for i in range(n):
        xi = x[i]
        yi = y[i]
        if not np.isnan(xi) and not np.isnan(yi) and xi != 0.0 and yi != 0.0:
            tmpx[m] = xi
            tmpy[m] = yi
            m += 1
    if m < 2:
        return 1000.0

    # compute means
    xm = 0.0
    ym = 0.0
    for i in range(m):
        xm += tmpx[i]
        ym += tmpy[i]
    xm /= m
    ym /= m

    # compute centered sums
    ssx = 0.0
    ssy = 0.0
    cov = 0.0
    for i in range(m):
        dx = tmpx[i] - xm
        dy = tmpy[i] - ym
        ssx += dx * dx
        ssy += dy * dy
        cov += dx * dy

    if ssx == 0.0 or ssy == 0.0:
        return 1

    r = cov / np.sqrt(ssx * ssy)
    denom = 1.0 + r
    return (1.0 / denom) if denom > 0.0 else 1000.0

@njit(parallel=True, fastmath=True)
def kmeans_numba_parallel_fixed(data, n_clusters, max_iters=100, tol=1e-4):
    """
    Parallel K-Means clustering using Euclidean distance.
    """
    n_samples, n_features = data.shape

    if n_clusters <= 0:
        raise ValueError("n_clusters must be positive.")
    if n_clusters > n_samples:
        raise ValueError("n_clusters cannot be greater than n_samples.")

    initial_indices = np.random.choice(n_samples, n_clusters, replace=False)
    centroids = np.empty((n_clusters, n_features), dtype=data.dtype)
    for k in range(n_clusters):
        centroids[k, :] = data[initial_indices[k], :]

    labels = np.full(n_samples, -1, dtype=np.int64)

    num_threads_for_accumulation = config.NUMBA_NUM_THREADS
    if num_threads_for_accumulation <= 0: # Should not happen with default config
        num_threads_for_accumulation = 1


    private_centroids_sum = np.zeros((num_threads_for_accumulation, n_clusters, n_features), dtype=data.dtype)
    private_cluster_counts = np.zeros((num_threads_for_accumulation, n_clusters), dtype=np.int64)

    for iteration in range(max_iters):
        changed_assignments = False

        for i in prange(n_samples):
            best_k = 0
            min_dist_sq = pearson_dist(data[i, :], centroids[0, :])
            for k in range(1, n_clusters):
                dist_sq = pearson_dist(data[i, :], centroids[k, :])
                if dist_sq < min_dist_sq:
                    min_dist_sq = dist_sq
                    best_k = k
            
            if labels[i] != best_k:
                changed_assignments = True
            labels[i] = best_k

        if not changed_assignments:
            
            break
        
        private_centroids_sum[:, :, :] = 0.0
        private_cluster_counts[:, :] = 0

        for i in prange(n_samples):
            thread_id = get_thread_id()
            k_assigned = labels[i]
            
            for j in range(n_features):
                private_centroids_sum[thread_id, k_assigned, j] += data[i, j]
            private_cluster_counts[thread_id, k_assigned] += 1

        final_centroids_sum = np.zeros((n_clusters, n_features), dtype=data.dtype)
        final_cluster_counts = np.zeros(n_clusters, dtype=np.int64)

        
        for tid in range(num_threads_for_accumulation):
            for k_idx in range(n_clusters):
                for feat_idx in range(n_features):
                    final_centroids_sum[k_idx, feat_idx] += private_centroids_sum[tid, k_idx, feat_idx]
                final_cluster_counts[k_idx] += private_cluster_counts[tid, k_idx]
        
        max_centroid_shift_sq = 0.0 # For tolerance-based convergence

        for k in range(n_clusters):
            if final_cluster_counts[k] > 0:
                new_centroid_k = final_centroids_sum[k, :] / final_cluster_counts[k]
                shift_sq = pearson_dist(new_centroid_k, centroids[k, :])
                if shift_sq > max_centroid_shift_sq:
                    max_centroid_shift_sq = shift_sq
                centroids[k, :] = new_centroid_k
            else:
                random_idx = np.random.randint(0, n_samples)
                centroids[k, :] = data[random_idx, :]
                
                max_centroid_shift_sq = np.inf # Or a very large number

        # Tolerance-based convergence check
        if tol > 0 and max_centroid_shift_sq < tol * tol : # Compare with squared tolerance
            break

    return labels

# --------------------
# 1) CLUSTERING METHODS
# --------------------

def hard_cluster(user_item):
    sample = user_item.to_numpy()
    sample[np.isnan(sample)] = 0
    
    labels = kmeans_numba_parallel_fixed(sample, n_clusters, max_iters=100)
    
    cluster_members = {
        cid: np.where(labels == cid)[0]
        for cid in range(n_clusters)
    }

    
    return labels, cluster_members


def fuzzy_cmeans_cluster(user_item):
    filled = user_item.values.copy()
    user_means = np.nanmean(filled, axis=1, keepdims=True)
    filled[np.isnan(filled)] = np.take(user_means, np.where(np.isnan(filled))[0])

    # 2) Estandarizar usuarios
    scaler = StandardScaler()
    scaled = scaler.fit_transform(filled)
    data = scaled.T  # (features × samples) for skfuzzy

    # 3) Fuzzy C-means
    _, u, _, _, _, _, _ = fuzz.cluster.cmeans(
        data, c=n_clusters, m=m, error=error, maxiter=maxiter, init=None, seed=seed
    )

    # 4) Defuzzify — COG
    labels = np.arange(n_clusters)[:, None]         # 0,1,…,n_clusters-1
    cog = (labels * u).sum(axis=0)                  # center of gravity
    cluster_assign = np.rint(cog).astype(int)       # now in {0,…,n_clusters-1}

    # 5) Agrupar índices de usuarios por clúster asignado
    cluster_members = {
        cid: np.where(cluster_assign == cid)[0]
        for cid in range(n_clusters)
    }

    return cluster_assign, cluster_members

def hac_cluster(similarity_df):

    # 1) Convertir similitud a distancia: d = 1/s si s>0, o un valor grande si s<=0
    sim = 1 + similarity_df
    with np.errstate(divide='ignore'):
        dist = np.where(sim > 0, 1.0/sim, 1000.0)
    # Aseguramos diagonal cero
    np.fill_diagonal(dist, 0.0)

    # 2) Obtener el vector condensado para pdist
    dist_vec = squareform(dist, checks=False)
    
    # 3) HAC linkage
    Z_average  = linkage(dist_vec, method='average')
    """
    Z_weighted = linkage(dist_vec, method='weighted')
    Z_centroid = linkage(dist_vec, method='centroid')
    Z_single   = linkage(dist_vec, method='single')
    Z_complete = linkage(dist_vec, method='complete')
    """

    
    # 4) Cortar el dendrograma en k clusters
    labels_average  = fcluster(Z_average, t=n_clusters, criterion='maxclust') - 1
    """
    labels_weighted = fcluster(Z_weighted, t=n_clusters, criterion='maxclust') - 1
    labels_centroid = fcluster(Z_centroid, t=n_clusters, criterion='maxclust') - 1
    labels_single   = fcluster(Z_single, t=n_clusters, criterion='maxclust')  - 1
    labels_complete = fcluster(Z_complete, t=n_clusters, criterion='maxclust') - 1
    """
    
    # 6) Construir dict de miembros por cluster
    cluster_members = {
        cid: np.where(labels_average == cid)[0]
        for cid in range(n_clusters)
    }
    
    return labels_average, cluster_members

def density_cluster(sim):
    sim = 1 + sim
    with np.errstate(divide='ignore'):
        dist = np.where(sim > 0, 1.0/sim, 1000.0)
    # Aseguramos diagonal cero
    np.fill_diagonal(dist, 0.0)

    # 4) Aplicar DBSCAN
    # eps y min_samples pueden ajustarse según el dataset
    dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='precomputed', n_jobs=-1)
    labels = dbscan.fit_predict(dist)

    # Calcular y mostrar el porcentaje de usuarios sin clúster asignado
    num_unassigned = np.sum(labels == -1)
    total_users = len(labels)
    percent_unassigned = 100 * num_unassigned / total_users
    print(f"Porcentaje de usuarios sin clúster asignado: {percent_unassigned:.2f}%")


    # DBSCAN asigna -1 a ruido, convertimos en cluster separado o lo ignoramos
    unique_labels = set(labels)
    # Map labels to consecutive indices (optional)
    label_map = {lab: idx for idx, lab in enumerate(sorted(unique_labels))}
    clusters = np.array([label_map[lab] for lab in labels])

    # 5) Agrupar índices de usuarios por clúster asignado
    cluster_members = {
        cid: np.where(clusters == cid)[0]
        for cid in np.unique(clusters)
    }

    return clusters, cluster_members


def gmm_cluster(user_item):
    # 1) Imputar NaNs con la media de cada usuario
    filled = user_item.values.copy()
    user_means = np.nanmean(filled, axis=1, keepdims=True)
    filled[np.isnan(filled)] = np.take(user_means, np.where(np.isnan(filled))[0])

    # 2) Estandarizar características (items)
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(filled)

    # 3) Ajustar Gaussian Mixture Model
    # n_components = n_clusters global, covariance_type puede ajustarse
    gmm = GaussianMixture(n_components=n_clusters, covariance_type='diag', random_state=seed)
    gmm.fit(data_scaled)

    # 4) Predecir etiquetas de cluster
    labels = gmm.predict(data_scaled)

    # 5) Construir dict de miembros por cluster
    cluster_members = {
        cid: np.where(labels == cid)[0]
        for cid in range(n_clusters)
    }

    return labels, cluster_members


# Diccionario de métodos disponibles
CLUSTER_METHODS = {
    'hard': hard_cluster,
    'fuzzy': fuzzy_cmeans_cluster,
    'hac': hac_cluster,
    'density': density_cluster,
    'gmm': gmm_cluster
}

# --------------------
# 2) SIMILARIDAD
# --------------------

def similarity(train_df):
    user_item = train_df.pivot_table(index='userId', columns='itemId', values='rating')

    R = user_item.values
    mask = ~np.isnan(R)

    # Media de valoraciones por usuario
    sums = np.nansum(R, axis=1)
    counts = mask.sum(axis=1)
    means = sums / counts

    # Restar la media
    R_centered = (R - means[:, None]) * mask
    R_centered[np.isnan(R_centered)] = 0

    # Producto punto entre usuarios
    norms = np.linalg.norm(R_centered, axis=1)
    numerator = R_centered.dot(R_centered.T)

    # Producto de las normas de los vectores
    denominator = norms[:, None] * norms[None, :]

    # Pearson
    sim = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator != 0)

    return sim, R_centered, means, user_item

# --------------------
# 3) PREDICCIÓN
# --------------------

@njit
def predict_with_numba(top_sims: np.ndarray,
                       deviations: np.ndarray,
                       means: np.ndarray,
                       user_idx: int) -> float:
    denom = 0.0
    # sum of absolute similarities
    for i in range(top_sims.shape[0]):
        denom += abs(top_sims[i])

    if denom < 1e-9:
        # fallback to user's mean if no significant similarity
        return means[user_idx]
    else:
        # weighted average of deviations
        numer = 0.0
        for i in range(top_sims.shape[0]):
            numer += top_sims[i] * deviations[i]
        delta = numer / denom
        return means[user_idx] + delta
    

    

def predict_rating(user_id, item_id,
                   sim_df, R_centered, means,
                   user_item, clusters,
                   members_arr_dict):
    
    global time_cluster, time_candidates, time_similarities, time_deviations, time_topk, time_prediction
    # Map user_id to index in sim_df / R
    user_idx = user_item.index.get_loc(user_id)

    # 1) Cluster lookup
    start = time.time()
    cl = clusters[user_idx]
    members_arr = members_arr_dict[cl]
    # remove self from cluster
    members_arr = members_arr[members_arr != user_idx]
    time_cluster += time.time() - start

    # If no other members, return user mean
    if members_arr.size == 0:
        return float(means[user_idx])

    # 2) Similarities within cluster
    start = time.time()
    sims_all = sim_df[user_idx, members_arr]
    time_similarities += time.time() - start

    # 3) Select top-K most similar users
    start = time.time()
    # Get indices of top-K similar users in the cluster
    k = min(top_k, sims_all.size)
    top_idxs = np.argsort(-sims_all)[:k]
    top_users = members_arr[top_idxs]
    top_sims = sims_all[top_idxs]
    time_topk = time.time() - start

    # 4) Deviations for selected users on the target item
    start = time.time()
    item_idx = user_item.columns.get_loc(item_id)
    deviations = R_centered[top_users, item_idx]
    time_deviations += time.time() - start

    # 5) Prediction calculation
    start = time.time()
    pred = predict_with_numba(top_sims, deviations, means, user_idx)
    time_prediction += time.time() - start

    # Clip to allowed range and return
    return float(np.clip(pred, MIN_RATING, MAX_RATING))

# --------------------
# 4) EVALUACIÓN POR FOLDS
# --------------------

def evaluate_fold(args):
    train_index, test_index, ratings, cluster_method = args
    start_total = timer()

    # Split
    train_df = ratings.iloc[train_index]
    test_df = ratings.iloc[test_index]

    # 1) Similarity computation
    start_sim = timer()
    sim_df, R_centered, means, user_item = similarity(train_df)
    end_sim = timer()

    # 2) Clustering
    start_cluster = timer()
    cluster_fn = CLUSTER_METHODS[cluster_method]
    if cluster_method in ('hac', 'density'):
        clusters, cluster_members = cluster_fn(sim_df)
    else:
        clusters, cluster_members = cluster_fn(user_item)
    n_clusters = len(cluster_members)
    entropy = cluster_entropy(clusters, n_clusters)
    end_cluster = timer()

    # 3) Precompute helper structures
    # Map each cluster to a numpy array of member indices
    members_arr_dict = {
        cl: np.array(members_list, dtype=int)
        for cl, members_list in cluster_members.items()
    }

    # 4) Prediction on test set
    y_true, y_pred = [], []
    train_users = set(train_df['userId'])
    train_items = set(train_df['itemId'])

    def _predict_one(row):
        u, i, r = row['userId'], row['itemId'], row['rating']
        if u not in train_users or i not in train_items:
            return None
        p = predict_rating(u, i,
                           sim_df, R_centered, means,
                           user_item, clusters,
                           members_arr_dict)
        return (r, p)

    
    start_pred = timer()
    
    rows = list(test_df.to_dict('records'))
    
    for row in rows:
        result = _predict_one(row)
        if result:
            y_true.append(result[0])
            y_pred.append(result[1])

    end_pred = timer()
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    nmae = mae / (ratings['rating'].max() - ratings['rating'].min())
    nrmse = rmse / (ratings['rating'].max() - ratings['rating'].min())

    end_total = timer()

    return mae, nmae, rmse, nrmse, end_sim - start_sim, end_cluster - start_cluster, end_pred - start_pred, end_total - start_total, entropy, n_clusters

# --------------------
# 5) CROSS-VALIDATION
# --------------------

def cross_validate(ratings, cluster_method):
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    args_list = [(train_idx, test_idx, ratings, cluster_method) for i, (train_idx, test_idx) in enumerate(kf.split(ratings))]

    results = []
    for args in tqdm(args_list, desc="Evaluando folds", unit="fold"):
        results.append(evaluate_fold(args))

    mae_scores, nmae_scores, rmse_scores, nrmse_scores, sim_times, cluster_times, pred_times, total_times, entropyes, n_clusters = zip(*results)

    print(f'\n--- Promedios por fold ---')
    print(f'Average MAE: {np.mean(mae_scores):.4f}')
    print(f'Average NMAE: {np.mean(nmae_scores):.4f}')
    print(f'Average RMSE: {np.mean(rmse_scores):.4f}')
    print(f'Average NRMSE: {np.mean(nrmse_scores):.4f}')
    print(f'Average Entropy: {np.mean(entropyes):.4f}')
    print(f'Average Clusters: {np.mean(n_clusters):.4f}')
    print(f'Average Time - Similarity: {np.mean(sim_times):.2f}s')
    print(f'Average Time - Clustering: {np.mean(cluster_times):.2f}s')
    print(f'Average Time - Prediction: {np.mean(pred_times):.2f}s')
    print(f'Average Time - Fold: {np.mean(total_times):.2f}s')

# --------------------
# 6) MAIN
# --------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Ejecuta validación cruzada con clustering hard o fuzzy sobre un dataset de ratings.'
    )
    parser.add_argument('-m', '--method', choices=['hard', 'fuzzy', 'hac', 'density', 'gmm'], default='hard',
                        help="Tipo de clustering ('hard' o 'fuzzy').")
    parser.add_argument('-n', '--n_clusters', type=int, default=1,
                        help='Número de clusters a usar.')
    parser.add_argument('-d', '--dataset', default='ml-small',
                        help="Nombre del fichero CSV (sin extensión) en ../Datasets/. Default: 'ml-small'.")
    args = parser.parse_args()

    # Parámetros globales
    n_clusters = args.n_clusters
    seed = 42
    top_k = 10
    m = 2.0
    error = 0.005
    maxiter = 1000

    # Carga del dataset
    dataset_path = '../Datasets/'
    csv_file = os.path.join(dataset_path, f'{args.dataset}.csv')
    if not os.path.isfile(csv_file):
        print(f"Error: fichero no encontrado: {csv_file}")
        exit(1)
    ratings = pd.read_csv(csv_file)

    MIN_RATING = ratings['rating'].min()
    MAX_RATING = ratings['rating'].max()

    eps = 0.85
    min_samples = 5

    cross_validate(ratings, cluster_method=args.method)

    print(f"\n--- Tiempos ---")
    print(f"Tiempo total - Clustering: {time_cluster:.2f}s")
    print(f"Tiempo total - Similitudes: {time_similarities:.2f}s")
    print(f"Tiempo total - Desviaciones: {time_deviations:.2f}s")
    print(f"Tiempo total - Predicción: {time_prediction:.2f}s")