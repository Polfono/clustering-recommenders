import os
import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.cluster import KMeans, DBSCAN
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

def hard_cluster_pearson(user_item):
    sample = user_item.to_numpy()
    sample[np.isnan(sample)] = 0
    
    labels = kmeans_numba_parallel_fixed(sample, n_clusters, max_iters=100)
    
    cluster_members = {
        cid: np.where(labels == cid)[0]
        for cid in range(n_clusters)
    }

    
    return labels, cluster_members

def hard_cluster(user_item):
    # 1) Imputación de NaNs con la media de cada usuario
    filled = user_item.values.copy()
    user_means = np.nanmean(filled, axis=1, keepdims=True)
    filled[np.isnan(filled)] = np.take(user_means, np.where(np.isnan(filled))[0])

    # 3) Estandarizar características (items)
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(filled)

    # 5) Aplicar clustering duro (KMeans)
    kmeans = KMeans(n_clusters=n_clusters, random_state=seed)
    clusters = kmeans.fit_predict(data_scaled)

    # 6) Agrupar índices de usuarios por clúster asignado
    cluster_members = {
        cid: np.where(clusters == cid)[0]
        for cid in range(n_clusters)
    }

    return clusters, cluster_members


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
    sim = 1 + similarity_df.values.copy()
    with np.errstate(divide='ignore'):
        dist = np.where(sim > 0, 1.0/sim, 1000.0)
    # Aseguramos diagonal cero
    np.fill_diagonal(dist, 0.0)
    '''
    dist = 1 - similarity_df.values.copy()
    dist[dist < 0] = 0.0
    np.fill_diagonal(dist, 0.0)
    '''

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
    sim = 1 + sim.values.copy()
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
    'hard_pearson': hard_cluster_pearson,
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

    return pd.DataFrame(sim, index=user_item.index, columns=user_item.index), R_centered, means, user_item

# --------------------
# 3) PREDICCIÓN
# --------------------

# --- Global Caches for Precomputation ---
_UIDX_CACHE = {}
_ITEM_COL_LOC_CACHE = {}
_MEMBERS_IN_CLUSTER_CACHE = {} 
_HAS_RATED_MASK_CACHE = {} 
_SIM_DF_NP_CACHE = {} 
_USER_ITEM_ISNA_VALUES_CACHE = {}


def predict_rating(user_id, item_id, sim_df, R_centered, means, user_item, clusters, cluster_members):
    global time_cluster, time_deviations, time_prediction, time_similarities, top_k, MIN_RATING, MAX_RATING
    global _UIDX_CACHE, _ITEM_COL_LOC_CACHE, _MEMBERS_IN_CLUSTER_CACHE, _HAS_RATED_MASK_CACHE, _SIM_DF_NP_CACHE

    user_idx = _UIDX_CACHE[user_id]

    start_time_cluster = time.time()
    members_array = _MEMBERS_IN_CLUSTER_CACHE[user_idx]
    time_cluster += time.time() - start_time_cluster
    
    start_time_similarities = time.time()
    item_col_loc = _ITEM_COL_LOC_CACHE[item_id]
    has_rated_item_mask = _HAS_RATED_MASK_CACHE[item_id]
        
    if members_array.size > 0:
        valid_candidates_mask = has_rated_item_mask[members_array]
        candidate_neighbors = members_array[valid_candidates_mask]
    else:
        candidate_neighbors = np.array([], dtype=int)
        
    time_similarities += time.time() - start_time_similarities

    if not candidate_neighbors.size:
        return means[user_idx]

    start_time_deviations = time.time()
    sim_df_np_ = _SIM_DF_NP_CACHE['cached']
    neighbor_sims = sim_df_np_[user_idx, candidate_neighbors]
    neighbor_deviations = R_centered[candidate_neighbors, item_col_loc]

    num_candidates = neighbor_sims.shape[0]

    if num_candidates == 0: 
        time_deviations += time.time() - start_time_deviations
        return means[user_idx]

    if top_k < num_candidates:
        top_k_relative_indices_unsorted = np.argpartition(-neighbor_sims, top_k - 1)[:top_k]
        sims_of_top_k_partitioned = neighbor_sims[top_k_relative_indices_unsorted]
        order_within_top_k = np.argsort(-sims_of_top_k_partitioned)
        final_top_k_relative_indices = top_k_relative_indices_unsorted[order_within_top_k]
    else:
        final_top_k_relative_indices = np.argsort(-neighbor_sims)

    top_sims = neighbor_sims[final_top_k_relative_indices]
    top_devs = neighbor_deviations[final_top_k_relative_indices]

    sum_abs_top_sims = np.sum(np.abs(top_sims))
    time_deviations += time.time() - start_time_deviations
    
    if sum_abs_top_sims < 1e-9:
        return means[user_idx]

    start_time_prediction = time.time()
    weighted_sum_devs = np.dot(top_sims, top_devs)
    prediction_delta = weighted_sum_devs / sum_abs_top_sims
    predicted_rating = means[user_idx] + prediction_delta
    time_prediction += time.time() - start_time_prediction

    return float(np.clip(predicted_rating, MIN_RATING, MAX_RATING))


# --------------------
# 4) EVALUACIÓN POR FOLDS
# --------------------

def evaluate_fold(args):
    train_index, test_index, ratings, cluster_method = args
    start_total = timer()

    start_sim = timer()
    train_df = ratings.iloc[train_index]
    test_df = ratings.iloc[test_index]
    sim_df, R_centered, means, user_item = similarity(train_df)
    end_sim = timer()

    start_cluster = timer()
    cluster_fn = CLUSTER_METHODS[cluster_method]

    if(cluster_method == 'hac' or cluster_method == 'density'):
        clusters, cluster_members = cluster_fn(sim_df)
    else:
        clusters, cluster_members = cluster_fn(user_item)

    n_clusters = len(cluster_members)
    entropy = cluster_entropy(clusters, n_clusters)
    end_cluster = timer()

    global _UIDX_CACHE, _ITEM_COL_LOC_CACHE, _MEMBERS_IN_CLUSTER_CACHE, _HAS_RATED_MASK_CACHE, _SIM_DF_NP_CACHE, _USER_ITEM_ISNA_VALUES_CACHE
    
    _UIDX_CACHE.clear()
    _ITEM_COL_LOC_CACHE.clear()
    _MEMBERS_IN_CLUSTER_CACHE.clear()
    _HAS_RATED_MASK_CACHE.clear()
    _SIM_DF_NP_CACHE.clear()
    _USER_ITEM_ISNA_VALUES_CACHE.clear()

    for uid_ in user_item.index:
        _UIDX_CACHE[uid_] = user_item.index.get_loc(uid_)
    
    for iid_ in user_item.columns:
        _ITEM_COL_LOC_CACHE[iid_] = user_item.columns.get_loc(iid_)

    for u_idx_ in range(len(user_item.index)):
        current_user_cluster_ = clusters[u_idx_]
        cluster_member_indices_ = cluster_members[current_user_cluster_]
        
        if isinstance(cluster_member_indices_, np.ndarray):
            member_indices_list_ = cluster_member_indices_.tolist()
        elif isinstance(cluster_member_indices_, pd.Series):
            member_indices_list_ = cluster_member_indices_.tolist()
        else:
            member_indices_list_ = list(cluster_member_indices_)
        
        members_for_uidx_ = [m_idx for m_idx in member_indices_list_ if m_idx != u_idx_]
        _MEMBERS_IN_CLUSTER_CACHE[u_idx_] = np.array(members_for_uidx_, dtype=int)

    _USER_ITEM_ISNA_VALUES_CACHE['cached'] = user_item.isna().values
    user_item_is_na_matrix_ = _USER_ITEM_ISNA_VALUES_CACHE['cached']

    for item_id_val_ in user_item.columns: 
        if item_id_val_ in _ITEM_COL_LOC_CACHE: 
            item_col_idx_ = _ITEM_COL_LOC_CACHE[item_id_val_]
            _HAS_RATED_MASK_CACHE[item_id_val_] = ~user_item_is_na_matrix_[:, item_col_idx_]

    _SIM_DF_NP_CACHE['cached'] = sim_df.values

    y_true = []
    y_pred = []

    train_users = set(train_df['userId'])
    train_items = set(train_df['itemId'])

    from joblib import Parallel, delayed

    def _predict_one(row):
        u, i, r = row['userId'], row['itemId'], row['rating']
        if u not in train_users or i not in train_items:
            return None
        p = predict_rating(u, i, sim_df, R_centered, means,
                        user_item, clusters, cluster_members)
        return (r, p)
    
    start_pred = timer()
    rows = list(test_df.to_dict('records'))
    
    for row in rows:
        result = _predict_one(row)
        if result is not None:
            y_true.append(result[0])
            y_pred.append(result[1])

    end_pred = timer()
    
    if not y_true:
        mae, rmse, nmae = np.nan, np.nan, np.nan
    else:
        y_true_np, y_pred_np = np.array(y_true), np.array(y_pred)
        mae = mean_absolute_error(y_true_np, y_pred_np)
        rmse = np.sqrt(mean_squared_error(y_true_np, y_pred_np))
        rating_range = ratings['rating'].max() - ratings['rating'].min()
        if rating_range == 0:
            nmae = np.nan if mae > 0 else 0.0 # Or handle as per specific requirements for NMAE with zero range
        else:
            nmae = mae / rating_range
    
    end_total = timer()

    return mae, nmae, rmse, end_sim - start_sim, end_cluster - start_cluster, end_pred - start_pred, end_total - start_total, entropy, n_clusters

# --------------------
# 5) CROSS-VALIDATION
# --------------------

def cross_validate(ratings, cluster_method):
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    args_list = [(train_idx, test_idx, ratings, cluster_method) for i, (train_idx, test_idx) in enumerate(kf.split(ratings))]

    results = []
    for args in tqdm(args_list, desc="Evaluando folds", unit="fold"):
        results.append(evaluate_fold(args))

    mae_scores, nmae_scores, rmse_scores, sim_times, cluster_times, pred_times, total_times, entropyes, n_clusters = zip(*results)

    print(f'\n--- Promedios por fold ---')
    print(f'Average MAE: {np.mean(mae_scores):.4f}')
    print(f'Average NMAE: {np.mean(nmae_scores):.4f}')
    print(f'Average RMSE: {np.mean(rmse_scores):.4f}')
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
    parser.add_argument('-m', '--method', choices=['hard', 'hard_pearson', 'fuzzy', 'hac', 'density', 'gmm'], default='hard',
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

    print(f"\nTiempo total de clustering: {time_cluster:.2f}s")
    print(f"Tiempo total de similitudes: {time_similarities:.2f}s")
    print(f"Tiempo total de desviaciones: {time_deviations:.2f}s")
    print(f"Tiempo total de predicción: {time_prediction:.2f}s")