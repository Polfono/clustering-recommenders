import os
import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.cluster import KMeans
import skfuzzy as fuzz
from sklearn.preprocessing import StandardScaler
from timeit import default_timer as timer
from tqdm import tqdm
from sklearn.preprocessing import normalize


# --------------------
# 1) CLUSTERING METHODS
# --------------------

def hard_cluster(user_item):
    # 1) Imputación de NaNs con la media de cada usuario
    filled = user_item.values.copy()
    user_means = np.nanmean(filled, axis=1, keepdims=True)
    filled[np.isnan(filled)] = np.take(user_means, np.where(np.isnan(filled))[0])

    # 3) Estandarizar características (items)
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(filled)

    # 4) Normalizar cada vector usuario a norma L2 (opcional)
    
    data_norm = normalize(data_scaled, norm='l2', axis=1)

    # 5) Aplicar clustering duro (KMeans)
    kmeans = KMeans(n_clusters=n_clusters, random_state=seed)
    clusters = kmeans.fit_predict(data_norm)

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


# Diccionario de métodos disponibles
CLUSTER_METHODS = {
    'hard': hard_cluster,
    'fuzzy': fuzzy_cmeans_cluster
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

def predict_rating(user_id, item_id, sim_df, R_centered, means, user_item, clusters, cluster_members):
    users = list(user_item.index)
    user_idx = users.index(user_id)

    # Cluster del usuario
    cl = clusters[user_idx]

    # miembros del cluster
    members = cluster_members[cl].tolist()
    members = [m for m in members if m != user_idx]
    
    # Vecinos que hayan valorado el item
    ratings_col = user_item[item_id]
    rated_mask = ~ratings_col.isna().values
    candidates = [m for m in members if rated_mask[m]]

    # Si no hay devoler la media
    if not candidates:
        return means[user_idx]

    # Similitudes
    sims = sim_df.iloc[user_idx, candidates].values

    # Desviaciones
    deviations = R_centered[candidates, user_item.columns.get_loc(item_id)]

    # Seleccionar los k vecinos más similares
    top_k_idxs = np.argsort(-sims)[:top_k]
    top_sims = sims[top_k_idxs]
    top_devs = deviations[top_k_idxs]

    # Calcular la suma ponderada de desviaciones
    denom = np.sum(np.abs(top_sims))
    if denom < 1e-9:
        return means[user_idx]

    # Añadir desviación ponderada a la media del usuario
    delta = np.dot(top_sims, top_devs) / denom
    pred = means[user_idx] + delta

    # Limitar predicción al rango permitido
    return float(np.clip(pred, MIN_RATING, MAX_RATING))




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
    clusters, cluster_members = cluster_fn(user_item)
    end_cluster = timer()

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
    results = Parallel(n_jobs=-1)(
    delayed(_predict_one)(row) for row in rows
    )
    end_pred = timer()

    # unpack
    results = [r for r in results if r is not None]
    if results:
        y_true, y_pred = zip(*results)
        y_true, y_pred = np.array(y_true), np.array(y_pred)
    else:
        y_true, y_pred = np.array([]), np.array([])

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    nmae = mae / (ratings['rating'].max() - ratings['rating'].min())

    end_total = timer()

    return mae, nmae, rmse, end_sim - start_sim, end_cluster - start_cluster, end_pred - start_pred, end_total - start_total

# --------------------
# 5) CROSS-VALIDATION
# --------------------

def cross_validate(ratings, cluster_method):
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    args_list = [(train_idx, test_idx, ratings, cluster_method) for i, (train_idx, test_idx) in enumerate(kf.split(ratings))]

    results = []
    for args in tqdm(args_list, desc="Evaluando folds", unit="fold"):
        results.append(evaluate_fold(args))

    mae_scores, nmae_scores, rmse_scores, sim_times, cluster_times, pred_times, total_times = zip(*results)

    print(f'\n--- Promedios por fold ---')
    print(f'Average MAE: {np.mean(mae_scores):.4f}')
    print(f'Average NMAE: {np.mean(nmae_scores):.4f}')
    print(f'Average RMSE: {np.mean(rmse_scores):.4f}')
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
    parser.add_argument('-m', '--method', choices=['hard', 'fuzzy'], default='hard',
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

    cross_validate(ratings, cluster_method=args.method)