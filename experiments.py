# coding: utf-8

import math

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from coclust.clustering import SphericalKmeans
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import normalized_mutual_info_score, adjusted_rand_score, adjusted_mutual_info_score
from tqdm import trange
from scipy.stats import pearsonr

from utils import pca, pca_whiten, umap, get_embeddings, accuracy

# global params
n_neighbors = 15
n_dims = 10
dim_red_funcs = {
    'raw': lambda x, d: x,
    'pca': lambda x, d: pca(x, d),
    'pca_whiten': lambda x, d: pca_whiten(x, d),
    'umap': lambda x, d: umap(x, d, n_neighbors=n_neighbors),
}

def pc_anisotropy(X):
    _, eig_vectors = np.linalg.eig(np.matmul(np.transpose(X), X))
    max_f = -math.inf
    min_f =  math.inf

    all_f = []
    
    for i in range(eig_vectors.shape[1]):
        f = np.matmul(X, np.expand_dims(eig_vectors[:, i], 1))
        f = np.sum(np.exp(f))
        all_f.append(f)

        min_f = min(min_f, f)
        max_f = max(max_f, f)

    isotropy_1 = min_f / max_f
    isotropy_2 = np.std(all_f) / np.mean(all_f)
    
    return isotropy_1, isotropy_2

def cosine_anisotropy(left, right):
    similarities = np.sum(left * right, axis=1) / (np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1))
    return similarities.mean()

def sim_stats(X):
    similarities = cosine_similarity(X)
    np.fill_diagonal(similarities, np.nan)
    similarities = [x for x in similarities.flatten() if x == x]
    similarities = np.array(similarities)
    
    mean, median, std, min_, max_ = similarities.mean(), np.median(similarities), np.std(similarities), similarities.min(), similarities.max()
    range_ = max_ - min_
    
    return {'mean': mean, 'median': median, 'std': std, 'min': min_, 'max': max_, 'range': range_}

def anisotropy_sim_classes(X, real_classes, pred):
    all_similarities = {}
    
    sim_stats_values = sim_stats(X)
    
    all_similarities['cos_aniso'] = sim_stats_values['mean']
    all_similarities['cos_std'] = sim_stats_values['std']
    all_similarities['pc_aniso'] = pc_anisotropy(X)
    
    all_similarities['NMI'] = normalized_mutual_info_score(real_classes, pred)
    all_similarities['ARI'] = adjusted_rand_score(real_classes, pred)
    all_similarities['AMI'] = adjusted_mutual_info_score(real_classes, pred)
    all_similarities['ACC'] = accuracy(real_classes, pred)
    
    return all_similarities

def select_pairs(X, n_pairs):
    left = []
    right = []
    for pair in trange(n_pairs):
        i, j = np.random.RandomState(pair).choice(range(len(X)), 2, replace=False)
        left.append(X[i])
        right.append(X[j])
    
    left = np.vstack(left)
    right = np.vstack(right)

    return left, right

def anisotropy_stats(texts, model_name, n_samples):
    embeddings = get_embeddings(texts, model_name)
    
    results = []
    for layers in ['last', 'avg']:
        X = embeddings[layers]
        
        left, right = select_pairs(X, n_samples)
        
        for dr_name, dr_func in dim_red_funcs.items():
            X_all = np.concatenate([left, right])
            X_red = dr_func(X_all, n_dims)
            
            X_left = X_red[:left.shape[0]]
            X_right = X_red[left.shape[0]:]
            assert len(X_left) == len(left)
            assert len(X_right) == len(right)
            
            cos = cosine_anisotropy(X_left, X_right)
            pc1, pc2 = pc_anisotropy(X_red)
            
            X_red = pca(X_red, 2)

            results.append({
                'dr_name': dr_name, 'layers': layers, 
                'cos': cos, 'pc': pc1, 'pc2': pc2,
                'components': X_red})

    return results

def get_external_measures(transformer_models, texts):
    external_results = {}

    n_samples = 5000

    task = 'word'
    for model_name in transformer_models:
        res = anisotropy_stats(texts, model_name, n_samples)
        for row in res:
            row.update({'model_name': model_name, 'task': task})
            external_results[len(external_results)] = row

    return external_results

def get_internal_measures(transformer_models):
    internal_results = {}

    datasets = ["bbc", "classic3", "classic4", "ag_news", 'dbpedia']

    for model_name in transformer_models:
        for dataset_name in datasets:
            for layers in ['last', 'avg']:
                filename = 'vectors/vectors_{}_{}_{}.npy'.format(model_name, dataset_name, layers)
                X = np.load(filename)
                
                df = pd.read_csv("datasets/{}.csv".format(dataset_name))
                labels = df['label'].values
                k = len(df['label'].unique())
                
                for dr_name, dr_func in dim_red_funcs.items():
                    X_red = dr_func(X, n_dims)
                    pc1, pc2 = pc_anisotropy(X_red)
                    
                    row = {
                        'model_name': model_name,
                        'dataset_name': dataset_name,
                        'layers': layers,
                        'dr_name': dr_name,
                        'pc_anisotropy': pc1,
                        'pc_anisotropy_2': pc2
                    }

                    for algo in ['km', 'skm']:
                        if algo == 'km':
                            pred = KMeans(k, n_init=10, random_state=42).fit(X_red).labels_
                            stats = anisotropy_sim_classes(X_red, labels, pred)
                        elif algo == 'skm':
                            model = SphericalKmeans(k, n_init=10, random_state=42)
                            model.fit(X_red)
                            pred =  model.labels_
                            stats = anisotropy_sim_classes(X_red, labels, pred)
                        else:
                            raise Exception('algo unknown')

                        stats = {str(k_) + '_' + algo: v for k_, v in stats.items()}
                        row.update(stats)

                    i = max(internal_results) + 1 if internal_results else 1
                    internal_results[i] = row
    
    return internal_results

if __name__ == "__main__":
    transformer_models = [
        'bert-large-cased',
        'roberta-large',
    ]

    # compute external measures
    df = pd.read_csv('datasets/wiki_English.csv')
    texts = df['Sentence']
    external_results = get_external_measures(transformer_models, texts)

    # compute internal measures
    internal_results = get_internal_measures(transformer_models)

    # display pairwise correlation and p-value
    df_internal = pd.DataFrame.from_dict(internal_results).T
    df_internal = df_internal[['model_name', 'dataset_name', 'layers', 'dr_name', 'cos_aniso_km', 'pc_anisotropy', 'pc_anisotropy_2',  
            'NMI_km', 'NMI_skm', 'ARI_km', 'ARI_skm']]
    df_external = pd.DataFrame.from_dict(external_results).T
    df_external = df_external.rename({col: '{}_word'.format(col) for col in ['cos', 'pc', 'pc2']}, axis=1)
    df_external = df_external[['model_name', 'layers', 'dr_name', 'cos_word', 'pc_word', 'pc2_word']]
    df = df_internal.merge(df_external, on=['model_name', 'layers', 'dr_name'])
    df['cos_aniso_km'] = 1 - df['cos_aniso_km']
    df['cos_word'] = 1 - df['cos_word']
    df['pc_anisotropy_2'] = 1 / (df['pc_anisotropy_2'])
    df['pc2_word'] = 1 / (df['pc2_word'])
    df = df[['NMI_km', 'NMI_skm', 'ARI_km', 'ARI_skm', 'cos_aniso_km', 'pc_anisotropy', 'pc_anisotropy_2' , 
                'cos_word', 'pc_word', 'pc2_word']]

    for method in ['pearson', 'p-value']:
        print('corr measure:', method)
        if method == 'pearson':
            corr = df.astype(float).corr(method='pearson')
            print(corr)
            for col in corr.columns:
                corr[col] = corr[col].apply(lambda x: "\gradient{" + str(round(x, 2)) + "}")
        elif method == 'p-value':
            corr = df.astype(float).corr(lambda x, y: pearsonr(x, y)[1])
            print(corr)
            for col in corr.columns:
                corr[col] = corr[col].apply(lambda x: round(x, 3) if x >= 0.001 else 'approx 0.0')
                corr.loc[col, col] = 0.0
                corr[col] = corr[col].apply(lambda x: "\gradient{" + str(x) + "}")