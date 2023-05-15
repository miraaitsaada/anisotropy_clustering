import os

import numpy as np
from scipy.linalg import svd
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import confusion_matrix
from umap import UMAP
import torch
from transformers import AutoTokenizer, AutoModel

def get_embeddings(texts, model_name):
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    
    model = AutoModel.from_pretrained(model_name, output_hidden_states=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model.to(device)
    n_layers = model.config.num_hidden_layers
    
    embeddings = {'avg': [], 'last': []}
    for text in texts:
        encoded_input = tokenizer(text, return_tensors='pt')
        encoded_input.to(device)

        with torch.no_grad():
            model_output = model(**encoded_input)
            hidden_states = model_output['hidden_states'][1:]
            hidden_states = torch.cat(hidden_states)
            
            embeddings['avg'] += list(torch.mean(hidden_states, dim=0).cpu().numpy())
            embeddings['last'] += list(hidden_states[-1].cpu().numpy())

    embeddings['avg'] = np.vstack(embeddings['avg'])
    embeddings['last'] = np.vstack(embeddings['last'])
    return embeddings

def scale(X):
    matrix = X.copy()

    if scale:
        mean_ = np.mean(matrix, axis=0)
        matrix -= mean_
        std = np.std(matrix, axis=0)
        matrix /= std

    return matrix

def pca_whiten(X, n_components, scale=False):
    if scale:
        matrix = scale(X)
    else:
        matrix = X.copy()

    U, _, _ = svd(matrix)
    U = U[:, :n_components]
    return U

def pca(X, n_components, scale=False):
    if scale:
        matrix = scale(X)
    else:
        matrix = X.copy()

    U, S, _ = svd(matrix)
    U = U[:, :n_components]
    U *= S[:n_components]
    return U

def umap(X, n_components, **kwargs):
    return UMAP(n_components=n_components, random_state=42, **kwargs).fit_transform(X)

def _make_cost_m(cm):
    s = np.max(cm)
    return (- cm + s)
    
def accuracy(true_row_labels, predicted_row_labels):
    """
        Clustering (unsupervised) accuracy
    """
    cm = confusion_matrix(true_row_labels, predicted_row_labels)
    indexes = linear_sum_assignment(_make_cost_m(cm))
    total = 0
    for row, column in zip(*indexes):
        value = cm[row][column]
        total += value

    return (total * 1. / np.sum(cm))