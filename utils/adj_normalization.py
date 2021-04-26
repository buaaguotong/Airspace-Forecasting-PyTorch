
import pandas as pd
import numpy as np
import scipy.sparse as sp

def calculate_normalized_laplacian(adj):
    """
    # D = diag(A 1)
    # L = D^-1/2 (D-A) D^-1/2 = I - D^-1/2 A D^-1/2
    """
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    normalized_laplacian = sp.eye(adj.shape[0]) - adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
    return normalized_laplacian.astype(np.float32).todense()


def asym_adj(adj):
    """
    # D = diag(A 1)
    # P = D^-1 A
    """
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1)).flatten()
    d_inv = np.power(rowsum, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat= sp.diags(d_inv)
    return d_mat.dot(adj).astype(np.float32).todense()


def load_adj(file_path, adjtype):
    df = pd.read_csv(file_path, header=None)
    adj_mx = df.to_numpy().astype(np.float32)
    if adjtype == "normlap":
        adj = [calculate_normalized_laplacian(adj_mx)]
    elif adjtype == "doubletransition":
        adj = [asym_adj(adj_mx), asym_adj(np.transpose(adj_mx))]
    return adj

if __name__ == "__main__":
    adj = load_adj('../data/adj_mx_geo_126.csv', 'normlap')
    print(adj)