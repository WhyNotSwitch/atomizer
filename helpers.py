import numpy as np

from .utils.math import fast_dot, norm_cols, norm, normalize, frobenius_squared
from .utils import set_openblas_threads


def average_mutual_coherence(D):
    n_atoms = D.shape[1]
    G = np.abs(np.dot(D.T, D))
    np.fill_diagonal(G, 0)
    return np.sum(G) / float(n_atoms * (n_atoms - 1))


def approx_error(D, Z, X, n_jobs=1):
    """computes the approximation error ||X-DZ||_{F}^{2} """
    if n_jobs > 1:
        set_openblas_threads(n_jobs)
    error = frobenius_squared(X - fast_dot(D, Z))
    return error


def force_mi(D, X, Z, unused_data, eta, max_tries=100):
    # force mutual incoherence within a dictionary
    n_atoms = D.shape[1]
    G = np.abs(np.dot(D.T, D))
    np.fill_diagonal(G, 0)

    for atom_idx1 in range(n_atoms):

        atom_idx2 = np.argmax(G[atom_idx1, :])
        # the maximum coherence
        mcoh = G[atom_idx1, atom_idx2]
        if mcoh < eta:
            print("less than the eta")
            continue
        # choose one of the two to replace
        # should we choose the one least used?
        if norm(Z[atom_idx1, :]) > norm(Z[atom_idx2, :]):
            c_atom = atom_idx1
        else:
            c_atom = atom_idx2

        # new_atom = None
        cnt = 0
        available_data = unused_data[:]
        min_idx = None
        min_coh = mcoh

        while mcoh > eta:
            # replace the coherent atom
            if cnt > max_tries:
                break
            # no datapoint available to be used as atom
            if len(available_data) == 0:
                return D
            _idx = np.random.choice(available_data, size=1)
            if len(_idx) == 0:
                return D, unused_data
            idx = _idx[0]
            new_atom = X[:, idx]
            new_atom = normalize(new_atom)
            available_data.remove(idx)
            g = np.abs(np.dot(D.T, new_atom))
            mcoh = np.max(g)
            if mcoh < min_coh or min_coh is None:
                min_coh = mcoh
                min_idx = idx
            cnt += 1

        D[:, c_atom] = X[:, min_idx]
        D[:, c_atom] = normalize(D[:, c_atom])
        unused_data.remove(min_idx)

    return D, unused_data


def init_dictionary(X, n_features, n_atoms, method='data', return_unused_data=False, normalize=True):
    """ create the initial dictionary with n_atoms method: can be {data,svd,random}"""
    if method == "svd":
        from numpy.linalg import svd
        D, S, Z = svd(X, full_matrices=False)
        Z = S[:, np.newaxis] * Z
        r = len(Z)
        if n_atoms <= r:
            D = D[:, :n_atoms]
            Z = Z[:n_atoms, :]
        else:
            D = np.c_[D, np.zeros((len(D), n_atoms - r))]
            Z = np.r_[Z, np.zeros((n_atoms - r, Z.shape[1]))]

    elif method == "data":
        # select atoms randomly from the dataset
        # make sure they have non-zero norms(to avoid singular matrices)
        from numpy.linalg import norm
        from time import time
        n_samples = X.shape[1]
        idxs = [i for i in range(n_samples) if np.sum(X[:, i] ** 2) > 1e-6]

        if len(idxs) < n_atoms:
            print("not enough datapoints to initialize the dictionary")
            raise ValueError("not enough datapoints to initialize the dictionary")

        subset = np.random.choice(len(idxs), size=n_atoms, replace=False)
        subset_idxs = np.array(idxs).astype(int)[subset]

        D = X[:, subset_idxs]
        if normalize:
            D = norm_cols(D)
        if return_unused_data:
            s = set(subset_idxs)
            unused_data = [x for x in idxs if x not in s]
            return D, unused_data

    elif method == "random":
        D = np.random.randn(n_features, n_atoms)
        D = norm_cols(D)
    return D

