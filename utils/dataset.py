import numpy as np
import os
import gc
import os.path

from .config import save_paths

max_mmap_files = 100000
mmap_base = os.path.join(save_paths[0], 'mmap_files/')


def get_mmap(X):
    """
    converts a numpy array to
    a numpy memmory mapped array
    """
    # TODO: use tempfile.NamedTemporaryFile
    if type(X) is np.core.memmap:
        return X
    fid = 0
    filename = mmap_base+"data"+str(fid)+".dat"
    for i in range(max_mmap_files):
        if os.path.isfile(filename):
            fid += 1
            filename = mmap_base+"data"+str(fid)+".dat"
        else:
            break

    _X = np.memmap(filename, dtype='float64', mode='w+', shape=X.shape)
    _X[:] = X[:]
    del X
    gc.collect()
    return _X


def get_empty_mmap(shape):
    """
    creates an empty memmory mapped array
    """
    fid = 0
    filename = mmap_base+"data"+str(fid)+".dat"
    for i in range(max_mmap_files):
        if os.path.isfile(filename):
            fid += 1
            filename = mmap_base+"data"+str(fid)+".dat"
        else:
            break

    return np.memmap(filename, dtype='float64', mode='w+', shape=shape)

