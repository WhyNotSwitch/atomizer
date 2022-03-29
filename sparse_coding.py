import numpy as np
from numpy.linalg import inv
from numpy.linalg.linalg import LinAlgError

from functools import partial
from scipy.optimize import nnls

from .utils import set_openblas_threads, run_parallel, cpu_count
from .utils.math import norm, fast_dot


"""A module that implements Sparse Coding algorithms"""

gram_singular_msg = "Gram matrix is singular due to linear dependencies in the dictionary"


def _omp(x, D, Gram, alpha, n_nonzero_coefs=None, tol=None):
    _, n_atoms = D.shape
    # the dict indexes of the atoms this datapoint uses
    Dx = np.array([]).astype(int)
    z = np.zeros(n_atoms)
    # the residual
    r = np.copy(x)
    i = 0
    if n_nonzero_coefs is not None:
        tol = 1e-10
        def cont_criterion():
            not_reached_sparsity = i < n_nonzero_coefs
            return (not_reached_sparsity and norm(r) > tol)
    else:
        cont_criterion = lambda: norm(r) >= tol

    while (cont_criterion()):

        # find the atom that correlates the
        # most with the residual
        k = np.argmax(np.abs(alpha))
        if k in Dx:
            break
        Dx = np.append(Dx, k)
        # solve the Least Squares problem to find the coefs z
        G = Gram[Dx, :][:, Dx]
        G = np.atleast_2d(G)
        try:
            G_inv = inv(G)
        except LinAlgError:
            print(gram_singular_msg)
            break

        z[Dx] = np.dot(G_inv, np.dot(D.T, x)[Dx])
        r = x - np.dot(D[:, Dx], z[Dx])
        alpha = np.dot(D.T, r)
        i += 1

    return z

def omp(X, Alpha, D, Gram, n_nonzero_coefs=None, tol=None):
    n_samples = X.shape[1]
    n_atoms = D.shape[1]
    Z = np.zeros((n_atoms, n_samples))
    for i in range(n_samples):
        Z[:, i] = _omp(X[:, i], D, Gram, Alpha[:, i], n_nonzero_coefs=n_nonzero_coefs, tol=tol)
    return Z


def nn_omp(X, D, n_nonzero_coefs=None, tol=None):
    """ The Non Negative OMP algorithm of
        'On the Uniqueness of Nonnegative Sparse Solutions to Underdetermined Systems of Equations'"""

    n_samples = X.shape[1]
    n_atoms = D.shape[1]
    Z = np.zeros((n_atoms, n_samples))
    _norm = np.sum(D ** 2, axis=0)
    for i in range(n_samples):

        x = X[:, i]
        r = x
        z = np.zeros(n_atoms)
        Dx = np.array([]).astype(int)
        j = 0
        if n_nonzero_coefs is not None:
            tol = 1e-20

            def cont_criterion():
                not_reached_sparsity = j < n_nonzero_coefs
                return (not_reached_sparsity and norm(r) > tol)
        else:
            # tol = 1e-20
            cont_criterion = lambda: norm(r) > tol

        while (cont_criterion()):
            a = np.dot(D.T, r)
            a[a < 0] = 0
            e = (norm(r) ** 2) - (a ** 2) / _norm
            k = np.argmin(e)
            Dx = np.append(Dx, k)

            z_est = nnls(D[:, Dx], x)[0]
            r = x - np.dot(D[:, Dx], z_est)
            j += 1

        if j != 0:
            z[Dx] = z_est
        Z[:, i] = z
    return Z


class sparse_encoder(object):
    """
    A class that interfaces the functions defined above.
    The user must specify the Sparse Coding algorithm and it's
    parameters in the param dictionary.

    algorithm can be one of the following:

    'omp' => Orthogonal Matching Pursuit with Least Sqaures

             params:
                    n_nonzero_coefs: the number of non-zero coefficients
                                     of the sparse representations (i.e sparsity)

                    tol: the error bound that should be achieved
                         in the approximation


    'bomp' => Batch Orthogonal Matching Pursuit algorithm

             params:
                    n_nonzero_coefs: the number of non-zero coefficients
                                     of the sparse representations (i.e sparsity)

                    tol: to be implemented

    'nnomp' => Non-Negative Orthogonal Matching Pursuit algorithm. Solves the
               l0 problem like 'omp' and 'bomp' but enforce the solutions to
               be non-negative vectors.

               params: (same as 'omp' and 'bomp')




    'iht' => Iterative Hard Thresholding

             params:
                    learning_rate: the learning rate of the gradient procedure

                    n_iter: the number of iterations

                    threshold: the threshold of the hard thresholding operator


    'lasso' => Least Absolute Shrinkage and Selection operator

               params:
                       lambda: the l1 penalty parameter


    'somp' => Simultaneous Orthogonal Matching Pursuit. It jointly encodes signals
               of the same group.

              params:

                     data_groups:  a list of the datapoint indices
                                   that belong to the same group

                     n_nonzero_coefs: the number of non-zero coefficients
                                      of the sparse representations (i.e sparsity)

    'group_omp' => sparsity constraint Group Orthogonal Matching Pursuit as described in
                     "Aurelie C. Lozano, Grzegorz Swirszcz, Naoki Abe:  Group Orthogonal Matching Pursuit for
                   Variable Selection and Prediction"

                   params:

                          groups:   a list of the atom indices
                                      that belong to the same group

                          n_groups  the number of atom groups to be selected
                                      per atom
    """

    def __init__(self, algorithm='nnomp', params=None, n_jobs=1, verbose=True, mmap=False, name='sparse_coder'):
        self.name = name

        self.algorithm = algorithm
        self.params = params
        if self.params is None:
            self.params = {}
        if n_jobs == -1:
            n_jobs = cpu_count
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.mmap = mmap

    def encode(self, X, D):
        return self.__call__(X, D)

    def __call__(self, X, D):
        # assume X has datapoints in columns
        # use self.params.get('key') because it does not throw exception
        # when the key does not exist, it just returns None.

        n_samples = X.shape[1]
        n_atoms = D.shape[1]
        n_batches = 100

        if self.params.get('lambda') is not None:
            assert self.params.get('lambda') <= n_atoms

        if self.n_jobs > 1:
            set_openblas_threads(self.n_jobs)

        batched_args = None


        if self.algorithm == 'omp':
            Gram = fast_dot(D.T, D)
            args = [D, Gram]
            Alpha = fast_dot(D.T, X)
            batched_args = [Alpha]
            data = X
            func = partial(omp, n_nonzero_coefs=self.params.get('n_nonzero_coefs'), tol=self.params.get('tol'))

        elif self.algorithm == "nnomp":
            args = [D]
            data = X
            func = partial(nn_omp, n_nonzero_coefs=self.params.get('n_nonzero_coefs'), tol=self.params.get('tol'))

        if self.verbose:
            msg = "sparse coding"
        else:
            msg = None

        if self.n_jobs > 1:
            # disable OpenBLAS to
            # avoid the hanging problem
            set_openblas_threads(1)

        Z = run_parallel(func=func, data=data, args=args, batched_args=batched_args,
                         result_shape=(n_atoms, n_samples), n_batches=n_batches,
                         mmap=self.mmap, msg=msg, n_jobs=self.n_jobs)

        # restore the previous setting
        if self.n_jobs > 1:
            set_openblas_threads(self.n_jobs)

        return Z
