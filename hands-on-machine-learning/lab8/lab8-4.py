import numpy as np

from sklearn.random_projection import johnson_lindenstrauss_min_dim
from sklearn.random_projection import GaussianRandomProjection
from sklearn.random_projection import SparseRandomProjection

m, ε = 5_000, 0.1
d = johnson_lindenstrauss_min_dim(m, eps=ε)

print(d)

d = int(4 * np.log(m) / (ε ** 2 / 2 - ε ** 3 / 3))
print(d)

n = 20_000
np.random.seed(42)
P = np.random.randn(d, n) / np.sqrt(d)  # std dev = square root of variance

X = np.random.randn(m, n)  # generate a fake dataset
X_reduced = X @ P.T

gaussian_rnd_proj = GaussianRandomProjection(eps=ε, random_state=42)
X_reduced = gaussian_rnd_proj.fit_transform(X)

components_pinv = np.linalg.pinv(gaussian_rnd_proj.components_)
X_recovered = X_reduced @ components_pinv.T

print("GaussianRandomProjection fit")
print("SparseRandomProjection fit")

gaussian_rnd_proj = GaussianRandomProjection(random_state=42).fit(X)
sparse_rnd_proj = SparseRandomProjection(random_state=42).fit(X)
print("GaussianRandomProjection transform")
print("SparseRandomProjection transform")