import numpy as np

from sklearn.decomposition import PCA

m = 60
X = np.zeros((m, 3))

X_centered = X - X.mean(axis=0)
U, s, Vt = np.linalg.svd(X_centered)
c1 = Vt[0]
c2 = Vt[1]

m, n = X.shape
sigma = np.zeros_like(X_centered)
sigma[:n, :n] = np.diag(s)
assert np.allclose(X_centered, U @ sigma @ Vt)

W2 = Vt[:2].T
X2D = X_centered @ W2

pca = PCA(n_components=2)
X2D = pca.fit_transform(X)

print(pca.components_)

print(pca.explained_variance_ratio_)

print(1 - pca.explained_variance_ratio_.sum())