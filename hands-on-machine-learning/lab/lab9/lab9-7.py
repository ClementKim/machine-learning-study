import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_digits
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression

X_digits, y_digits = load_digits(return_X_y = True)
X_train, y_train = X_digits[:1400], y_digits[:1400]
X_train, y_test = X_digits[1400:], y_digits[1400:]

n_labeled = 50

log_reg = LogisticRegression(max_iter=10_000)

log_reg.fit(X_train[:n_labeled], y_train[:n_labeled])

print(log_reg.score(X_train, y_test))

log_reg_full = LogisticRegression(max_iter=10_000)
log_reg_full.fit(X_train, y_train)
print(log_reg_full.score(X_train, y_test))

k = 50
kmeans = KMeans(n_clusters = k, n_init = 10, random_state = 42)
X_digits_dist = kmeans.fit_transform(X_train)
representative_digit_idx = np.argmin(X_digits_dist, axis = 0)
X_representative_digits = X_train[representative_digit_idx]

plt.figure(figsize=(8, 2))
for index, X_representative_digit in enumerate(X_representative_digits):
    plt.subplot(k // 10, 10, index + 1)
    plt.imshow(X_representative_digit.reshape(8, 8), cmap="binary",
               interpolation="bilinear")
    plt.axis('off')

plt.show()

y_representative_digits = np.array([
        1, 3, 6, 0, 7, 9, 2, 4, 8, 9,
        5, 4, 7, 1, 2, 6, 1, 2, 5, 1,
        4, 1, 3, 3, 8, 8, 2, 5, 6, 9,
        1, 4, 0, 6, 8, 3, 4, 6, 7, 2,
        4, 1, 0, 7, 5, 1, 9, 9, 3, 7
])

log_reg = LogisticRegression(max_iter=10_000)
log_reg.fit(X_representative_digits, y_representative_digits)
print(log_reg.score(X_train, y_test))

y_train_propagated = np.empty(len(X_train), dtype = np.int64)
for i in range(k):
    y_train_propagated[kmeans.labels_ == i] = y_representative_digits[i]

log_reg = LogisticRegression(max_iter=10_000)
log_reg.fit(X_train, y_train_propagated)
print(log_reg.score(X_train, y_test))

percentile_closest = 99

X_cluster_dist = X_digits_dist[np.arange(len(X_train)), kmeans.labels_]
for i in range(k):
    in_cluster = (kmeans.labels_ == i)
    cluster_dist = X_cluster_dist[in_cluster]
    cutoff_distance = np.percentile(cluster_dist, percentile_closest)
    above_cutoff = (X_cluster_dist > cutoff_distance)
    X_cluster_dist[in_cluster & above_cutoff] = -1

partially_propagated = (X_cluster_dist != -1)
X_train_partially_propagated = X_train[partially_propagated]
y_train_partially_propagated = y_train_propagated[partially_propagated]

log_reg = LogisticRegression(max_iter=10_000)
log_reg.fit(X_train_partially_propagated, y_train_partially_propagated)
print(log_reg.score(X_train, y_test))

print((y_train_partially_propagated == y_train[partially_propagated]).mean())
