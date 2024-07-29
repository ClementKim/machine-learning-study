import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import add_dummy_feature
from sklearn.preprocessing import PolynomialFeatures


np.random.seed(42)
m = 100                               # 샘플 개수
X = 2 * np.random.rand(m, 1)          # 열 벡터
y = 4 + 3 * X + np.random.randn(m, 1) # 열 벡터

X_b = add_dummy_feature(X) # 각 샘플에 x0 = 1 추가
theta_best = np.linalg.inv(X_b.T @ X_b ) @ X_b.T @ y # A @ B = np.matmul(A, B)

X_new = np.array([[0], [2]])
X_new_b = add_dummy_feature(X_new) # 각 샘플에 X0 = 1 추가
y_predict = X_new_b @ theta_best


plt.plot(X_new, y_predict, "r-", label = "prediction")
plt.plot(X, y, "b.")

# add label, axis, grid, range
plt.xlabel("$x_1$")
plt.ylabel("$y$", rotation=0)
plt.axis([0, 2, 0, 15])
plt.grid()
plt.legend(loc="upper left")

plt.show()

lin_reg = LinearRegression()
lin_reg.fit(X, y)
print(lin_reg.intercept_, lin_reg.coef_)
print(lin_reg.predict(X_new))

theta_best_svd, residuals, rank, s = np.linalg.lstsq(X_b, y, rcond=1e-6)
print(theta_best_svd)

etas = [0.1, 0.02, 0.5]  # 학습률
n_epochs = 1000
m = len(X_b)  # 샘플 개수

theta = np.random.randn(2, 1)  # 모델 파라미터를 랜덤하게 초기화

for eta in etas:
    for epoch in range(n_epochs):
        gradients = 2 / m * X_b.T @ (X_b @ theta - y)
        theta = theta - eta * gradients

    print(f"eta: {eta}, theta: {theta}")

n_epochs = 50
t0, t1 = 5, 50  # 학습 스케줄 하이퍼 파라미터


def learning_schedule(t):
    return t0 / (t + t1)


np.random.seed(42)
theta = np.random.randn(2, 1)  # 랜덤 초기화

for epoch in range(n_epochs):
    for iteration in range(m):
        random_index = np.random.randint(m)
        xi = X_b[random_index: random_index + 1]
        yi = y[random_index: random_index + 1]
        gradients = 2 * xi.T @ (xi @ theta - yi)
        eta = learning_schedule(epoch * m + iteration)
        theta = theta - eta * gradients

print(f"theta: {theta}")

# 잡음을 추가한 비선형 데이터
# y = 0.5 * X ** 2 + X + 가우스 잡음
np.random.seed(42)
m = 100
X = 6 * np.random.rand(m, 1) - 3
y = 0.5 * X ** 2 + X + np.random.randn(m, 1)

#X_poly로 X와 X의 특성의 제곱을 가지는 array생성
poly_features = PolynomialFeatures(degree = 2, include_bias = False)
X_poly = poly_features.fit_transform(X)
X[0]
X_poly[0]
"""
array([-0.75275929,  0.56664654])
"""

# LinearRegression적용하기
# 예측값 = 0.56*X ** 2 + 0.93 *x + 1.78
lin_reg = LinearRegression()
lin_reg.fit(X_poly, y)
lin_reg.intercept_, lin_reg.coef_
"""
(array([1.78134581]), array([[0.93366893, 0.56456263]]))
"""

#그래프로 확인
# 정의역은 (-3,3)으로 설정
X_new = np.linspace(-3, 3, 100).reshape(100, 1)
X_new_poly = poly_features.transform(X_new)
y_new = lin_reg.predict(X_new_poly)

plt.figure(figsize=(6, 4))
plt.plot(X, y, "b.", label = "non-linear data")
plt.plot(X_new, y_new, "r-", linewidth=2, label="Predictions")
plt.xlabel("$x_1$")
plt.ylabel("$y$", rotation=0)
plt.legend(loc="upper left")
plt.axis([-3, 3, -2, 10])
plt.grid()
plt.show()