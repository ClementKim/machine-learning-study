import numpy as np

from sklearn.mixture import GaussianMixture
from sklearn.datasets import make_blobs

# 1000개의 샘플과 2개의 중심을 가진 합성 데이터셋 생성
# make_blobs(): 가우시안 혼합 모델을 위한 합성 데이터셋 생성
# 합성 데이터셋: 실제 데이터를 모방한, 인간이 생성하지 않은 데이터셋
X1, y1 = make_blobs(n_samples=1000, centers=((4, -4), (0, 0)), random_state=42)

# 데이터셋에 선형 변환 적용
# dot(): 행렬 곱셈
X1 = X1.dot(np.array([[0.374, 0.95], [0.732, 0.598]]))

# 250개의 샘플과 1개의 중심을 가진 또 다른 합성 데이터셋 생성
X2, y2 = make_blobs(n_samples=250, centers=1, random_state=42)

# 두 번째 데이터셋 이동
X2 += [6, -8]

# 두 데이터셋을 하나로 결합
# r_[]: 배열을 좌우로 결합
X = np.r_[X1, X2]
y = np.r_[y1, y2]

# 3개의 컴포넌트를 가진 가우시안 혼합 모델 생성
# GaussianMixture(): 가우시안 혼합 모델 생성
# n_components: 클러스터 개수
# n_init: 초기화 실행 횟수
gm = GaussianMixture(n_components=3, n_init=10, random_state=42)

# 결합된 데이터셋에 모델 피팅
gm.fit(X)

# 가우시안 혼합 모델의 각 컴포넌트의 가중치 출력
print(gm.weights_)

# 가우시안 혼합 모델의 각 컴포넌트의 평균 출력
print(gm.means_)

# 가우시안 혼합 모델의 각 컴포넌트의 공분산 행렬 출력
print(gm.covariances_)

# 모델 수렴 여부 출력
print(gm.converged_)

# 모델 반복 횟수 출력
print(gm.n_iter_)

# 각 샘플에 대한 예측된 클러스터 레이블 출력
print(gm.predict(X))

# 각 샘플에 대한 클러스터 소속 확률 출력 (소수점 셋째 자리까지 반올림)
print(gm.predict_proba(X).round(3))

# gm 모델에서 새로운 샘플 6개 생성
# 반환된 샘플은 클러스터 인덱스순으로 정렬되어 있음
# sample(): 새로운 샘플 생성
# 가우스 혼합 모델은 생성 모델임을 보여주기 위함
X_new, y_new = gm.sample(6)
print(X_new)
print(y_new)

# 각 샘플에 대한 로그 확률 밀도 출력
# score_samples(): 각 샘플에 대한 로그 확률 밀도를 계산, 점수가 높을수록 밀도가 높음
print(gm.score_samples(X).round(2))