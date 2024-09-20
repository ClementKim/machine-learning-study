# 아주대학교 소프트웨어학과 대신러닝 스터디 기초반

Member: 김준성, 구세은, 한태호


# 대신러닝이란?

대신러닝은 머신러닝/딥러닝에 관심있는 아주대학교 학생이 방학 기간에 모여 자율적으로 운영하는 스터디입니다.


# 대신러닝 기초반 운영

2024년 여름 방학 기간 중 대신러닝 기초반은 주 3회(월/수/금) 10시부터 14시까지 공식적으로 모임을 가져 홍콩과기대 김성훈 교수님의 '모두를 위한 딥러닝'과 오렐리앙 제롱의 'Hands on Machine Learning with Scikit-Learn, Keras & TensorFlow'를 시청하고 읽으며 다음 사항을 공부하였습니다.

1. 머신러닝의 기본 개념

2. 분류

3. 모델 훈련

4. 서포트 벡터 머신 (SVM)

5. 결정 트리 (Decision Tree)

6. 앙상블 (Ensemble)

7. 랜덤 포레스트 (Random Forest)

8. 차원 축소 (Dimensionality Reduction)

9. 비지도 학습 (K-means와 DBSCAN 중심)


# To run files with venv

```sh
python3 -m venv environment_name

source environment_name/bin/active

pip install --upgrade pip

pip install -r requirements.txt

python3 file_name

deactivate
```


# About directories

```bash
├── deep-learning-zero-to-all
│   ├── data-01-test-score.csv
│   ├── data-02-stock_daily.csv
│   ├── data-03-diabetes.csv
│   ├── data-04-zoo.csv
│   ├── lab01-1.py
│   ├── lab01-2.py
│   ├── lab01-3.py
│   ├── lab02.py
│   ├── lab03-1.py
│   ├── lab03-2.py
│   ├── lab03-3.py
│   ├── lab04-1.py
│   ├── lab04-2.py
│   ├── lab05-1.py
│   ├── lab05-2.py
│   ├── lab06-1.py
│   ├── lab06-2.py
│   ├── lab07-1.py
│   ├── lab07-2.py
│   ├── lab07-3.py
│   ├── lab07-4.py
│   ├── lab09-1.py
│   ├── lab09-2.py
│   ├── lab09-3.py
│   ├── lab09-4.py
│   ├── lab10-1.py
│   ├── lab10-2.py
│   ├── lab10-3.py
│   ├── lab10-4.py
│   ├── lab10-5.py
│   ├── lab11-1.py
│   ├── lab11-2.py
│   ├── lab12-1.py
│   ├── lab12-2.py
│   ├── lab12-3.py
│   ├── lab12-4.py
│   └── lab12-5.py
├── hands-on-machine-learning
│   ├── lab
│   │   ├── lab1
│   │   │   └── lab1-4-3.py
│   │   ├── lab3
│   │   │   ├── lab3-1.py
│   │   │   ├── lab3-2.py
│   │   │   ├── lab3-3.py
│   │   │   ├── lab3-4.py
│   │   │   ├── lab3-5.py
│   │   │   ├── lab3-6.py
│   │   │   └── lab3-7.py
│   │   ├── lab4
│   │   │   ├── lab4-1.py
│   │   │   ├── lab4-2.py
│   │   │   ├── lab4-3.py
│   │   │   ├── lab4-4.py
│   │   │   ├── lab4-5.py
│   │   │   └── lab4-6.py
│   │   ├── lab5
│   │   │   ├── lab5-1.py
│   │   │   ├── lab5-2.py
│   │   │   ├── lab5-3.py
│   │   │   ├── lab5-4.py
│   │   │   └── lab5-Extra.py
│   │   ├── lab6
│   │   │   ├── lab6-1.py
│   │   │   ├── lab6-2.py
│   │   │   ├── lab6-3.py
│   │   │   └── lab6-4.py
│   │   ├── lab7
│   │   │   ├── lab7-1.py
│   │   │   ├── lab7-2.py
│   │   │   ├── lab7-3.py
│   │   │   ├── lab7-4.py
│   │   │   ├── lab7-5.py
│   │   │   ├── lab7-6.py
│   │   │   ├── lab7-7.py
│   │   │   └── lab7-8.py
│   │   ├── lab8
│   │   │   ├── lab8-1.py
│   │   │   ├── lab8-2.py
│   │   │   ├── lab8-3.py
│   │   │   ├── lab8-4.py
│   │   │   └── lab8-5.py
│   │   └── lab9
│   │       ├── lab9-1.py
│   │       ├── lab9-10.py
│   │       ├── lab9-11.py
│   │       ├── lab9-12.py
│   │       ├── lab9-2.py
│   │       ├── lab9-3.py
│   │       ├── lab9-4.py
│   │       ├── lab9-5.py
│   │       ├── lab9-6.py
│   │       ├── lab9-7.py
│   │       ├── lab9-8.py
│   │       └── lab9-9.py
│   └── prac
│       ├── datasets
│       │   ├── titanic
│       │   │   ├── test.csv
│       │   │   └── train.csv
│       │   └── titanic.tgz
│       └── prac3-3.py
```


# References

홍콩과기대 김성훈 교수님 - '모두를 위한 딥러닝'

오렐리앙 제롱 - 'Hands on Machine Learning with Scikit-Learn, Keras & TensorFlow'
