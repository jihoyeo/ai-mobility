# 4주차 — 분류 평가와 모델 훈련

교재 범위: 『핸즈온 머신러닝 3판』 Ch3 발췌·Ch4

## 학습 목표

- 정확도 이외의 분류 성능 지표 해석
- 선형회귀의 비용함수와 경사하강법 이해
- 배치·확률적·미니배치 경사하강법 비교
- 다항회귀와 학습곡선을 이용한 과대적합 진단
- Ridge·Lasso·Elastic Net과 조기 종료의 역할
- 로지스틱 회귀와 Softmax 회귀의 기본 구조

## 1. 분류 평가

분류 모델은 예측한 클래스와 실제 클래스를 비교해 평가한다. 클래스 비율이 불균형하면 정확도만으로 모델을 판단하기 어렵다.

### 혼동행렬

이진 분류의 결과를 네 가지로 구분한다.

|  | 실제 양성 | 실제 음성 |
|---|---:|---:|
| 예측 양성 | TP | FP |
| 예측 음성 | FN | TN |

```python
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_true, y_pred)
print(cm)
```

### 주요 지표

\[
\mathrm{Precision}=\frac{TP}{TP+FP}
\]

\[
\mathrm{Recall}=\frac{TP}{TP+FN}
\]

\[
F_1=2\frac{\mathrm{Precision}\times\mathrm{Recall}}
{\mathrm{Precision}+\mathrm{Recall}}
\]

| 지표 | 중점 | 예시 |
|---|---|---|
| 정밀도 | 양성 예측의 신뢰도 | 혼잡 경보의 오경보 최소화 |
| 재현율 | 실제 양성의 탐지율 | 사고·고장 누락 최소화 |
| F1 | 정밀도와 재현율의 조화평균 | 두 오류가 모두 중요한 경우 |

```python
from sklearn.metrics import classification_report

print(classification_report(y_true, y_pred))
```

### 임계값

분류 모델은 점수나 확률을 계산한 뒤 임계값을 기준으로 클래스를 결정한다. 임계값이 변하면 정밀도와 재현율도 함께 변한다.

```python
from sklearn.metrics import precision_recall_curve

precision, recall, thresholds = precision_recall_curve(
    y_true,
    y_score,
)
```

### ROC와 AUC

ROC 곡선은 임계값 변화에 따른 재현율과 거짓 양성 비율을 나타낸다.

\[
\mathrm{FPR}=\frac{FP}{FP+TN}
\]

```python
from sklearn.metrics import RocCurveDisplay, roc_auc_score

RocCurveDisplay.from_predictions(y_true, y_score)
auc = roc_auc_score(y_true, y_score)
```

- ROC-AUC가 1에 가까울수록 두 클래스를 잘 구분
- 양성 클래스가 매우 적으면 정밀도-재현율 곡선도 함께 확인
- 운영 목적에 따라 임계값을 별도로 선택

## 2. 선형회귀

선형회귀는 특성의 가중합과 절편으로 연속값을 예측한다.

\[
\hat{y}=\theta_0+\theta_1x_1+\cdots+\theta_nx_n
\]

벡터 표기:

\[
\hat{y}=\boldsymbol{\theta}^{\mathsf T}\mathbf{x}
\]

평균제곱오차 비용함수:

\[
\mathrm{MSE}(\boldsymbol{\theta})=
\frac{1}{m}\sum_{i=1}^{m}
(\boldsymbol{\theta}^{\mathsf T}\mathbf{x}^{(i)}-y^{(i)})^2
\]

```python
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train, y_train)

print(model.intercept_)
print(model.coef_)
```

### 정규방정식과 최소제곱

선형회귀는 해석적인 해를 구할 수 있지만, 특성 수가 많을수록 행렬 연산 비용이 증가한다. scikit-learn의 `LinearRegression`은 안정적인 최소제곱 계산을 사용한다.

## 3. 경사하강법

경사하강법은 비용함수가 감소하는 방향으로 파라미터를 반복 갱신한다.

\[
\boldsymbol{\theta}_{t+1}=
\boldsymbol{\theta}_t-eta\nabla_{\boldsymbol{\theta}}
\mathrm{MSE}(\boldsymbol{\theta}_t)
\]

`η`는 학습률이다.

| 학습률 | 결과 |
|---|---|
| 너무 작음 | 수렴에 많은 반복 필요 |
| 적절함 | 안정적인 비용 감소 |
| 너무 큼 | 최솟값을 지나치거나 발산 |

특성의 스케일 차이가 크면 비용함수의 등고선이 길게 늘어나 수렴이 느려질 수 있다. 경사하강법 사용 전 표준화를 기본으로 고려한다.

### 배치 경사하강법

한 번의 갱신에 전체 훈련 데이터를 사용한다.

```python
import numpy as np

eta = 0.1
n_epochs = 1_000
m = len(X_b)
theta = np.random.randn(X_b.shape[1], 1)

for _ in range(n_epochs):
    gradients = 2 / m * X_b.T @ (X_b @ theta - y)
    theta = theta - eta * gradients
```

- 안정적인 기울기
- 데이터가 크면 한 번의 갱신 비용 증가

### 확률적 경사하강법

한 번에 하나의 샘플로 파라미터를 갱신한다.

- 빠른 갱신과 낮은 메모리 사용
- 비용함수 주변에서 진동
- 학습률을 점차 낮추는 학습 스케줄 사용

```python
from sklearn.linear_model import SGDRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sgd = make_pipeline(
    StandardScaler(),
    SGDRegressor(
        max_iter=1_000,
        tol=1e-5,
        penalty=None,
        eta0=0.01,
        random_state=42,
    ),
)

sgd.fit(X_train, y_train)
```

### 미니배치 경사하강법

작은 샘플 묶음으로 파라미터를 갱신한다.

- 벡터화와 GPU 연산에 적합
- 배치 방식보다 잦은 갱신
- 딥러닝 학습의 기본 방식

## 4. 다항회귀

비선형 관계를 다항 특성으로 변환한 뒤 선형 모델을 적용할 수 있다.

```python
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression

poly_model = make_pipeline(
    PolynomialFeatures(degree=2, include_bias=False),
    StandardScaler(),
    LinearRegression(),
)

poly_model.fit(X_train, y_train)
```

차수가 높아질수록 표현력은 증가하지만 과대적합 위험과 특성 수가 함께 증가한다.

## 5. 학습곡선

학습곡선은 훈련 데이터 크기에 따른 훈련·검증 오차를 보여준다.

```python
from sklearn.model_selection import learning_curve

train_sizes, train_scores, valid_scores = learning_curve(
    model,
    X_train,
    y_train,
    scoring="neg_root_mean_squared_error",
    train_sizes=np.linspace(0.1, 1.0, 10),
    cv=5,
)
```

| 패턴 | 해석 |
|---|---|
| 훈련·검증 오차가 모두 높고 비슷함 | 과소적합 가능성 |
| 훈련 오차는 낮고 검증 오차와 차이가 큼 | 과대적합 가능성 |
| 데이터 증가에 따라 검증 오차가 계속 감소 | 데이터 추가 효과 기대 |

## 6. 규제

규제는 모델 파라미터의 크기에 제약을 주어 과대적합을 줄인다.

### Ridge 회귀

L2 패널티 사용:

\[
\mathrm{MSE}(\boldsymbol{\theta})+
\alpha\frac{1}{m}\sum_{j=1}^{n}\theta_j^2
\]

```python
from sklearn.linear_model import Ridge

ridge = make_pipeline(
    StandardScaler(),
    Ridge(alpha=1.0),
)
```

### Lasso 회귀

L1 패널티 사용:

\[
\mathrm{MSE}(\boldsymbol{\theta})+
2\alpha\frac{1}{m}\sum_{j=1}^{n}|\theta_j|
\]

Lasso는 일부 계수를 정확히 0으로 만들 수 있다.

```python
from sklearn.linear_model import Lasso

lasso = make_pipeline(
    StandardScaler(),
    Lasso(alpha=0.01),
)
```

### Elastic Net

L1과 L2 패널티를 결합한다.

```python
from sklearn.linear_model import ElasticNet

elastic = make_pipeline(
    StandardScaler(),
    ElasticNet(alpha=0.01, l1_ratio=0.5),
)
```

### 규제 모델 비교

| 모델 | 패널티 | 특징 |
|---|---|---|
| Ridge | L2 | 모든 특성을 유지하며 계수 축소 |
| Lasso | L1 | 일부 계수를 0으로 만들어 특성 선택 효과 |
| Elastic Net | L1+L2 | 상관된 특성이 많은 경우 안정적인 절충 |

`alpha`가 커질수록 규제가 강해진다. 규제를 사용하는 모델은 특성 스케일에 민감하므로 스케일링을 파이프라인에 포함한다.

## 7. 조기 종료

검증 오차가 더 이상 개선되지 않을 때 학습을 중단한다.

```python
sgd = SGDRegressor(
    max_iter=10_000,
    early_stopping=True,
    validation_fraction=0.2,
    n_iter_no_change=10,
    random_state=42,
)
```

조기 종료는 반복 횟수 자체를 규제 수단으로 사용한다.

## 8. 로지스틱 회귀

로지스틱 회귀는 선형 결합을 로지스틱 함수에 통과시켜 양성 클래스의 확률을 계산한다.

\[
\hat{p}=\sigma(\boldsymbol{\theta}^{\mathsf T}\mathbf{x}),
\qquad
\sigma(t)=\frac{1}{1+e^{-t}}
\]

```python
from sklearn.linear_model import LogisticRegression

classifier = make_pipeline(
    StandardScaler(),
    LogisticRegression(),
)

classifier.fit(X_train, y_train)
probability = classifier.predict_proba(X_test)
prediction = classifier.predict(X_test)
```

### 결정경계

- 예측 확률이 임계값 이상이면 양성
- 선형 특성 공간에서 선형 결정경계
- `predict_proba()`를 이용해 목적에 맞는 임계값 선택 가능

## 9. Softmax 회귀

여러 클래스의 점수를 계산한 뒤 Softmax 함수로 클래스별 확률을 구한다.

\[
\hat{p}_k=
\frac{e^{s_k(\mathbf{x})}}
{\sum_{j=1}^{K} e^{s_j(\mathbf{x})}}
\]

학습에는 교차엔트로피 손실을 사용한다.

\[
J(\Theta)=-\frac{1}{m}\sum_{i=1}^{m}
\sum_{k=1}^{K}y_k^{(i)}\log(\hat{p}_k^{(i)})
\]

```python
softmax = make_pipeline(
    StandardScaler(),
    LogisticRegression(C=10, multi_class="multinomial"),
)
```

Softmax와 교차엔트로피는 6~7주차 신경망의 출력층과 손실함수에서 다시 사용한다.

## 10. 실습

1. 불균형 이진 분류 데이터에서 정확도·정밀도·재현율·F1 비교
2. 임계값 변화에 따른 정밀도-재현율 곡선 작성
3. 배치 경사하강법으로 선형회귀 파라미터 갱신
4. 다항회귀 차수별 학습곡선 비교
5. Ridge·Lasso·Elastic Net의 계수 비교
6. 로지스틱 회귀의 확률과 결정경계 확인

## 확인 문제

1. 정확도가 높은 모델의 재현율이 낮을 수 있는 이유는 무엇인가?
2. 학습률이 너무 크거나 작을 때 각각 어떤 현상이 나타나는가?
3. 경사하강법에서 특성 스케일링이 필요한 이유는 무엇인가?
4. Ridge와 Lasso가 모델 계수에 미치는 영향은 어떻게 다른가?
5. 로지스틱 회귀가 회귀라는 이름을 가지면서 분류에 사용되는 이유는 무엇인가?

## 요약

- 분류 지표는 오류 비용과 클래스 비율을 고려해 선택
- 경사하강법은 비용함수의 기울기를 이용한 반복 최적화
- 학습곡선을 이용한 과소적합·과대적합 진단
- Ridge·Lasso·Elastic Net을 이용한 모델 복잡도 제어
- 로지스틱·Softmax 회귀의 확률 기반 분류
