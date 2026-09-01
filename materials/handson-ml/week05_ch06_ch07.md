# 5주차 — 결정 트리와 앙상블 학습

교재 범위: 『핸즈온 머신러닝 3판』 Ch6·Ch7

## 학습 목표

- 결정 트리의 분할 기준과 예측 과정 이해
- 결정 트리의 과대적합 제어
- 배깅·랜덤 포레스트·부스팅의 차이 설명
- Random Forest와 XGBoost를 정형 데이터에 적용
- 교차 검증과 시간 순서 분할을 이용한 모델 비교
- 특성 중요도와 구간별 오차 분석

## 1. 결정 트리

결정 트리는 특성에 대한 조건을 순서대로 적용해 예측값을 결정한다. 수치형·범주형 특성의 비선형 관계와 특성 간 상호작용을 표현할 수 있다.

```python
from sklearn.tree import DecisionTreeRegressor

tree = DecisionTreeRegressor(
    max_depth=4,
    min_samples_leaf=20,
    random_state=42,
)

tree.fit(X_train, y_train)
y_pred = tree.predict(X_valid)
```

### 트리의 구성

| 용어 | 의미 |
|---|---|
| 루트 노드 | 전체 훈련 샘플 |
| 내부 노드 | 특성 조건을 이용한 분할 |
| 리프 노드 | 최종 예측값 또는 클래스 |
| 깊이 | 루트에서 리프까지의 분할 수 |

회귀 트리의 리프 예측값은 해당 리프에 속한 훈련 샘플 타깃의 평균이다.

## 2. CART 분할

scikit-learn의 결정 트리는 이진 분할을 반복한다. 회귀에서는 분할 후 제곱오차가 작아지는 조건을 선택한다.

\[
J(k,t_k)=
\frac{m_{\text{left}}}{m}\mathrm{MSE}_{\text{left}}
+
\frac{m_{\text{right}}}{m}\mathrm{MSE}_{\text{right}}
\]

분류에서는 Gini 불순도나 엔트로피를 사용할 수 있다.

\[
G_i=1-\sum_{k=1}^{K}p_{i,k}^2
\]

```python
from sklearn.tree import DecisionTreeClassifier

classifier = DecisionTreeClassifier(
    criterion="gini",
    max_depth=5,
    random_state=42,
)
```

### 트리 시각화

```python
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

plt.figure(figsize=(16, 8))
plot_tree(
    tree,
    feature_names=X_train.columns,
    filled=True,
    rounded=True,
)
plt.show()
```

특성이 원-핫 인코딩되거나 파이프라인에서 변환되면 `get_feature_names_out()`으로 변환 후 특성 이름을 확인한다.

## 3. 결정 트리의 특성

### 장점

- 스케일링 없이 사용 가능
- 비선형 관계와 특성 상호작용 표현
- 회귀와 분류에 모두 사용
- 예측 경로와 특성 중요도 확인 가능

### 한계

- 데이터의 작은 변화에 민감
- 깊은 트리의 과대적합
- 축에 수직인 분할로 인한 회전 민감성
- 리프별 상수 예측으로 인한 매끄럽지 않은 회귀 결과

### 규제 하이퍼파라미터

| 하이퍼파라미터 | 역할 |
|---|---|
| `max_depth` | 최대 깊이 제한 |
| `min_samples_split` | 내부 노드 분할에 필요한 최소 샘플 수 |
| `min_samples_leaf` | 리프의 최소 샘플 수 |
| `max_leaf_nodes` | 리프 노드 수 제한 |
| `max_features` | 분할마다 검토하는 특성 수 제한 |
| `ccp_alpha` | 비용 복잡도 가지치기 강도 |

## 4. 앙상블 학습

앙상블은 여러 모델의 예측을 결합한다. 개별 모델의 오류가 완전히 같지 않다면 결합 결과가 더 안정적일 수 있다.

| 방법 | 학습 방식 | 결합 방식 |
|---|---|---|
| 투표 | 서로 다른 모델을 병렬 학습 | 다수결·확률 평균 |
| 배깅 | 재표본 데이터로 같은 모델을 병렬 학습 | 평균·다수결 |
| 부스팅 | 이전 모델의 오차를 다음 모델이 보완 | 가중합 |
| 스태킹 | 여러 모델의 예측을 새 특성으로 사용 | 메타 모델 |

## 5. 투표 앙상블

```python
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

voting = VotingClassifier(
    estimators=[
        ("lr", LogisticRegression()),
        ("rf", RandomForestClassifier(random_state=42)),
        ("svc", SVC(probability=True, random_state=42)),
    ],
    voting="soft",
)
```

- Hard voting: 각 모델의 클래스 예측을 다수결로 결합
- Soft voting: 클래스별 예측 확률을 평균
- 확률의 품질이 충분하면 Soft voting이 더 많은 정보를 사용

회귀에는 `VotingRegressor`를 사용할 수 있다.

## 6. 배깅과 페이스팅

- 배깅: 복원추출로 여러 훈련 세트 생성
- 페이스팅: 비복원추출로 여러 훈련 세트 생성

```python
from sklearn.ensemble import BaggingRegressor
from sklearn.tree import DecisionTreeRegressor

bagging = BaggingRegressor(
    estimator=DecisionTreeRegressor(),
    n_estimators=200,
    max_samples=0.8,
    bootstrap=True,
    n_jobs=-1,
    random_state=42,
)

bagging.fit(X_train, y_train)
```

개별 트리는 높은 분산을 가질 수 있지만, 여러 트리의 평균은 분산을 낮춘다.

### OOB 평가

배깅 과정에서 특정 모델의 훈련에 사용되지 않은 샘플을 Out-of-Bag 샘플이라고 한다. OOB 샘플로 별도의 검증 세트 없이 일반화 성능을 추정할 수 있다.

```python
bagging = BaggingRegressor(
    estimator=DecisionTreeRegressor(),
    n_estimators=200,
    bootstrap=True,
    oob_score=True,
    n_jobs=-1,
    random_state=42,
)
```

시간 순서가 중요한 데이터에서는 OOB 평가가 미래 예측 상황을 그대로 재현하지 못한다. ETA 모델 평가는 날짜 기준 검증을 우선한다.

## 7. Random Forest

Random Forest는 배깅한 결정 트리에서 각 분할마다 일부 특성만 무작위로 선택한다. 트리 사이의 상관을 줄여 앙상블의 분산을 낮춘다.

```python
from sklearn.ensemble import RandomForestRegressor

forest = RandomForestRegressor(
    n_estimators=500,
    max_depth=None,
    min_samples_leaf=5,
    max_features=0.7,
    n_jobs=-1,
    random_state=42,
)

forest.fit(X_train, y_train)
```

### Extra Trees

Extra Trees는 분할 임계값까지 더 무작위로 선택한다. 일반적으로 훈련이 빠르고 분산이 낮지만, 편향은 증가할 수 있다.

```python
from sklearn.ensemble import ExtraTreesRegressor

extra = ExtraTreesRegressor(
    n_estimators=500,
    min_samples_leaf=5,
    n_jobs=-1,
    random_state=42,
)
```

## 8. 특성 중요도

### 불순도 기반 중요도

```python
import pandas as pd

importance = pd.Series(
    forest.feature_importances_,
    index=X_train.columns,
).sort_values(ascending=False)
```

불순도 기반 중요도는 범주 수가 많거나 연속값이 다양한 특성에 높은 값을 줄 수 있다.

### 순열 중요도

특성 값을 무작위로 섞었을 때 검증 성능이 얼마나 나빠지는지 측정한다.

```python
from sklearn.inspection import permutation_importance

result = permutation_importance(
    forest,
    X_valid,
    y_valid,
    scoring="neg_mean_absolute_error",
    n_repeats=10,
    random_state=42,
    n_jobs=-1,
)
```

상관된 특성이 여러 개 있으면 하나를 섞어도 다른 특성이 정보를 대신할 수 있어 중요도가 낮게 나타날 수 있다.

## 9. AdaBoost

AdaBoost는 앞선 모델이 틀린 샘플에 더 높은 가중치를 주며 약한 학습기를 순차적으로 추가한다.

```python
from sklearn.ensemble import AdaBoostRegressor

ada = AdaBoostRegressor(
    n_estimators=200,
    learning_rate=0.05,
    random_state=42,
)
```

- 얕은 결정 트리를 약한 학습기로 사용
- `learning_rate`와 모델 수 사이의 절충
- 이상치와 잘못된 레이블에 민감할 수 있음

## 10. Gradient Boosting

Gradient Boosting은 이전 모델의 잔차를 다음 모델이 예측하도록 순차적으로 모델을 추가한다.

초기 예측:

\[
\hat{y}^{(0)}=\operatorname{mean}(y)
\]

각 단계의 잔차:

\[
r_i^{(t)}=y_i-\hat{y}_i^{(t-1)}
\]

업데이트:

\[
\hat{y}^{(t)}=
\hat{y}^{(t-1)}+\eta h_t(\mathbf{x})
\]

```python
from sklearn.ensemble import GradientBoostingRegressor

gbr = GradientBoostingRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=3,
    random_state=42,
)
```

### HistGradientBoosting

특성 값을 구간으로 묶어 대규모 정형 데이터의 학습 속도를 높인다.

```python
from sklearn.ensemble import HistGradientBoostingRegressor

hist_gbr = HistGradientBoostingRegressor(
    learning_rate=0.05,
    max_iter=300,
    max_leaf_nodes=31,
    l2_regularization=1.0,
    random_state=42,
)
```

## 11. XGBoost

XGBoost는 Gradient Boosting에 규제, 결측치 처리, 병렬 계산, 조기 종료 등을 추가한 구현이다.

```python
from xgboost import XGBRegressor

xgb = XGBRegressor(
    n_estimators=1_000,
    learning_rate=0.03,
    max_depth=6,
    min_child_weight=5,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    objective="reg:squarederror",
    random_state=42,
    n_jobs=-1,
)

xgb.fit(
    X_train,
    y_train,
    eval_set=[(X_valid, y_valid)],
    verbose=False,
)
```

주요 하이퍼파라미터:

| 항목 | 역할 |
|---|---|
| `n_estimators` | 트리 수 |
| `learning_rate` | 각 트리의 반영 비율 |
| `max_depth` | 개별 트리의 깊이 |
| `min_child_weight` | 자식 노드 생성 제약 |
| `subsample` | 각 트리에 사용할 샘플 비율 |
| `colsample_bytree` | 각 트리에 사용할 특성 비율 |
| `reg_alpha`, `reg_lambda` | L1·L2 규제 |

작은 `learning_rate`에는 일반적으로 더 많은 트리가 필요하다. 최적 반복 횟수는 검증 데이터와 조기 종료로 결정한다.

## 12. 스태킹

스태킹은 여러 기본 모델의 예측을 메타 모델의 입력으로 사용한다.

```python
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import Ridge

stacking = StackingRegressor(
    estimators=[
        ("forest", forest),
        ("extra", extra),
        ("hist_gbr", hist_gbr),
    ],
    final_estimator=Ridge(alpha=1.0),
    cv=5,
    n_jobs=-1,
)
```

메타 모델 학습에는 기본 모델이 해당 샘플을 보지 않고 만든 교차 검증 예측을 사용해야 한다. 같은 데이터에 학습하고 같은 데이터에 예측한 값을 사용하면 누수가 발생한다.

## 13. ETA 모델 적용

### 정형 특성 예시

- 시간대와 요일
- 현재 정류장과 목표 정류장
- 남은 정류장 수와 거리
- 최근 구간 평균·중앙값 통행시간
- 동일 노선·시간대의 과거 통행시간
- 현재까지의 지연 정도

### 날짜 기준 분할

```python
train = eta[eta["service_date"] < "2026-08-01"]
valid = eta[(eta["service_date"] >= "2026-08-01") &
            (eta["service_date"] < "2026-08-08")]
test = eta[eta["service_date"] >= "2026-08-08"]
```

### 기준 모델

```python
group_cols = ["route_id", "current_stop_id", "target_stop_id", "hour"]

median_table = (
    train.groupby(group_cols)["eta_seconds"]
    .median()
    .rename("baseline_eta")
)

valid_with_baseline = valid.join(median_table, on=group_cols)
```

### 평가

```python
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

mae = mean_absolute_error(y_valid, y_pred)
rmse = root_mean_squared_error(y_valid, y_pred)
```

전체 점수 외에 다음 그룹의 오차를 함께 확인한다.

- 출근·퇴근·비첨두 시간대
- 현재 정류장에서 목표 정류장까지의 거리 구간
- 노선의 혼잡 구간
- 평일·주말
- 큰 오차가 발생한 운행 사례

## 14. 모델 비교 표

| 모델 | 기준 검증 MAE | 장점 | 확인 사항 |
|---|---:|---|---|
| 과거 중앙값 |  | 해석이 단순 | 그룹별 표본 수 |
| 선형회귀 |  | 빠른 학습, 계수 해석 | 비선형 관계 |
| 결정 트리 |  | 규칙 확인 가능 | 높은 분산 |
| Random Forest |  | 안정적인 비선형 모델 | 모델 크기 |
| XGBoost |  | 정형 데이터의 높은 성능 | 튜닝·과대적합 |

성능 수치는 동일한 데이터 분할과 동일한 특성으로 비교한다.

## 15. 실습

1. 결정 트리의 `max_depth`에 따른 훈련·검증 MAE 비교
2. Random Forest의 OOB 점수와 날짜 기준 검증 점수 비교
3. 불순도 기반 중요도와 순열 중요도 비교
4. Gradient Boosting의 `learning_rate`와 `n_estimators` 조합 비교
5. XGBoost 기준 모델 작성
6. 시간대별·구간별 오차 표 생성

## 확인 문제

1. 결정 트리가 특성 스케일링 없이도 학습 가능한 이유는 무엇인가?
2. 배깅이 개별 결정 트리보다 안정적인 이유는 무엇인가?
3. Random Forest가 일반 배깅 트리와 다른 점은 무엇인가?
4. Gradient Boosting에서 학습률과 트리 수는 어떤 관계인가?
5. ETA 데이터에서 OOB 점수보다 날짜 기준 검증이 중요한 이유는 무엇인가?
6. 특성 중요도가 인과관계를 의미하지 않는 이유는 무엇인가?

## 요약

- 결정 트리는 비선형 관계와 특성 상호작용을 표현
- 깊이·리프 크기·가지치기를 이용한 과대적합 제어
- 배깅과 Random Forest를 이용한 분산 감소
- 부스팅을 이용한 순차적 오차 보완
- 정형 데이터의 주요 후보 모델로 Random Forest·XGBoost 사용
- ETA에서는 날짜 기준 검증과 기준 모델 비교가 필수
