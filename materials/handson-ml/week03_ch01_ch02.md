# 3주차 — 머신러닝 개요와 End-to-End 프로젝트

교재 범위: 『핸즈온 머신러닝 3판』 Ch1·Ch2

## 학습 목표

- 지도학습·비지도학습, 회귀·분류의 구분
- 과대적합·과소적합과 일반화 성능의 의미
- 문제 정의부터 최종 평가까지의 머신러닝 프로젝트 절차
- 수치형·범주형 특성을 처리하는 전처리 파이프라인 구성
- 교차 검증과 하이퍼파라미터 탐색

## 1. 머신러닝 문제의 구성

머신러닝 프로젝트는 데이터에서 입력과 정답의 관계를 학습한 뒤, 새로운 입력에 대한 결과를 예측하는 작업이다.

| 구성 요소 | 의미 | 버스 ETA 예시 |
|---|---|---|
| 샘플 | 하나의 관측 단위 | 특정 시각의 버스 운행 상태 |
| 특성 `X` | 모델 입력 | 노선, 정류장, 시간대, 최근 구간 통행시간 |
| 타깃 `y` | 예측 대상 | 목표 정류장까지 남은 시간 |
| 모델 | 입력을 출력으로 변환하는 함수 | 선형회귀, Random Forest, XGBoost |
| 손실·평가지표 | 예측 오차의 측정 방식 | MAE, RMSE |

### 지도학습과 비지도학습

| 유형 | 데이터 | 대표 과제 |
|---|---|---|
| 지도학습 | 입력과 정답 | 회귀, 분류 |
| 비지도학습 | 입력만 제공 | 군집, 차원 축소, 이상치 탐지 |
| 준지도학습 | 일부 샘플에만 정답 | 소량의 레이블을 활용한 분류 |
| 자기지도학습 | 데이터에서 학습용 정답 생성 | 마스킹 복원, 다음 값 예측 |

수요량·통행시간·ETA는 연속값을 예측하므로 회귀 문제에 해당한다. 사고 유무나 혼잡 등급을 예측하면 분류 문제에 해당한다.

### 배치학습과 온라인학습

- 배치학습: 일정 기간의 데이터를 모아 한 번에 모델 갱신
- 온라인학습: 새 데이터가 들어올 때 점진적으로 모델 갱신
- 데이터 드리프트: 시간의 변화에 따라 입력 분포나 입력과 타깃의 관계가 달라지는 현상

이 수업의 실습은 재현 가능한 비교를 위해 배치학습을 기본으로 사용한다.

## 2. 일반화와 데이터 품질

훈련 데이터에서의 성능보다 새로운 데이터에서의 성능이 중요하다. 훈련 데이터에만 맞춘 모델은 실제 운영 환경에서 성능이 낮아질 수 있다.

### 데이터 확인 항목

- 표본 수와 관측 기간
- 대상 지역·노선·시간대의 대표성
- 결측치, 중복, 이상치
- 측정 단위와 시간대
- 타깃 생성 과정
- 예측 시점에 실제로 사용할 수 있는 특성인지 여부

### 과소적합과 과대적합

| 상태 | 훈련 성능 | 검증 성능 | 대응 |
|---|---:|---:|---|
| 과소적합 | 낮음 | 낮음 | 모델 복잡도 증가, 특성 개선, 규제 완화 |
| 적정 | 높음 | 높음 | 최종 평가 진행 |
| 과대적합 | 매우 높음 | 낮음 | 규제 강화, 모델 단순화, 데이터 추가 |

모델 파라미터는 학습 과정에서 결정되는 값이고, 하이퍼파라미터는 학습 전에 지정하는 설정값이다.

## 3. 프로젝트 절차

1. 문제와 모델 출력 정의
2. 데이터 수집과 구조 확인
3. 훈련·검증·테스트 데이터 분할
4. 탐색적 데이터 분석
5. 전처리와 특성 생성
6. 기준 모델과 후보 모델 학습
7. 교차 검증과 하이퍼파라미터 탐색
8. 테스트 데이터 최종 평가와 결과 정리

### 문제 정의

모델 학습 전에 다음 항목을 먼저 고정한다.

- 예측 단위
- 예측 시점
- 타깃과 단위
- 모델 결과의 사용 목적
- 사용할 수 있는 입력 정보
- 성능 지표
- 운영상 허용 가능한 오차

ETA 프로젝트의 경우 “현재 버스가 목표 정류장에 언제 도착하는가”만으로는 부족하다. 현재 시점의 정의, 목표 정류장 범위, 운행 방향, 예측값의 단위를 함께 정해야 한다.

## 4. 데이터 구조 확인

```python
import pandas as pd

housing = pd.read_csv("housing.csv")

housing.head()
housing.info()
housing.describe(include="all")
housing.isna().sum().sort_values(ascending=False)
```

초기 확인 순서:

1. 행과 열의 의미
2. 데이터 타입
3. 결측치
4. 범주별 빈도
5. 수치형 변수의 범위와 분포
6. 중복 행과 식별자

```python
housing["ocean_proximity"].value_counts(dropna=False)
housing.hist(bins=40, figsize=(12, 8))
```

## 5. 데이터 분할

### 무작위 분할

독립적인 샘플로 구성된 정형 데이터의 기본 분할 방식이다.

```python
from sklearn.model_selection import train_test_split

train_set, test_set = train_test_split(
    housing,
    test_size=0.2,
    random_state=42,
)
```

### 계층적 분할

중요한 범주의 비율을 훈련·테스트 데이터에 유지할 때 사용한다.

```python
import pandas as pd

housing["income_cat"] = pd.cut(
    housing["median_income"],
    bins=[0, 1.5, 3.0, 4.5, 6.0, float("inf")],
    labels=[1, 2, 3, 4, 5],
)

train_set, test_set = train_test_split(
    housing,
    test_size=0.2,
    stratify=housing["income_cat"],
    random_state=42,
)
```

### 시간 순서 분할

ETA처럼 시간에 따라 관측되는 데이터는 미래 자료가 훈련 데이터에 섞이지 않도록 날짜 순서로 분할한다.

```python
df = df.sort_values("observed_at")

train = df[df["observed_at"] < "2026-08-01"]
valid = df[(df["observed_at"] >= "2026-08-01") &
           (df["observed_at"] < "2026-08-08")]
test = df[df["observed_at"] >= "2026-08-08"]
```

테스트 데이터는 모델 선택이나 특성 설계에 사용하지 않는다.

## 6. 탐색적 데이터 분석

EDA는 모델을 학습하기 전에 데이터의 구조와 문제점을 확인하는 단계다.

### 수치형 특성

```python
numeric = train_set.select_dtypes(include="number")
corr = numeric.corr(numeric_only=True)
corr["median_house_value"].sort_values(ascending=False)
```

확인 항목:

- 타깃과의 관계
- 긴 꼬리와 극단값
- 상한·하한 처리 여부
- 서로 강하게 연관된 특성
- 단위가 다른 특성

상관계수는 선형 관계만 나타내며 인과관계를 의미하지 않는다.

### 특성 생성

원본 합계보다 비율이나 단위당 값이 유용할 수 있다.

```python
housing["rooms_per_house"] = (
    housing["total_rooms"] / housing["households"]
)
housing["bedrooms_ratio"] = (
    housing["total_bedrooms"] / housing["total_rooms"]
)
housing["people_per_house"] = (
    housing["population"] / housing["households"]
)
```

모빌리티 데이터의 예:

- 남은 정류장 수
- 남은 거리
- 최근 3개 구간의 평균 통행시간
- 동일 노선·시간대의 과거 중앙값
- 정류장 체류시간

## 7. 전처리

### 수치형 특성

- 결측치: 중앙값 대체
- 스케일: 표준화
- 긴 꼬리: 로그 변환 검토

### 범주형 특성

- 명목형 범주: 원-핫 인코딩
- 순서가 있는 범주: 순서형 인코딩
- 새로운 범주: `handle_unknown="ignore"`

```python
import numpy as np
from sklearn.compose import make_column_selector, make_column_transformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

num_pipeline = make_pipeline(
    SimpleImputer(strategy="median"),
    StandardScaler(),
)

cat_pipeline = make_pipeline(
    SimpleImputer(strategy="most_frequent"),
    OneHotEncoder(handle_unknown="ignore"),
)

preprocessing = make_column_transformer(
    (num_pipeline, make_column_selector(dtype_include=np.number)),
    (cat_pipeline, make_column_selector(dtype_include=object)),
)
```

`fit`은 훈련 데이터에서 필요한 값을 학습하고, `transform`은 학습된 값을 새 데이터에 적용한다. 검증·테스트 데이터에는 `fit`을 다시 수행하지 않는다.

## 8. 기준 모델과 후보 모델

복잡한 모델보다 먼저 단순한 기준 모델을 만든다.

```python
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline

baseline = make_pipeline(
    preprocessing,
    DummyRegressor(strategy="median"),
)

linear_model = make_pipeline(
    preprocessing,
    LinearRegression(),
)

forest_model = make_pipeline(
    preprocessing,
    RandomForestRegressor(random_state=42, n_jobs=-1),
)
```

## 9. 교차 검증

```python
from sklearn.model_selection import cross_val_score

scores = -cross_val_score(
    forest_model,
    X_train,
    y_train,
    scoring="neg_mean_absolute_error",
    cv=5,
    n_jobs=-1,
)

print(scores.mean(), scores.std())
```

시계열 데이터에는 무작위 K-fold 대신 `TimeSeriesSplit`이나 날짜 구간을 직접 지정한 분할을 사용한다.

```python
from sklearn.model_selection import TimeSeriesSplit

tscv = TimeSeriesSplit(n_splits=5)
```

## 10. 하이퍼파라미터 탐색

```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint

search = RandomizedSearchCV(
    forest_model,
    param_distributions={
        "randomforestregressor__n_estimators": randint(100, 500),
        "randomforestregressor__max_depth": randint(3, 20),
        "randomforestregressor__min_samples_leaf": randint(1, 20),
    },
    n_iter=20,
    scoring="neg_mean_absolute_error",
    cv=5,
    random_state=42,
    n_jobs=-1,
)

search.fit(X_train, y_train)
search.best_params_
```

탐색 범위와 교차 검증 방식도 실험 설정의 일부다. 최종 테스트 평가는 모델과 설정을 모두 확정한 뒤 한 번 수행한다.

## 11. 평가 지표

### MAE

\[
\mathrm{MAE}=\frac{1}{n}\sum_{i=1}^{n}|y_i-\hat{y}_i|
\]

- 원래 타깃과 같은 단위
- 해석이 단순
- 큰 오차에 대한 영향이 RMSE보다 작음

### RMSE

\[
\mathrm{RMSE}=\sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i-\hat{y}_i)^2}
\]

- 큰 오차에 더 높은 가중치
- 이상치에 민감

ETA 프로젝트의 주 지표는 MAE를 사용하고, 시간대·정류장 구간별 오차를 함께 확인한다.

## 12. 실습

1. `housing.csv`의 데이터 타입과 결측치 확인
2. 훈련·테스트 데이터 분할
3. 수치형·범주형 전처리 파이프라인 작성
4. 중앙값 기준 모델과 선형회귀 비교
5. Random Forest의 교차 검증 MAE 계산
6. 가장 큰 오차를 보인 샘플 확인

## 확인 문제

1. 검증 데이터와 테스트 데이터의 역할 차이는 무엇인가?
2. 범주형 특성에 정수 코드를 바로 부여하면 어떤 문제가 생길 수 있는가?
3. 전처리기를 전체 데이터에 먼저 `fit`하면 왜 데이터 누수인가?
4. ETA 데이터에 무작위 분할이 부적절할 수 있는 이유는 무엇인가?
5. 기준 모델이 필요한 이유는 무엇인가?

## 요약

- 모델 학습보다 먼저 예측 단위·타깃·평가지표 정의
- 데이터 분할 후 탐색·전처리 수행
- 전처리와 모델을 하나의 파이프라인으로 구성
- 교차 검증을 이용한 후보 모델 비교
- 테스트 데이터는 최종 평가에만 사용
