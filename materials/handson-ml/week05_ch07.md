# 핸즈온 ML Ch7 — 앙상블 학습과 랜덤 포레스트

- 교재: 『핸즈온 머신러닝 3판』 1부 7장 (pp. 268~296)
- 사용 주차: 5주차 후반 (앙상블, Random Forest, 부스팅)
- 관련 슬라이드: [slides/week05/deck.md](../../slides/week05/deck.md)
- 교재 노트북: `07_ensemble_learning_and_random_forests.ipynb`

> 랜덤으로 선택된 수천 명의 사람에게 복잡한 질문을 하고 대답을 모은다고 가정합시다. 많은 경우 이렇게 모은 답이 전문가의 답보다 낫습니다. 이를 **대중의 지혜**라고 합니다. 이와 비슷하게 일련의 예측기(즉, 분류나 회귀 모델)로부터 예측을 수집하면 가장 좋은 모델 하나보다 더 좋은 예측을 얻을 수 있을 것입니다. 일련의 예측기를 **앙상블**이라고 부르기 때문에 이를 **앙상블 학습**이라고 하며, 앙상블 학습 알고리즘을 **앙상블 방법**이라고 합니다.

> (2장에서 언급한 것처럼) 프로젝트의 마지막에 다다르면 흔히 앙상블 방법을 사용하여 여러 괜찮은 예측기를 연결하여 더 좋은 예측기를 만듭니다. 사실 **머신러닝 경연 대회에서 우승하는 솔루션은 여러 가지 앙상블 방법을 사용한 경우가 많습니다.**

## 장의 구성

| 절 | 제목 | 교재 쪽 |
|---|---|---:|
| 7.1 | 투표 기반 분류기 | 269 |
| 7.2 | 배깅과 페이스팅 | 273 |
| 7.3 | 랜덤 패치와 랜덤 서브스페이스 | 278 |
| 7.4 | 랜덤 포레스트 | 278 |
| 7.5 | 부스팅 | 281 |
| 7.6 | 스태킹 | 292 |

> **모빌리티 적용.** 이 장은 프로젝트에서 **XGBoost 기준 모델**을 만드는 근거를 전부 담고 있다. 7.5.3의 히스토그램 기반 그레이디언트 부스팅(HGB)과 그 TIP에 나오는 XGBoost가 우리가 실제로 쓸 도구다.

---

## 7.1 투표 기반 분류기

정확도가 80%인 분류기 여러 개(로지스틱 회귀, SVM, 랜덤 포레스트, k-최근접 이웃 등)를 훈련시켰다고 가정하자. 더 좋은 분류기를 만드는 매우 간단한 방법은 **각 분류기의 예측을 집계하는 것**이다. 가장 많은 표를 얻은 클래스가 앙상블의 예측이 된다. 이렇게 다수결 투표로 정해지는 분류기를 **직접 투표**(hard voting) 분류기라고 한다.

조금 놀랍게도 이 다수결 투표 분류기가 앙상블에 포함된 개별 분류기 중 가장 뛰어난 것보다도 정확도가 높은 경우가 많다. 사실 **각 분류기가 약한 학습기**(랜덤 추측보다 조금 더 높은 성능을 내는 분류기)일지라도 **앙상블에 있는 약한 학습기가 충분하게 많고 다양하다면 앙상블은 (높은 정확도를 내는) 강한 학습기**가 될 수 있다.

### 큰 수의 법칙

앞면이 51%, 뒷면이 49%가 나오는 조금 균형이 맞지 않는 동전이 있다고 가정하자.

| 던진 횟수 | 앞면이 다수가 될 확률 |
|---:|---:|
| 1,000번 | 약 75% |
| 10,000번 | 97% 이상 |

이는 **큰 수의 법칙** 때문이다. 동전을 자꾸 던질수록 앞면이 나오는 비율은 점점 더 앞면이 나올 확률(51%)에 가까워진다.

이와 비슷하게 (랜덤 추측보다 조금 더 나은) 51% 정확도를 가진 1,000개의 분류기로 앙상블 모델을 구축한다고 가정하면 가장 많은 클래스를 예측으로 삼는다면 75%의 정확도를 기대할 수 있다.

> **하지만 이런 가정은 모든 분류기가 완벽하게 독립적이고 오차에 상관관계가 없어야 가능합니다. 하지만 여기서는 같은 데이터로 훈련시키기 때문에 이런 가정이 맞지 않습니다. 분류기들이 같은 종류의 오차를 만들기 쉽기 때문에 잘못된 클래스가 다수인 경우가 많고 앙상블의 정확도가 낮아집니다.**

> **TIP(교재).** 앙상블 방법은 **예측기가 가능한 한 서로 독립적일 때 최고의 성능을 발휘한다.** 다양한 분류기를 얻는 한 가지 방법은 **각기 다른 알고리즘으로 학습시키는 것**이다. 이렇게 하면 매우 다른 종류의 오차를 만들 가능성이 높기 때문에 앙상블 모델의 정확도가 향상된다.

```python
from sklearn.datasets import make_moons
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

X, y = make_moons(n_samples=500, noise=0.30, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

voting_clf = VotingClassifier(
    estimators=[
        ('lr', LogisticRegression(random_state=42)),
        ('rf', RandomForestClassifier(random_state=42)),
        ('svc', SVC(random_state=42))
    ]
)
voting_clf.fit(X_train, y_train)
```

개별 분류기와 앙상블의 테스트 정확도:

| 모델 | 정확도 |
|---|---:|
| 로지스틱 회귀 | 0.864 |
| 랜덤 포레스트 | 0.896 |
| SVC | 0.896 |
| **직접 투표 앙상블** | **0.912** |
| **간접 투표 앙상블** | **0.92** |

모든 분류기가 클래스의 확률을 예측할 수 있으면(즉, `predict_proba()` 메서드가 있으면) 개별 분류기의 예측을 **평균 내어 확률이 가장 높은 클래스**를 예측할 수 있다. 이를 **간접 투표**(soft voting)라고 한다. **이 방식은 확률이 높은 투표에 비중을 더 두기 때문에 직접 투표 방식보다 성능이 높다.**

```python
voting_clf.voting = "soft"
voting_clf.named_estimators["svc"].probability = True
voting_clf.fit(X_train, y_train)
voting_clf.score(X_test, y_test)   # 0.92
```

SVC는 기본값에서는 클래스 확률을 제공하지 않으므로 `probability` 매개변수를 `True`로 지정해야 한다(이렇게 하면 클래스 확률을 추정하기 위해 교차 검증을 사용하므로 훈련 속도가 느려진다).

---

## 7.2 배깅과 페이스팅

다양한 분류기를 만드는 또 다른 방법은 **같은 알고리즘을 사용하고 훈련 세트의 서브셋을 랜덤으로 구성하여 분류기를 각기 다르게 학습시키는 것**이다.

| 방식 | 샘플링 | 원어 |
|---|---|---|
| **배깅** | 훈련 세트에서 **중복을 허용하여** 샘플링 | bootstrap aggregating의 줄임말 |
| **페이스팅** | 중복을 허용하지 **않고** 샘플링 | pasting |

다시 말해 배깅과 페이스팅에서는 같은 훈련 샘플을 여러 개의 예측기에 걸쳐 사용할 수 있다. 하지만 **배깅만 한 예측기를 위해 같은 훈련 샘플을 여러 번 샘플링할 수 있다.**

모든 예측기가 훈련을 마치면 앙상블은 모든 예측기의 예측을 모아서 새로운 샘플에 대한 예측을 만든다. 집계 함수는 일반적으로 **분류일 때는 통계적 최빈값**(직접 투표 분류기처럼 가장 많은 예측 결과)을, **회귀에 대해서는 평균**을 계산한다.

> **개별 예측기는 원본 훈련 세트로 훈련시킨 것보다 훨씬 크게 편향되어 있지만 집계 함수를 통과하면 편향과 분산이 모두 감소합니다. 일반적으로 앙상블의 결과는 원본 데이터셋으로 하나의 예측기를 훈련시킬 때와 비교해 편향은 비슷하지만 분산은 줄어듭니다.**

예측기는 동시에 다른 CPU 코어나 서버에서 **병렬로 학습**시킬 수 있다. 예측도 병렬로 수행할 수 있다. 이런 확장성 덕분에 배깅과 페이스팅의 인기가 높다.

### 7.2.1 사이킷런의 배깅과 페이스팅

```python
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

bag_clf = BaggingClassifier(DecisionTreeClassifier(), n_estimators=500,
                            max_samples=100, n_jobs=-1, random_state=42)
bag_clf.fit(X_train, y_train)
```

- 각 분류기는 훈련 세트에서 중복을 허용하여 랜덤으로 선택된 100개의 샘플로 훈련된다(배깅). 페이스팅을 사용하려면 `bootstrap=False`로 지정한다.
- `n_jobs`는 사이킷런이 훈련과 예측에 사용할 CPU 코어 수를 지정한다. `-1`로 지정하면 가용한 모든 코어를 사용한다.

> **NOTE(교재).** `BaggingClassifier`는 기반이 되는 분류기가 결정 트리 분류기처럼 클래스 확률을 추정할 수 있으면(즉, `predict_proba()` 함수가 있으면) 직접 투표 대신 자동으로 **간접 투표 방식**을 사용한다.

그림 7-5에서 단일 결정 트리와 500개 트리 배깅 앙상블을 비교하면, **앙상블의 예측이 결정 트리 하나의 예측보다 일반화가 훨씬 잘 된 것 같다. 앙상블은 비슷한 편향에서 더 작은 분산을 만든다**(훈련 세트의 오차 수가 거의 동일하지만 결정 경계는 덜 불규칙하다).

배깅은 각 예측기가 학습하는 서브셋에 다양성을 추가하므로 **배깅이 페이스팅보다 편향이 조금 더 높다.** 하지만 다양성을 추가한다는 것은 예측기들의 상관관계를 줄이므로 앙상블의 분산이 줄어든다는 것을 의미한다. **전반적으로 배깅이 더 나은 모델을 만들기 때문에 일반적으로 더 선호된다.** 그러나 시간과 CPU 파워에 여유가 있다면 교차 검증으로 배깅과 페이스팅을 모두 평가해서 더 나은 쪽을 선택하는 것이 좋다.

### 7.2.2 OOB 평가

배깅을 사용하면 어떤 샘플은 한 예측기를 위해 여러 번 샘플링되고 어떤 것은 전혀 선택되지 않을 수 있다. `BaggingClassifier`는 기본값으로 중복을 허용하여(`bootstrap=True`) 훈련 세트의 크기만큼인 m개 샘플을 선택한다. **이는 평균적으로 각 예측기에 훈련 샘플의 63% 정도만 샘플링된다는 것을 의미한다**(m이 커지면 이 비율은 `1 − exp(−1) ≈ 63.212%`에 가까워진다). 선택되지 않은 나머지 37%를 **OOB(out-of-bag) 샘플**이라고 부른다. 예측기마다 남겨진 37%는 모두 다르다.

예측기가 훈련되는 동안에는 OOB 샘플을 사용하지 않으므로 **별도의 검증 세트를 사용하지 않고 OOB 샘플을 사용해 평가할 수 있다.**

```python
bag_clf = BaggingClassifier(DecisionTreeClassifier(), n_estimators=500,
                            oob_score=True, n_jobs=-1, random_state=42)
bag_clf.fit(X_train, y_train)
bag_clf.oob_score_          # 0.896

from sklearn.metrics import accuracy_score
y_pred = bag_clf.predict(X_test)
accuracy_score(y_test, y_pred)   # 0.912
```

테스트 세트에서 0.92의 정확도를 얻었다. **OOB 평가는 2% 이상 낮아 조금 비관적이었다.**

OOB 샘플에 대한 결정 함수의 값도 `oob_decision_function_` 변수에서 확인할 수 있다.

> **모빌리티 적용.** OOB 평가는 편리하지만 **시계열 데이터에는 쓰면 안 된다.** OOB 샘플은 시간과 무관하게 랜덤 추출된 것이라 미래 정보가 섞인다. ETA 프로젝트에서는 반드시 날짜 기준 검증 세트를 쓴다.

---

## 7.3 랜덤 패치와 랜덤 서브스페이스

`BaggingClassifier`는 **특성 샘플링**도 지원한다. 샘플링은 `max_features`, `bootstrap_features` 두 매개변수로 조절된다. 따라서 각 예측기는 랜덤으로 선택한 입력 특성의 일부분으로 훈련된다.

이 기법은 훈련 속도를 크게 높일 수 있기 때문에 특히 (이미지와 같은) 매우 고차원의 데이터셋을 다룰 때 유용하다.

| 방식 | 조건 |
|---|---|
| **랜덤 패치 방식** | 훈련 특성과 샘플을 **모두 샘플링** |
| **랜덤 서브스페이스 방식** | 훈련 샘플을 모두 사용하고(`bootstrap=False`이고 `max_samples=1.0`) **특성만 샘플링**(`bootstrap_features=True` 그리고/또는 `max_features`는 1.0보다 작게 설정) |

**특성 샘플링은 더 다양한 예측기를 만들며 편향을 늘리는 대신 분산을 낮춘다.**

---

## 7.4 랜덤 포레스트

**랜덤 포레스트**는 일반적으로 배깅 방법(또는 페이스팅)을 적용한 **결정 트리의 앙상블**이다. 일반적으로 `max_samples`를 훈련 세트의 크기로 지정한다. `BaggingClassifier`에 `DecisionTreeClassifier`를 넣어 만드는 대신 결정 트리에 최적화되어 사용하기 편리한 `RandomForestClassifier`를 사용할 수 있다(회귀 문제를 위한 클래스로는 `RandomForestRegressor`가 있다).

```python
from sklearn.ensemble import RandomForestClassifier

rnd_clf = RandomForestClassifier(n_estimators=500, max_leaf_nodes=16,
                                 n_jobs=-1, random_state=42)
rnd_clf.fit(X_train, y_train)
y_pred_rf = rnd_clf.predict(X_test)
```

**랜덤 포레스트 알고리즘은 트리의 노드를 분할할 때 전체 특성 중에서 최선의 특성을 찾는 대신 랜덤으로 선택한 특성 후보 중에서 최적의 특성을 찾는 식으로 무작위성을 더 주입한다.** 기본적으로 √n개의 특성을 선택한다(n은 전체 특성 개수). 이는 결국 **트리를 더욱 다양하게 만들고 편향을 손해 보는 대신 분산을 낮추어 전체적으로 더 훌륭한 모델**을 만들어낸다.

다음은 `BaggingClassifier`를 앞의 `RandomForestClassifier`와 거의 동일하게 만든 것이다.

```python
bag_clf = BaggingClassifier(
    DecisionTreeClassifier(max_features="sqrt", max_leaf_nodes=16),
    n_estimators=500, n_jobs=-1, random_state=42)
```

### 7.4.1 엑스트라 트리

랜덤 포레스트에서 트리를 만들 때 각 노드는 랜덤으로 특성의 서브셋을 만들어 분할에 사용한다. 트리를 더욱 랜덤하게 만들기 위해 (보통의 결정 트리처럼) **최적의 임곗값을 찾는 대신 후보 특성을 사용해 랜덤으로 분할한 다음 그중에서 최상의 분할을 선택한다.** `DecisionTreeClassifier`를 만들 때 `splitter="random"`으로 지정하기만 하면 된다.

이와 같이 극단적으로 랜덤한 트리의 랜덤 포레스트를 **익스트림 랜덤 트리 앙상블**(또는 줄여서 **엑스트라 트리**)이라고 부른다. **여기서도 역시 편향이 늘어나는 대신 분산이 낮아진다.** 모든 노드에서 특성마다 가장 최적의 임곗값을 찾는 것은 트리 알고리즘에서 가장 시간이 많이 소요되는 작업이므로 **일반적인 랜덤 포레스트보다 엑스트라 트리의 훈련 속도가 훨씬 빠르다.**

`ExtraTreesClassifier`는 `bootstrap` 매개변수가 기본적으로 `False`인 것을 제외하고 사용법은 `RandomForestClassifier`와 같다.

> **TIP(교재).** `RandomForestClassifier`가 `ExtraTreesClassifier`보다 더 나을지 혹은 나쁠지 예단하긴 어렵다. **둘 다 시도해보고 교차 검증으로 비교해보는 것이 유일한 방법이다.**

### 7.4.2 특성 중요도

랜덤 포레스트의 또 다른 장점은 **특성의 상대적 중요도를 측정하기 쉽다**는 점이다. 사이킷런은 어떤 특성을 사용한 노드가 (랜덤 포레스트에 있는 모든 트리에 걸쳐서) 평균적으로 **불순도를 얼마나 감소시키는지** 확인하여 특성의 중요도를 측정한다. 더 정확하게는 가중치 평균이며, 각 노드의 가중치는 연관된 훈련 샘플 수와 같다.

사이킷런은 훈련이 끝난 뒤 특성마다 자동으로 이 점수를 계산하고 중요도의 전체 합이 1이 되도록 결괏값을 정규화한다. 이 값은 `feature_importances_` 변수에 저장되어 있다.

```python
from sklearn.datasets import load_iris

iris = load_iris(as_frame=True)
rnd_clf = RandomForestClassifier(n_estimators=500, random_state=42)
rnd_clf.fit(iris.data, iris.target)
for score, name in zip(rnd_clf.feature_importances_, iris.data.columns):
    print(round(score, 2), name)

# 0.11 sepal length (cm)
# 0.02 sepal width (cm)
# 0.44 petal length (cm)
# 0.42 petal width (cm)
```

가장 중요한 특성은 꽃잎의 길이(44%)와 너비(42%)이고 꽃받침의 길이와 너비는 비교적 덜 중요한 것으로 나타난다(각각 11%와 2%).

**랜덤 포레스트는 특히 특성을 선택해야 할 때 어떤 특성이 중요한지 빠르게 확인할 수 있어 매우 편리하다.**

> **모빌리티 적용.** 프로젝트 14주차 오차 분석에서 `feature_importances_`로 "무엇이 ETA를 좌우하는가"를 정리한다. 다만 이 값은 **불순도 감소 기반**이라 고차원 범주형 특성(정류장 ID 원-핫)에 유리하게 편향된다는 점을 짚어야 한다. 보완책으로 순열 중요도를 함께 본다.

---

## 7.5 부스팅

**부스팅**(원래는 가설 부스팅이라 불렸다)은 **약한 학습기를 여러 개 연결하여 강한 학습기를 만드는 앙상블 방법**을 말한다. 부스팅 방법의 아이디어는 **앞의 모델을 보완해 나가면서 일련의 예측기를 학습시키는 것**이다. 가장 인기 있는 것은 **AdaBoost**(adaptive boosting의 줄임말)와 **그레이디언트 부스팅**이다.

### 7.5.1 AdaBoost

이전 예측기를 보완하는 새로운 예측기를 만드는 방법은 **이전 모델이 과소적합했던 훈련 샘플의 가중치를 더 높이는 것**이다. 이렇게 하면 새로운 예측기는 학습하기 어려운 샘플에 점점 더 맞춰지게 된다.

1. 첫 번째 분류기(예: 결정 트리)를 훈련 세트에서 훈련시키고 예측을 만든다.
2. 알고리즘이 잘못 분류된 훈련 샘플의 가중치를 상대적으로 높인다.
3. 두 번째 분류기는 업데이트된 가중치를 사용해 훈련 세트에서 훈련하고 다시 예측을 만든다.
4. 다시 가중치를 업데이트하는 방식으로 계속된다.

**그림에서 볼 수 있듯이 이런 연속된 학습 기법은 경사 하강법과 비슷한 면이 있다. 경사 하강법은 비용 함수를 최소화하기 위해 한 예측기의 모델 파라미터를 조정해가는 반면 AdaBoost는 점차 더 좋아지도록 앙상블에 예측기를 추가한다.**

**[식 7-1] j번째 예측기의 가중치가 적용된 오류율**

```
r_j = Σ_{ŷ_j⁽ⁱ⁾ ≠ y⁽ⁱ⁾} w⁽ⁱ⁾ / Σ_{i=1}^{m} w⁽ⁱ⁾
```

**[식 7-2] 예측기 가중치**

```
α_j = η log((1 − r_j) / r_j)
```

η는 학습률 하이퍼파라미터이며 기본값은 1이다. **예측기가 정확할수록 가중치는 더 높아진다.** 만약 랜덤 추측이라면 가중치는 0에 가까워진다. 그보다 나쁘면(즉, 랜덤 추측보다 정확도가 낮으면) 가중치는 음수가 된다.

**[식 7-3] 가중치 업데이트 규칙**

```
w⁽ⁱ⁾ ← w⁽ⁱ⁾              (ŷ_j⁽ⁱ⁾ = y⁽ⁱ⁾일 때)
w⁽ⁱ⁾ ← w⁽ⁱ⁾ exp(α_j)     (ŷ_j⁽ⁱ⁾ ≠ y⁽ⁱ⁾일 때)
```

그런 다음 모든 샘플의 가중치를 정규화한다.

**[식 7-4] AdaBoost 예측**

```
ŷ(x) = argmax_k Σ_{j=1, ŷ_j(x)=k}^{N} α_j       (N은 예측기 수)
```

> **CAUTION(교재).** 연속된 학습 기법에는 중요한 단점이 하나 있다. **각 예측기는 이전 예측기가 훈련되고 평가된 후에 학습될 수 있기 때문에 훈련을 병렬화할 수 없다.** 결국 배깅이나 페이스팅만큼 확장성이 높지 않다.

```python
from sklearn.ensemble import AdaBoostClassifier

ada_clf = AdaBoostClassifier(
    DecisionTreeClassifier(max_depth=1), n_estimators=30,
    learning_rate=0.5, random_state=42)
ada_clf.fit(X_train, y_train)
```

여기에서 사용하는 결정 트리는 `max_depth=1`이다. 다시 말해 결정 노드 하나와 리프 노드 두 개로 이루어진 트리다. 이 트리가 `AdaBoostClassifier`의 기본 추정기다.

사이킷런은 **SAMME**라는 AdaBoost의 다중 클래스 버전을 사용한다. 클래스가 두 개뿐이라면 SAMME는 AdaBoost와 동일하다. 예측기가 클래스의 확률을 추정할 수 있다면 사이킷런은 SAMME의 변형인 **SAMME.R**(끝의 R은 'Real'을 뜻한다)을 사용한다.

> **TIP(교재).** AdaBoost 앙상블이 훈련 세트에 과대적합되면 **추정기 수를 줄이거나 추정기의 규제를 더 강하게** 해보세요.

### 7.5.2 그레이디언트 부스팅

AdaBoost처럼 그레이디언트 부스팅은 앙상블에 이전까지의 오차를 보정하도록 예측기를 순차적으로 추가한다. 하지만 AdaBoost처럼 반복마다 샘플의 가중치를 수정하는 대신 **이전 예측기가 만든 잔여 오차에 새로운 예측기를 학습시킨다.**

결정 트리를 기반 예측기로 사용하는 것을 **그레이디언트 트리 부스팅** 또는 **그레이디언트 부스티드 회귀 트리(GBRT)**라고 한다.

```python
import numpy as np
from sklearn.tree import DecisionTreeRegressor

np.random.seed(42)
X = np.random.rand(100, 1) - 0.5
y = 3 * X[:, 0] ** 2 + 0.05 * np.random.randn(100)   # y = 3x² + 가우스_잡음

tree_reg1 = DecisionTreeRegressor(max_depth=2, random_state=42)
tree_reg1.fit(X, y)

# 첫 번째 예측기에서 생긴 잔여 오차에 두 번째 회귀 모델을 훈련
y2 = y - tree_reg1.predict(X)
tree_reg2 = DecisionTreeRegressor(max_depth=2, random_state=43)
tree_reg2.fit(X, y2)

# 두 번째 예측기가 만든 잔여 오차에 세 번째 회귀 모델을 훈련
y3 = y2 - tree_reg2.predict(X)
tree_reg3 = DecisionTreeRegressor(max_depth=2, random_state=44)
tree_reg3.fit(X, y3)

# 새로운 샘플에 대한 예측 = 모든 트리의 예측을 더한 것
X_new = np.array([[-0.4], [0.], [0.5]])
sum(tree.predict(X_new) for tree in (tree_reg1, tree_reg2, tree_reg3))
# array([0.49484029, 0.04021166, 0.75026781])
```

**트리가 앙상블에 추가될수록 앙상블의 예측이 점차 좋아지는 것을 알 수 있다.**

```python
from sklearn.ensemble import GradientBoostingRegressor

gbrt = GradientBoostingRegressor(max_depth=2, n_estimators=3,
                                 learning_rate=1.0, random_state=42)
gbrt.fit(X, y)
```

`learning_rate` 매개변수가 각 트리의 기여도를 조절한다. 이를 0.05처럼 낮게 설정하면 앙상블을 훈련 세트에 학습시키기 위해 많은 트리가 필요하지만 일반적으로 예측의 성능은 좋아진다. **이는 축소(shrinkage)라고 부르는 규제 방법이다.**

**최적의 트리 개수 찾기.** `GridSearchCV`나 `RandomizedSearchCV`를 사용할 수도 있지만 더 간단한 방법이 있다. `n_iter_no_change` 하이퍼파라미터를 정숫값(예: 10)으로 설정하면 훈련 중에 마지막 10개의 트리가 도움이 되지 않는 경우 `GradientBoostingRegressor`가 트리 추가를 자동으로 중지한다. **이것은 (4장에서 소개한) 단순한 조기 종료 기법이지만 약간의 인내심을 가지고 몇 번의 반복에서 진전이 없는 것을 확인한 후 중지한다.**

```python
gbrt_best = GradientBoostingRegressor(
    max_depth=2, learning_rate=0.05, n_estimators=500,
    n_iter_no_change=10, random_state=42)
gbrt_best.fit(X, y)
gbrt_best.n_estimators_   # 92
```

`n_iter_no_change`를 너무 낮게 설정하면 훈련이 너무 일찍 중단되어 모델이 과소적합될 수 있고, 너무 높게 설정하면 오히려 과대적합된다.

`GradientBoostingRegressor`는 각 트리가 훈련할 때 사용할 훈련 샘플의 비율을 지정할 수 있는 `subsample` 매개변수도 지원한다. `subsample=0.25`라고 하면 각 트리는 랜덤으로 선택된 25%의 훈련 샘플로 학습된다. **편향이 높아지는 대신 분산이 낮아지게 된다. 또한 훈련 속도도 상당히 빨라진다.** 이런 기법을 **확률적 그레이디언트 부스팅**이라고 한다.

### 7.5.3 히스토그램 기반 그레이디언트 부스팅

사이킷런은 대규모 데이터셋에 최적화된 또 다른 GBRT 구현인 **히스토그램 기반 그레이디언트 부스팅(HGB)**도 제공한다. 이 알고리즘은 **입력 특성을 구간으로 나누어 정수로 대체하는 방식**으로 작동한다. 구간의 개수는 `max_bins` 하이퍼파라미터에 의해 제어되며 기본값은 255이고 이보다 높게 설정할 수 없다.

구간 분할을 사용하면 학습 알고리즘이 평가해야 하는 가능한 임곗값의 수를 크게 줄일 수 있고, 정수로 작업하면 더 빠르고 메모리 효율적인 데이터 구조를 사용할 수 있다. 그리고 구간을 분할하는 방식 덕분에 **각 트리를 학습할 때 특성을 정렬할 필요가 없다.**

그 결과 이 구현의 계산 복잡도는 `O(n × m × log(m))`이 아닌 **`O(b × m)`**이며, 여기서 b는 구간의 개수, m은 훈련 샘플의 개수, n은 특성의 개수다. **이는 실제로 HGB가 대규모 데이터셋에서 일반 GBRT보다 수백 배 빠르게 훈련할 수 있다는 것을 의미한다.**

그러나 **구간 분할은 규제처럼 작동해 정밀도 손실을 유발하므로 데이터셋에 따라 과대적합을 줄이는 데 도움이 될 수도 있고 과소적합을 유발할 수도 있다.**

`HistGradientBoostingRegressor`와 `HistGradientBoostingClassifier`가 `GradientBoosting*`와 다른 점:

- 인스턴스 수가 10,000개보다 많으면 **조기 종료가 자동으로 활성화**된다. `early_stopping` 매개변수를 `True` 또는 `False`로 설정하여 조기 종료를 항상 켜거나 끌 수 있다.
- `subsample` 매개변수가 지원되지 않는다.
- `n_estimators` 매개변수가 `max_iter`로 바뀌었다.
- 조정할 수 있는 결정 트리 하이퍼파라미터는 `max_leaf_nodes`, `min_samples_leaf`, `max_depth`뿐이다.

**HGB 클래스는 범주형 특성과 누락된 값을 지원한다. 이로 인해 전처리가 상당히 간소화된다.** 그러나 범주형 특성은 0 ~ `max_bins` 사이의 정수로 표현돼야 한다. 이를 위해 `OrdinalEncoder`를 사용할 수 있다.

```python
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import OrdinalEncoder

hgb_reg = make_pipeline(
    make_column_transformer((OrdinalEncoder(), ["ocean_proximity"]),
                            remainder="passthrough"),
    HistGradientBoostingRegressor(categorical_features=[0], random_state=42)
)
hgb_reg.fit(housing, housing_labels)
```

**전체 파이프라인이 임포트 구문만큼 짧다!** 누락된 값을 채우고, 스케일을 조정하고, 원-핫 인코딩을 처리하지 않아도 되므로 정말 편리하다. **하이퍼파라미터 튜닝 없이도 이 모델은 약 47,600의 RMSE를 산출하는데 이는 그리 나쁘지 않은 수치다.**

> **TIP(교재).** 파이썬 ML 생태계에는 최적화된 그레이디언트 부스팅 구현이 몇 가지 더 있는데, 특히 **XGBoost**(`https://github.com/dmlc/xgboost`), **CatBoost**(`https://catboost.ai/`), **LightGBM**(`https://lightgbm.readthedocs.io/`)이 잘 알려져 있다. 이러한 라이브러리는 몇 년 전부터 사용되어 왔다. 모두 그레이디언트 부스팅에 특화되어 있고, API가 사이킷런과 매우 유사하며, **GPU 가속을 비롯한 다양한 추가 기능을 제공**하므로 꼭 확인해보기 바란다!

> **모빌리티 적용.** 이 TIP이 프로젝트에서 XGBoost를 쓰는 근거이자, 2장의 무거운 전처리 파이프라인 없이도 **HGB 한 줄로 강력한 기준 모델**을 만들 수 있다는 실용적 요령이다. 5주차 실습은 다음 순서를 권한다.
>
> 1. 중앙값 기준 모델
> 2. `RandomForestRegressor`
> 3. `HistGradientBoostingRegressor` (범주형·결측 그대로)
> 4. `XGBRegressor` (같은 데이터, 같은 분할)

---

## 7.6 스태킹

마지막 앙상블 모델은 **스태킹**(stacked generalization의 줄임말)이다. 이는 **'앙상블에 속한 모든 예측기의 예측을 취합하는 (직접 투표 같은) 간단한 함수를 사용하는 대신 취합하는 모델을 훈련시킬 수 없을까?'**라는 기본 아이디어에서 출발한다.

그림 7-11에서 아래의 세 예측기는 각각 다른 값(3.1, 2.7, 2.9)을 예측하고 마지막 예측기(**블렌더** 또는 **메타 학습기**)가 이 예측을 입력으로 받아 최종 예측(3.0)을 만든다.

### 블렌더 훈련 방법

1. 앙상블의 모든 예측기에서 `cross_val_predict()`를 사용하여 원본 훈련 세트에 있는 각 샘플에 대한 **표본 외 예측**을 얻는다.
2. 이를 블렌더를 훈련하기 위한 입력 특성으로 사용하고 타깃은 원본 훈련 세트에서 간단히 복사한다.
3. 원본 훈련 세트의 특성 개수에 관계없이 **블렌딩 훈련 세트에는 예측기당 하나의 입력 특성이 포함된다.**
4. 블렌더가 학습되면 기본 예측기는 전체 원본 훈련 세트로 마지막에 한 번 더 재훈련된다.

여러 가지 블렌더(예: 선형 회귀를 사용하는 블렌더, 랜덤 포레스트 회귀를 사용하는 블렌더)를 이런 방식으로 훈련하여 전체 블렌더 계층을 얻은 다음 그 위에 다른 블렌더를 추가하여 최종 예측을 생성하는 것이 가능하다. 이렇게 하면 성능을 조금 더 끌어올릴 수 있지만 **훈련 시간과 시스템 복잡성 측면에서 비용이 증가한다.**

```python
from sklearn.ensemble import StackingClassifier

stacking_clf = StackingClassifier(
    estimators=[
        ('lr', LogisticRegression(random_state=42)),
        ('rf', RandomForestClassifier(random_state=42)),
        ('svc', SVC(probability=True, random_state=42))
    ],
    final_estimator=RandomForestClassifier(random_state=43),
    cv=5   # 교차 검증 폴드 개수
)
stacking_clf.fit(X_train, y_train)
```

각 예측기에 대해 스태킹 분류기는 사용 가능한 경우 `predict_proba()`를 호출하고, 그렇지 않은 경우 `decision_function()`을 사용하거나 최후의 수단으로 `predict()`를 호출한다. 최종 예측기를 제공하지 않으면 `StackingClassifier`는 `LogisticRegression`을, `StackingRegressor`는 `RidgeCV`를 사용한다.

테스트 세트에서 이 스태킹 모델을 평가하면 **92.8%의 정확도**를 얻을 수 있으며, 이는 92%를 얻은 간접 투표 방식의 분류기보다 약간 더 나은 결과다.

### 이 장의 결론 (교재 원문)

> 결론적으로 앙상블 방법은 다재다능하고 강력하며 사용법이 매우 간단합니다. **랜덤 포레스트, AdaBoost, GBRT는 대부분의 머신러닝 작업에서 가장 먼저 테스트해야 하는 모델이며, 특히 서로 다른 종류로 구성된 표 형식 데이터에서 빛을 발합니다.** 또한 전처리가 거의 필요하지 않기 때문에 프로토타입을 빠르게 구축하는 데 적합합니다. 마지막으로 투표 기반 분류기와 스태킹 분류기 같은 앙상블 방법은 시스템 성능을 한계까지 끌어올리는 데 도움이 될 수 있습니다.

이 문단이 우리 커리큘럼에서 5주차를 "프로젝트 기준 모델을 확정하는 주"로 삼는 이유다. BIS 데이터는 정확히 "서로 다른 종류로 구성된 표 형식 데이터"다.

---

## 연습문제 (교재 원문)

1. 정확히 같은 훈련 데이터로 다섯 개의 다른 모델을 훈련시켜서 모두 95% 정확도를 얻었다면 이 모델들을 연결하여 더 좋은 결과를 얻을 수 있을까요? 가능하다면 어떻게 해야 할까요? 그렇지 않다면 왜일까요?
2. 직접 투표와 간접 투표 분류기의 차이점은 무엇일까요?
3. 배깅 앙상블의 훈련을 여러 대의 서버에 분산시켜 속도를 높일 수 있을까요? 페이스팅 앙상블, 부스팅 앙상블, 랜덤 포레스트, 스태킹 앙상블의 경우는 어떨까요?
4. OOB 평가의 장점은 무엇인가요?
5. 무엇이 엑스트라 트리 앙상블을 일반 랜덤 포레스트보다 더 랜덤하게 만드나요? 추가적인 무작위성이 어떻게 도움이 될까요? 엑스트라 트리 분류기는 일반 랜덤 포레스트보다 느릴까요, 빠를까요?
6. AdaBoost 앙상블이 훈련 데이터에 과소적합되었다면 어떤 매개변수를 어떻게 바꾸어야 할까요?
7. 그레이디언트 부스팅 앙상블이 훈련 데이터에 과대적합되었다면 학습률을 높여야 할까요, 낮춰야 할까요?
8. (3장에서 소개한) MNIST 데이터를 불러들여 훈련 세트, 검증 세트, 테스트 세트로 나눕니다(예: 훈련에 50,000개 샘플, 검증에 10,000개 샘플, 테스트에 10,000개 샘플). 그런 다음 랜덤 포레스트 분류기, 엑스트라 트리 분류기, SVM 분류기 같은 여러 종류의 분류기를 훈련시킵니다. 그리고 검증 세트에서 개별 분류기보다 더 높은 성능을 내도록 이들을 간접 또는 직접 투표 방법을 사용해 앙상블로 연결해보세요.
9. 이전 연습문제의 각 분류기를 실행해서 검증 세트에서 예측을 만들고 그 결과로 새로운 훈련 세트를 만들어보세요. … 축하합니다, 방금 블렌더를 훈련시켰습니다! 그리고 이 분류기를 모아서 스태킹 앙상블을 구성했습니다!

---

## 수업용 정리

| 개념 | 5주차에서 사용하는 형태 | 프로젝트로 |
|---|---|---|
| 대중의 지혜, 큰 수의 법칙 | 51% 동전 예시 | 왜 앙상블인가 |
| 직접 투표 vs 간접 투표 | 표 하나 | — |
| 배깅 / 페이스팅 | 중복 허용 여부 | — |
| OOB 평가 | 63% / 37% | **시계열에는 쓰지 않는다** |
| 랜덤 포레스트 = 배깅 + 특성 무작위성 | √n 특성 | 기준 모델 2 |
| 엑스트라 트리 | 임곗값도 랜덤 | 비교용 |
| 특성 중요도 | `feature_importances_` | 오차 분석 |
| AdaBoost | 샘플 가중치 | 개념만 |
| 그레이디언트 부스팅 | **잔여 오차에 학습** | 핵심 |
| 축소(learning_rate) | 규제 | 튜닝 대상 |
| 조기 종료(`n_iter_no_change`) | 트리 개수 자동 결정 | 실습 |
| HGB | 구간 분할, 범주형·결측 지원 | 기준 모델 3 |
| XGBoost / LightGBM / CatBoost | 교재 TIP | 프로젝트 주력 |
| 스태킹 | 블렌더 훈련 | 여유 있으면 |

## 확인 문제

1. 앙상블이 개별 모델보다 나으려면 개별 모델들이 어떤 조건을 만족해야 하는가?
2. 배깅에서 OOB 샘플이 평균 37%가 되는 이유는 무엇인가?
3. AdaBoost와 그레이디언트 부스팅이 "이전 오차를 보완"하는 방식은 어떻게 다른가?
4. `learning_rate`를 낮추면 왜 트리가 더 많이 필요한가?
5. 부스팅은 왜 배깅처럼 병렬화할 수 없는가?
6. 시계열 ETA 데이터에서 OOB 점수를 검증 지표로 쓰면 안 되는 이유는 무엇인가?

## 관련 자료

- [Ch6 — 결정 트리](week05_ch06.md)
- [Ch2 — 머신러닝 프로젝트 처음부터 끝까지](week03_ch02.md)
- [주차별 일정](../../schedule.md)
- [공식 실습 코드](https://github.com/rickiepark/handson-ml3)
